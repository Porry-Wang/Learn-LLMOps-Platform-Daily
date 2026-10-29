#!/usr/bin/env python3
"""Day 3 · 调试入口：把 all-reduce 拆成能单步看的三段。

为什么需要它：
    `dist.py` 必须由 torchrun 拉起 4 个子进程。PyCharm 的断点打在父进程上，
    子进程里的断点不一定停得下来。想单步看清 all-reduce 到底做了什么，
    先在**单进程**里把它演一遍——语义完全一样，只是由一个进程分饰四角。

三段，从假到真：

    段 1  纯 torch 模拟 4 个 rank 的 all-reduce
          可断点、可单步。看清楚「每个 rank 手里有什么 → 汇总 → 每人拿回同一份」。
    段 2  真的调用 dist.init_process_group，但 world_size=1
          看清楚环境变量从哪来、API 的调用顺序、以及「只有自己时 all-reduce 是空转」。
    段 3  只打印真 4 进程的运行命令，不执行。

用法：
    PyCharm 里右键本文件 → Debug 'debug_dist'
    建议断点：run_simulated_all_reduce() 里 `total += before[rank]` 那一行，
              单步 3 次，看 total 从 [1,1,1,1] 一路加到 [10,10,10,10]。

    终端直接跑：
        ../../.venv/bin/python debug_dist.py

失真边界：
    段 1 是模拟，不走 gloo，验证不了通信本身（连接、超时、带宽）。
    段 2 走真 gloo，但 world_size=1，验证不了「多个进程互相等」。
    真要验证协同，还是得跑段 3 那条命令看 dist.py 的 4 行输出。
"""

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

WORLD_SIZE = 4          # 段 1 模拟的 rank 数，和 dist.py 的 --nproc_per_node=4 对齐
MASTER_PORT = "29517"   # 段 2 单机单进程用，避开 torchrun 常用的 29400 段


# ---------------------------------------------------------------------------
# 段 1：单进程分饰四角，把 all-reduce 演一遍
# ---------------------------------------------------------------------------

def run_simulated_all_reduce() -> None:
    print("=" * 64)
    print("段 1 · 模拟 4 个 rank 的 all-reduce（可断点单步）")
    print("=" * 64)

    # dist.py 里是 `t = torch.ones(4) * (rank + 1)`，4 个进程各拿到一个 rank。
    # 这里一个进程把 4 份都造出来，放进列表，before[r] 就是 rank r 手里的张量。
    before = [torch.ones(4) * (rank + 1) for rank in range(WORLD_SIZE)]
    for rank in range(WORLD_SIZE):
        print(f"  rank {rank} 手里：{before[rank].tolist()}")

    # all-reduce(SUM) 的动作：把所有 rank 的张量按元素相加。
    # ↓↓↓ 断点打在这一行，然后单步 3 次，盯着 total 看 ↓↓↓
    total = before[0].clone()
    for rank in range(1, WORLD_SIZE):
        total = total + before[rank]
        print(f"  加上 rank {rank} 之后 total = {total.tolist()}")

    # all-reduce 和 reduce 的全部区别就在这一行：
    # reduce 只把结果给 rank 0，all-reduce 把结果发回给**每一个** rank。
    after = [total.clone() for _ in range(WORLD_SIZE)]

    print()
    for rank in range(WORLD_SIZE):
        print(f"  [rank {rank}/{WORLD_SIZE}] "
              f"before={before[rank].tolist()} after={after[rank].tolist()}")

    # 训练里真正用来更新参数的是平均值，不是总和。
    # dist.all_reduce 只负责 SUM，除以 world_size 是 DDP 自己做的第二步。
    mean_grad = total / WORLD_SIZE
    print(f"\n  DDP 用的平均梯度 = {mean_grad.tolist()}   （10 / 4 = 2.5）")
    print("  4 个 rank 用同一份平均梯度更新 → 4 份模型不会分叉，这就是 DP。")


# ---------------------------------------------------------------------------
# 段 2：真的建一次进程组，但只有自己
# ---------------------------------------------------------------------------

def run_real_single_process() -> None:
    print()
    print("=" * 64)
    print("段 2 · 真调用 dist，world_size=1（看清 API 与环境变量）")
    print("=" * 64)

    # dist.py 里这些变量是 torchrun 注入的；这里自己写，才能不经 torchrun 直接跑。
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = MASTER_PORT

    print("  自己注入的环境变量：")
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        print(f"    {key:<16} = {os.environ[key]}")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # gloo = CPU / 以太网后端，macOS 上唯一可用（没有 NCCL）。
    # 这一行是集合操作：world_size 个进程全到齐才返回。这里只有 1 个，所以立刻返回。
    dist.init_process_group(backend="gloo")
    print(f"\n  init_process_group 返回了：dist.get_rank()={dist.get_rank()} "
          f"dist.get_world_size()={dist.get_world_size()}")
    assert dist.get_rank() == rank and dist.get_world_size() == world_size

    # 拿 rank 2 的那份数据试：只有自己的时候，SUM 之后还是自己。
    t = torch.ones(4) * 3
    before = t.tolist()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    print(f"  all_reduce 前 {before} → 后 {t.tolist()}")
    print("  没变。因为组里只有一个人，没人可加——")
    print("  这正说明段 1 里那个「加上其他 rank」的动作，是真由通信完成的。")

    dist.barrier()
    dist.destroy_process_group()
    print("  destroy_process_group 完成")


# ---------------------------------------------------------------------------
# 段 3：真 4 进程怎么跑
# ---------------------------------------------------------------------------

def print_real_command() -> None:
    here = Path(__file__).resolve().parent
    print()
    print("=" * 64)
    print("段 3 · 想看真的 4 个进程互相等，跑这条（不在本脚本里执行）")
    print("=" * 64)
    print(f"""
  cd {here}
  ../../.venv/bin/python torchrun_local.py --standalone --nproc_per_node=4 dist.py

  预期 4 行，每行 after 都是 [10.0, 10.0, 10.0, 10.0]。

  想给 4 进程版本打断点：PyCharm → Settings → Build, Execution, Deployment
  → Python Debugger → 勾上 "Attach to subprocess automatically while debugging"，
  再建一个 Python run config：
      script path     = {here / "torchrun_local.py"}
      parameters      = --standalone --nproc_per_node=4 dist.py
      working dir     = {here}
      interpreter     = 项目 .venv
  然后 Debug 那个 config。子进程能否停下取决于 PyCharm 版本，
  停不下来就退回段 1 看语义、用打印看 4 进程结果。
""")


def main() -> None:
    run_simulated_all_reduce()
    run_real_single_process()
    print_real_command()
    return 0


if __name__ == "__main__":
    sys.exit(main())
