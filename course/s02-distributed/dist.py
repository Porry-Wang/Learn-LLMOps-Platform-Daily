#!/usr/bin/env python3
"""S2：4 rank 的 gloo all-reduce 最小闭环。

它回答三个问题：
    1. torchrun 给每个训练进程注入了什么？
    2. rank / world_size / local_rank 分别是什么？
    3. all-reduce 之后，为什么每个 rank 都拿到同一份结果？

运行（在本目录）：
    ../../.venv/bin/python torchrun_local.py --standalone --nproc_per_node=4 dist.py

预期：4 行，每行 after 都是 [10.0, 10.0, 10.0, 10.0]。
（rank 0 的 before 是 1，rank 1 是 2，rank 2 是 3，rank 3 是 4；
  1+2+3+4 = 10，求和后每个 rank 都持有总和。）
"""

import os

import torch
import torch.distributed as dist

# 改动实验：把 SUM 改成 MAX，先写下你的预测，再运行。
# MAX 之后 4 个 rank 的 after 会变成什么？为什么不再需要“加法”？
REDUCE_OP = dist.ReduceOp.SUM


def main() -> None:
    # 1) 这些环境变量不是我们手写的，是 torchrun 启动子进程前注入的。
    #    真实平台（或 K8s Training Operator）也是用同样的方式把 rank 告诉进程。
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # 2) gloo 是 CPU / 以太网后端；macOS 没有 NCCL，只能用 gloo 做教学实验。
    #    init_method 默认是 env://，会读 MASTER_ADDR / MASTER_PORT。
    dist.init_process_group(backend="gloo")

    # 再问一次库里的值，确认它和环境变量一致。
    # 真实训练里应当以 dist.get_rank() 为准，不要自己猜编号。
    assert dist.get_rank() == rank
    assert dist.get_world_size() == world_size

    # 3) 每个 rank 放一个不同的数：rank 0 -> 1，rank 1 -> 2，……
    #    这模拟“每张卡各有一份梯度”，all-reduce 把它们汇总。
    t = torch.ones(4) * (rank + 1)
    before = t.tolist()

    dist.all_reduce(t, op=REDUCE_OP)

    # 4) all-reduce 的语义：每个 rank 都得到同一份归约结果。
    #    不是只发给 rank 0；不是把结果平均后才发。
    print(
        f"[rank {rank}/{world_size} local_rank={local_rank}] "
        f"before={before} after={t.tolist()}",
        flush=True,
    )

    # 只让 rank 0 打印注入的变量，避免 4 份重复。
    if rank == 0:
        print("\n--- torchrun 注入的环境变量 ---", flush=True)
        for key in (
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "GROUP_RANK",
        ):
            print(f"  {key} = {os.environ.get(key, '<未设置>')}", flush=True)

    # barrier：等所有 rank 都走到这里再退出，避免某个 rank 先销毁进程组。
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
