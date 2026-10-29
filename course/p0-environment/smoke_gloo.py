"""P0 环境预检：验证本机 torch.distributed 可用。

这是《01-一个月学习计划.md》§二 步骤 4 的脚本。
macOS 上没有 NCCL，只能用 gloo 后端在 CPU 上模拟多 rank。

运行方式（用仓库根的 venv，不需要 activate）：

    cd <仓库根>
    .venv/bin/python course/s02-distributed/torchrun_local.py \
        --nproc_per_node 4 course/p0-environment/smoke_gloo.py

为什么不用 docstring 原先写的 `torchrun --standalone`：本机 macOS 上
`socket.getfqdn()` 返回一个畸形的反向名（`1.0.0.0...ip6.arpa`），原生 torchrun
会在 rendezvous 阶段无限重试。10-29 实测：`torchrun --standalone` 4 分钟无输出被手动
终止；走 `torchrun_local.py`（显式 MASTER_ADDR=127.0.0.1）4 个 rank 全部 OK。
S2 的 torchrun_local.py 就是这么来的，这里复用同一个绕行。

期望输出：4 行，每行 after 都是 [10.0, 10.0, 10.0, 10.0]
（因为 1+2+3+4 = 10，all_reduce SUM 后每个 rank 都拿到总和）

跑不通就不要开始 P1，先按计划里那张「常见报错」表排查。
"""

import os

import torch
import torch.distributed as dist


def main() -> None:
    # ★ gloo，不是 nccl —— macOS/无 GPU 环境的唯一选择
    dist.init_process_group(backend="gloo")

    rank = dist.get_rank()
    world = dist.get_world_size()

    t = torch.ones(4) * (rank + 1)
    before = t.clone()

    dist.all_reduce(t, op=dist.ReduceOp.SUM)

    expected = float(sum(range(1, world + 1)))
    ok = all(abs(v - expected) < 1e-6 for v in t.tolist())

    print(
        f"[rank {rank}/{world}] before={before.tolist()} "
        f"after={t.tolist()} expected={expected} {'OK' if ok else 'FAIL'}",
        flush=True,
    )

    # 顺带打印 torchrun 注入的环境变量 —— S2 会用到这份清单
    if rank == 0:
        keys = [
            "RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
            "GROUP_RANK", "LOCAL_WORLD_SIZE", "TORCHELASTIC_RUN_ID",
            "TORCHELASTIC_MAX_RESTARTS",
        ]
        print("\n--- torchrun 注入的环境变量 ---", flush=True)
        for k in keys:
            print(f"  {k} = {os.environ.get(k, '<未设置>')}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
