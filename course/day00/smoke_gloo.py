"""Day 0 环境预检：验证本机 torch.distributed 可用。

这是《01-一个月学习计划.md》§二 步骤 4 的脚本。
macOS 上没有 NCCL，只能用 gloo 后端在 CPU 上模拟多 rank。

运行方式（务必先 source .venv/bin/activate）：

    cd course/day00
    MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
    torchrun --standalone --nproc_per_node=4 smoke_gloo.py

期望输出：4 行，每行的 result 都是 [10.0, 10.0, 10.0, 10.0]
（因为 1+2+3+4 = 10，all_reduce SUM 后每个 rank 都拿到总和）

跑不通就不要开始 Day 1，先按计划里那张「常见报错」表排查。
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

    # 顺带打印 torchrun 注入的环境变量 —— Day 4 会用到这份清单
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
