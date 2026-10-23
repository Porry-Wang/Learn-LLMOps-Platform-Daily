#!/usr/bin/env python3
"""
Day 2 · 调试入口：把命令行参数写成 Python 用例

为什么单独一个文件：
  - day02_capacity.py 负责命令行解析和报告打印；
  - 本文件负责“把参数写死”，方便在 PyCharm 里直接 Debug、
    在函数里打断点、单步看每个变量。

用法：
  1) 在 PyCharm 打开本文件；
  2) 在 run_case() 的第一行，或 day02_capacity.py 的函数里打断点；
  3) 右键本文件 → Debug 'day02_debug'（不需要配置命令行参数）；
  4) 想换案例，改下面 CASES 里的数字即可；
  5) 想只跑一个案例，把中间的 ONLY 改成对应 case 的 name；
  6) 终端也可以直接跑：python3 day02_debug.py

每个 case 都对应 day02_capacity.py 文件末尾的一条命令。
"""
import sys
from pathlib import Path

# 保证从任何工作目录运行都能 import 到同目录的 day02_capacity
sys.path.insert(0, str(Path(__file__).resolve().parent))

from types import SimpleNamespace  # noqa: E402

from day02_capacity import (  # noqa: E402
    per_gpu_model_state_bytes,
    per_gpu_activation_bytes,
    train_time_hours,
    pp_bubble_ratio,
    scan,
)

GB = 1e9


def run_case(
    *,
    name,
    params,
    layers,
    hidden,
    heads,
    tp,
    pp,
    dp,
    gpus,
    zero,
    recompute,
    tokens,
    seq_len=4096,
    micro_bs=1,
    mfu=0.4,
    peak_tflops=989.0,
    hbm_gb=80.0,
    micro_batches=0,
):
    """跑一个写死的案例，并把三本账的关键结果打印出来。

    断点建议打在下面第一行 model_state = ... 上，
    然后 Step Into 进入 per_gpu_model_state_bytes，逐行看变量。
    """
    model_state = per_gpu_model_state_bytes(params, zero, tp, pp, dp)
    activation = per_gpu_activation_bytes(
        seq_len, micro_bs, hidden, heads, layers, recompute, tp, pp
    )
    total = model_state + activation
    hours = train_time_hours(params, tokens, gpus, peak_tflops, mfu, recompute)

    m = micro_batches or 2 * pp
    bubble = pp_bubble_ratio(pp, m)

    print(f"\n=== {name} ===")
    print(f"  model_state = {model_state / GB:8.2f} GB")
    print(f"  activation  = {activation / GB:8.2f} GB")
    print(
        f"  total       = {total / GB:8.2f} GB  "
        f"({'OOM' if total / GB > hbm_gb else 'fits'}, HBM={hbm_gb:g} GB)"
    )
    print(f"  hours       = {hours:8.1f} h")
    print(f"  bubble      = {bubble:8.1%}")


# 每个 dict 就是一条 CLI 命令的 Python 版本。
# 想调试哪一个，就把哪一个留在列表里；其他的注释掉即可。
CASES = [
    # python3 day02_capacity.py --params 7 --layers 32 --hidden 4096 --heads 32 \
    #   --tp 1 --pp 1 --dp 8 --gpus 8 --zero 1 --recompute none --tokens 1
    # 预期：38.50 + 104.15 = 142.65 GB -> OOM
    dict(
        name="case1_7B_none (预期 OOM)",
        params=7, layers=32, hidden=4096, heads=32,
        tp=1, pp=1, dp=8, gpus=8, zero=1,
        recompute="none", tokens=1,
    ),
    # 同上，只把 recompute 改成 full
    # 预期：38.50 + 1.07 = 39.57 GB -> fits
    dict(
        name="case1_7B_full (预期 fits)",
        params=7, layers=32, hidden=4096, heads=32,
        tp=1, pp=1, dp=8, gpus=8, zero=1,
        recompute="full", tokens=1,
    ),
    # python3 day02_capacity.py --params 70 --layers 80 --hidden 8192 --heads 64 \
    #   --tp 8 --pp 1 --dp 16 --gpus 128 --zero 1 --recompute selective --tokens 1
    dict(
        name="case2_70B_TP8",
        params=70, layers=80, hidden=8192, heads=64,
        tp=8, pp=1, dp=16, gpus=128, zero=1,
        recompute="selective", tokens=1,
    ),
    # python3 day02_capacity.py --params 405 --layers 126 --hidden 16384 --heads 64 \
    #   --tp 8 --pp 16 --dp 128 --gpus 16384 --zero 1 --recompute full \
    #   --micro-batches 32 --tokens 1
    dict(
        name="case3_405B_TP8_PP16",
        params=405, layers=126, hidden=16384, heads=64,
        tp=8, pp=16, dp=128, gpus=16384, zero=1,
        recompute="full", tokens=1, micro_batches=32,
    ),
    # python3 day02_capacity.py --params 7 --layers 32 --hidden 4096 --heads 32 \
    #   --tp 8 --pp 1 --dp 125 --gpus 1000 --zero 1 --recompute none \
    #   --tokens 1 --mfu 0.4
    # 预期：训练时长约 29.5 小时
    dict(
        name="time_1000xH100",
        params=7, layers=32, hidden=4096, heads=32,
        tp=8, pp=1, dp=125, gpus=1000, zero=1,
        recompute="none", tokens=1, mfu=0.4,
    ),
    # 小序列，方便单步看激活数字，不会一上来就是 104GB
    # 预期：激活约 0.65 GB（同一公式，s 从 4096 改成 128）
    dict(
        name="tiny_activation_7B_s128",
        params=7, seq_len=128, layers=32, hidden=4096, heads=32,
        tp=1, pp=1, dp=8, gpus=8, zero=1,
        recompute="none", tokens=1,
    ),
]


# 想只跑一个 case，把它的 name 填到这里；None = 跑全部。
# 例：ONLY = "case2_70B_TP8"
ONLY = None


# ---- scan 的调试入口 ----
# RUN_SCAN=True 时依次扫描 SCAN_CASES；想只扫一个，把其他的注释掉。
# 断点建议打在 day02_capacity.scan() 的 for tp 那一行。
RUN_SCAN = True

# 所有 scan 用例共用的默认参数
SCAN_DEFAULTS = dict(
    seq_len=4096, micro_bs=1, peak_tflops=989.0,
    hbm_gb=80.0, tokens=1.0, mfu=0.4,
)

# 对应文件末尾的三个验证案例（案例3 沿用 --micro-batches 32）
SCAN_CASES = [
    dict(
        name="scan_case1_7B_8卡",
        params=7, layers=32, hidden=4096, heads=32, gpus=8,
        micro_batches=0,
    ),
    dict(
        name="scan_case2_70B_128卡",
        params=70, layers=80, hidden=8192, heads=64, gpus=128,
        micro_batches=0,
    ),
    dict(
        name="scan_case3_405B_16384卡",
        params=405, layers=126, hidden=16384, heads=64, gpus=16384,
        micro_batches=32,
    ),
    # 自检题5：7B / 1000 卡 / 1T token
    dict(
        name="scan_自检5_1000卡",
        params=7, layers=32, hidden=4096, heads=32, gpus=1000,
        micro_batches=0,
    ),
]


def main():
    if RUN_SCAN:
        for case in SCAN_CASES:
            case = dict(case)          # 复制一份，避免 pop 改到原列表
            name = case.pop("name")
            print(f"\n########## {name} ##########")
            scan(SimpleNamespace(**SCAN_DEFAULTS, **case))
        return
    for case in CASES:
        if ONLY is None or case["name"] == ONLY:
            run_case(**case)


if __name__ == "__main__":
    main()
