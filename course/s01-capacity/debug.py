#!/usr/bin/env python3
"""
Day 2 · 调试入口：把命令行参数写成 Python 用例

为什么单独一个文件：
  - capacity.py 负责命令行解析和报告打印；
  - 本文件负责把参数写死，方便在 PyCharm 里直接 Debug、
    在函数里打断点、单步看每个变量。

用法：
  1) 在 PyCharm 打开本文件；
  2) 在 run_case() 的 model_state = ... 那一行打断点；
  3) 右键本文件 → Debug 'debug'（不需要配置命令行参数）；
  4) 想换案例，改 CASES / SCAN_CASES 里的数字即可；
  5) 想只跑一个单配置案例，把 ONLY 改成对应 case 的 name；
  6) 终端也可以直接跑：.venv/bin/python course/s01-capacity/debug.py

两个开关：
  RUN_CASES=True  跑单配置案例（三本账明细）
  RUN_SCAN=True   跑 scan 案例（枚举 + 排序）
  两个都开也可以；默认只跑 scan。
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

# 保证从任何工作目录运行都能 import 到同目录的 capacity。
sys.path.insert(0, str(Path(__file__).resolve().parent))

from capacity import (  # noqa: E402
    GB,
    per_gpu_activation_bytes,
    per_gpu_model_state_bytes,
    pp_bubble_ratio,
    scan,
    train_time_hours,
)

Case = Dict[str, Any]


def run_case(
    *,
    name: str,
    params: float,
    layers: int,
    hidden: int,
    heads: int,
    tp: int,
    pp: int,
    dp: int,
    gpus: int,
    zero: int,
    recompute: str,
    tokens: float,
    seq_len: int = 4096,
    micro_bs: int = 1,
    mfu: float = 0.4,
    peak_tflops: float = 989.0,
    hbm_gb: float = 80.0,
    micro_batches: int = 0,
    expect_oom: Optional[bool] = None,
) -> None:
    """跑一个写死的单配置案例，并打印三本账的关键结果。

    断点建议打在 model_state = ... 这一行，然后 Step Into
    per_gpu_model_state_bytes()，逐行看 psi / dp / tp / pp 怎么变。
    """
    model_state = per_gpu_model_state_bytes(params, zero, tp, pp, dp)
    activation = per_gpu_activation_bytes(
        seq_len, micro_bs, hidden, heads, layers, recompute, tp, pp,
    )
    total = model_state + activation
    hours = train_time_hours(params, tokens, gpus, peak_tflops, mfu, recompute)

    m = micro_batches or 2 * pp
    bubble = pp_bubble_ratio(pp, m)
    is_oom = total / GB > hbm_gb

    print(f"\n=== {name} ===")
    print(f"  model_state = {model_state / GB:8.2f} GB")
    print(f"  activation  = {activation / GB:8.2f} GB")
    print(f"  total       = {total / GB:8.2f} GB  "
          f"({'OOM' if is_oom else 'fits'}, HBM={hbm_gb:g} GB)")
    print(f"  hours       = {hours:8.1f} h（纯算力下界）")
    print(f"  bubble      = {bubble:8.1%}（m={m}）")

    if expect_oom is not None:
        verdict = "✅ 符合预期" if is_oom == expect_oom else "❌ 与预期不符"
        print(f"  自检        = {verdict}")


# 每个 dict 就是一条 CLI 命令的 Python 版本。
# 想调试哪一个，就把哪一个留在列表里；其他的注释掉即可。
CASES: List[Case] = [
    # python3 capacity.py --params 7 --layers 32 --hidden 4096 --heads 32 \
    #   --tp 1 --pp 1 --dp 8 --gpus 8 --zero 1 --recompute none --tokens 1
    # 预期：38.50 + 104.15 = 142.65 GB -> OOM
    dict(
        name="case1_7B_none (预期 OOM)",
        params=7, layers=32, hidden=4096, heads=32,
        tp=1, pp=1, dp=8, gpus=8, zero=1,
        recompute="none", tokens=1, expect_oom=True,
    ),
    # 同上，只把 recompute 改成 full
    # 预期：38.50 + 1.07 = 39.57 GB -> fits
    dict(
        name="case1_7B_full (预期 fits)",
        params=7, layers=32, hidden=4096, heads=32,
        tp=1, pp=1, dp=8, gpus=8, zero=1,
        recompute="full", tokens=1, expect_oom=False,
    ),
    # python3 capacity.py --params 70 --layers 80 --hidden 8192 --heads 64 \
    #   --tp 8 --pp 1 --dp 16 --gpus 128 --zero 1 --recompute selective --tokens 1
    dict(
        name="case2_70B_TP8",
        params=70, layers=80, hidden=8192, heads=64,
        tp=8, pp=1, dp=16, gpus=128, zero=1,
        recompute="selective", tokens=1, expect_oom=False,
    ),
    # python3 capacity.py --params 405 --layers 126 --hidden 16384 --heads 64 \
    #   --tp 8 --pp 16 --dp 128 --gpus 16384 --zero 1 --recompute full \
    #   --micro-batches 32 --tokens 1
    dict(
        name="case3_405B_TP8_PP16",
        params=405, layers=126, hidden=16384, heads=64,
        tp=8, pp=16, dp=128, gpus=16384, zero=1,
        recompute="full", tokens=1, micro_batches=32, expect_oom=False,
    ),
    # python3 capacity.py --params 7 --layers 32 --hidden 4096 --heads 32 \
    #   --tp 8 --pp 1 --dp 125 --gpus 1000 --zero 1 --recompute none \
    #   --tokens 1 --mfu 0.4
    # 预期：训练时长约 29.5 小时
    dict(
        name="time_1000xH100",
        params=7, layers=32, hidden=4096, heads=32,
        tp=8, pp=1, dp=125, gpus=1000, zero=1,
        recompute="none", tokens=1, mfu=0.4, expect_oom=False,
    ),
    # 小序列，方便单步看激活数字，不会一上来就是 104GB。
    # 预期：激活约 0.65 GB（同一公式，s 从 4096 改成 128）
    dict(
        name="tiny_activation_7B_s128",
        params=7, seq_len=128, layers=32, hidden=4096, heads=32,
        tp=1, pp=1, dp=8, gpus=8, zero=1,
        recompute="none", tokens=1, expect_oom=False,
    ),
]


# 想只跑一个单配置 case，把它的 name 填到这里；None = 跑全部。
# 例：ONLY = "case2_70B_TP8"
ONLY: Optional[str] = None


# ---- 单配置案例开关 ----
# 默认只跑 scan；想看三本账明细时改成 True。
RUN_CASES = False

# ---- scan 的调试入口 ----
RUN_SCAN = True

# 所有 scan 用例共用的默认参数。
SCAN_DEFAULTS: Case = dict(
    seq_len=4096, micro_bs=1, peak_tflops=989.0,
    hbm_gb=80.0, tokens=1.0, mfu=0.4,
)

# 三个正式案例 + 自检题 5。
SCAN_CASES: List[Case] = [
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
    dict(
        name="scan_自检5_7B_1000卡",
        params=7, layers=32, hidden=4096, heads=32, gpus=1000,
        micro_batches=0,
    ),
]


def main() -> None:
    # 开关一：单配置案例。
    if RUN_CASES:
        for case in CASES:
            if ONLY is None or case["name"] == ONLY:
                run_case(**case)

    # 开关二：scan 案例。
    if RUN_SCAN:
        for case in SCAN_CASES:
            case = dict(case)  # 复制一份，避免 pop 改到原列表
            name = case.pop("name")
            print(f"\n########## {name} ##########")
            scan(SimpleNamespace(**SCAN_DEFAULTS, **case))

    if not RUN_CASES and not RUN_SCAN:
        print("RUN_CASES 和 RUN_SCAN 都是 False：把要跑的那个改成 True。")


if __name__ == "__main__":
    main()
