"""容量账本 → 可行放置候选。

这一层只回答一个问题：

    给定「机器数 × 单机卡数」和模型/序列参数，
    哪些 TP/PP/DP + ZeRO + recompute 组合放得下？按有效时长谁最快？

它不做的事（留给后面的 session）：

    · 准入/排队/Gang              → S6 scheduler.py
    · rank 分配、拒绝原因、top-3  → S7 placement.py
    · 真实 K8s 调度、GPU/NCCL     → 全项目不做

最小用法：

    >>> top = feasible_placements(16, 8, 70, 4096, 80, 8192, 64)[0]
    >>> top.parallel
    'TP8 PP1 DP16 ZeRO-1'
    >>> round(top.total_gb, 2)
    52.97

三个玩具边界，讲报告时要主动说：

    1. 通信缓冲区记 0：真实系统还要留 activation/通信 bucket 的余量。
    2. TP/ZeRO 的通信成本没有定价：只用 tp <= 单机卡数 做硬约束。
    3. 激活按「在飞 1 个 micro-batch」计：1F1B 下实际在飞 min(m, pp) 个。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

__all__ = [
    "Placement",
    "per_gpu_model_state_bytes",
    "per_gpu_activation_bytes",
    "train_time_hours",
    "pp_bubble_ratio",
    "feasible_placements",
]

GB = 1e9  # 十进制 GB（NVIDIA / 论文口径）

DEFAULT_PEAK_TFLOPS = 989.0  # H100 BF16 稠密峰值
DEFAULT_HBM_GB = 80.0
DEFAULT_MFU = 0.4


@dataclass(frozen=True)
class Placement:
    """一个放得下的并行方案，以及它的账本明细。"""

    tp: int
    pp: int
    dp: int
    zero: int
    recompute: str
    micro_batches: int

    bubble: float          # 流水线气泡占比
    model_gb: float        # 每卡模型状态
    activation_gb: float   # 每卡激活
    total_gb: float        # 每卡合计（通信缓冲记 0）
    hours: float           # 纯算力下界
    effective_hours: float # 扣掉气泡后的等效时长，排序键

    machines: int
    cards_per_machine: int
    world_size: int

    @property
    def parallel(self) -> str:
        return f"TP{self.tp} PP{self.pp} DP{self.dp} ZeRO-{self.zero}"

    def as_dict(self) -> dict:
        """给 CLI / JSON 序列化用。"""
        return {
            "tp": self.tp,
            "pp": self.pp,
            "dp": self.dp,
            "zero": self.zero,
            "recompute": self.recompute,
            "micro_batches": self.micro_batches,
            "bubble": self.bubble,
            "model_gb": self.model_gb,
            "activation_gb": self.activation_gb,
            "total_gb": self.total_gb,
            "hours": self.hours,
            "effective_hours": self.effective_hours,
            "machines": self.machines,
            "cards_per_machine": self.cards_per_machine,
            "world_size": self.world_size,
        }

    def summary(self) -> str:
        """一行摘要，方便 demo / CLI 打印。"""
        return (f"{self.parallel} {self.recompute:<9s} | "
                f"{self.total_gb:6.2f} GB | bubble {self.bubble:5.1%} | "
                f"{self.effective_hours:7.1f} h")


# ---------------------------------------------------------------------------
# 三本账（与 labs/w1/day02_capacity.py 同一套公式）
# ---------------------------------------------------------------------------

def per_gpu_model_state_bytes(params_b: float, zero: int, tp: int, pp: int, dp: int) -> float:
    """每卡模型状态字节数。

    16Ψ = 权重 2 + 梯度 2 + FP32 主权重 4 + Adam m 4 + Adam v 4。
    · ZeRO-0：16Ψ 不切，只按 TP×PP 切模型。
    · ZeRO-1：4Ψ 每卡保留；12Ψ（主权重+m+v）按 dp 切。
    """
    psi = params_b * 1e9

    if zero == 0:
        per_gpu = 16 * psi
    elif zero == 1:
        per_gpu = 4 * psi + 12 * psi / dp
    elif zero == 2:
        per_gpu = 2 * psi + 14 * psi / dp
    elif zero == 3:
        per_gpu = 16 * psi / dp
    else:
        raise ValueError(f"未知 ZeRO stage: {zero}")

    return per_gpu / (tp * pp)


def per_gpu_activation_bytes(
    seq_len: int,
    micro_bs: int,
    hidden: int,
    heads: int,
    layers: int,
    recompute: str,
    tp: int,
    pp: int,
) -> float:
    """每卡激活字节数。

    单层：s·b·h·(34 + 5·a·s/h)。
    · 34 项 ∝ s：线性层中间结果，保留。
    · 5·a·s/h 项 ∝ s²：attention scores/softmax，序列越长越贵。
    · selective 砍掉 s² 项；full 只留每层输入 2·s·b·h。

    只按 TP 切、按 PP 分层；dp 不切激活。
    """
    sbh = seq_len * micro_bs * hidden

    if recompute == "none":
        per_layer = sbh * (34 + 5 * heads * seq_len / hidden)
    elif recompute == "selective":
        per_layer = sbh * 34
    elif recompute == "full":
        per_layer = 2 * sbh
    else:
        raise ValueError(f"未知 recompute 模式: {recompute}")

    return per_layer * layers / pp / tp


def train_time_hours(
    params_b: float,
    tokens_t: float,
    gpus: int,
    peak_tflops: float,
    mfu: float,
    recompute: str,
) -> float:
    """纯算力下界（小时）：C / (卡数 × 峰值 × MFU)。

    C = 6ND；full recompute 多算一遍前向 → 8ND。
    单位陷阱：params_b 是 B、tokens_t 是 T，都要先展开成「个」。
    """
    if recompute not in ("none", "selective", "full"):
        raise ValueError(f"未知 recompute 模式: {recompute}")

    n = params_b * 1e9
    d = tokens_t * 1e12
    flops = (8 if recompute == "full" else 6) * n * d
    effective = gpus * peak_tflops * 1e12 * mfu
    return flops / effective / 3600


def pp_bubble_ratio(pp: int, micro_batches: int) -> float:
    """流水线气泡占比 (p-1)/(m+p-1)。"""
    return (pp - 1) / (micro_batches + pp - 1)


# ---------------------------------------------------------------------------
# 可行放置枚举
# ---------------------------------------------------------------------------

def feasible_placements(
    machines: int,
    cards_per_machine: int,
    params_b: float,
    seq_len: int,
    layers: int,
    hidden: int,
    heads: int,
    *,
    tokens_t: float = 1.0,
    peak_tflops: float = DEFAULT_PEAK_TFLOPS,
    hbm_gb: float = DEFAULT_HBM_GB,
    mfu: float = DEFAULT_MFU,
    micro_batches: int = 0,
    max_tp: int = 8,
    zero_stages: Sequence[int] = (0, 1),
    recompute_modes: Sequence[str] = ("none", "selective", "full"),
) -> List[Placement]:
    """枚举所有放得下的并行方案，按有效时长升序返回。

    约束（每一条都对应一个真实原因）：

    · tp <= min(max_tp, cards_per_machine)
      TP 组不出单机 NVLink 域。8 卡一台 → TP 最多 8。
    · tp 整除 heads
      attention head 不能切一半。
    · tp·pp·dp == world_size
      每张卡必须正好属于一个 rank。
    · pp <= layers
      每个 pipeline stage 至少分到一层。
    · m >= 2·pp
      气泡 (p-1)/(m+p-1) 的上限约束；m=0 时取 2·pp。
    · 每卡模型状态 + 激活 <= hbm_gb
      OOM 是准入门槛，不是排序键。

    micro_batches=0 表示每个方案取 m=2·pp；给固定值则只保留满足 m>=2·pp 的方案。

    zero_stages 默认只枚举 ZeRO-0/1。公式支持 0/1/2/3，但 ZeRO-2/3 的通信成本
    没有定价，混进排序会得到过于乐观的结论；面试里要主动说明这个边界。
    """
    _validate_inputs(
        machines, cards_per_machine, params_b, seq_len, layers, hidden, heads,
        tokens_t, peak_tflops, hbm_gb, mfu, micro_batches, max_tp,
        zero_stages, recompute_modes,
    )

    world_size = machines * cards_per_machine
    tp_limit = min(max_tp, cards_per_machine)

    # 同一 recompute 模式下，纯算力时长只由总卡数决定，与 TP/PP/DP 怎么切无关。
    hours_by_recompute = {
        mode: train_time_hours(params_b, tokens_t, world_size, peak_tflops, mfu, mode)
        for mode in recompute_modes
    }

    candidates: List[Placement] = []

    for tp in range(1, tp_limit + 1):
        if world_size % tp != 0:
            continue
        if heads % tp != 0:
            continue

        rest = world_size // tp
        for pp in range(1, min(rest, layers) + 1):
            if rest % pp != 0:
                continue

            dp = rest // pp
            m = micro_batches or 2 * pp
            if m < 2 * pp:
                continue

            bubble = pp_bubble_ratio(pp, m)

            for zero in zero_stages:
                # dp=1 时 ZeRO-1 与 ZeRO-0 等价，去重。
                if zero == 1 and dp == 1:
                    continue

                model = per_gpu_model_state_bytes(params_b, zero, tp, pp, dp)
                for recompute in recompute_modes:
                    activation = per_gpu_activation_bytes(
                        seq_len, 1, hidden, heads, layers, recompute, tp, pp,
                    )
                    total = model + activation
                    if total / GB > hbm_gb:
                        continue  # 显存关：先筛掉，不参与排序

                    hours = hours_by_recompute[recompute]
                    candidates.append(Placement(
                        tp=tp,
                        pp=pp,
                        dp=dp,
                        zero=zero,
                        recompute=recompute,
                        micro_batches=m,
                        bubble=bubble,
                        model_gb=model / GB,
                        activation_gb=activation / GB,
                        total_gb=total / GB,
                        hours=hours,
                        effective_hours=hours / (1 - bubble),
                        machines=machines,
                        cards_per_machine=cards_per_machine,
                        world_size=world_size,
                    ))

    # 先看扣气泡后的时长；同分优先小 TP、小 PP，再优先省显存。
    # 不能把「是否 OOM」当排序键——能走到这里的都已经不 OOM 了。
    candidates.sort(key=lambda p: (p.effective_hours, p.tp, p.pp, p.total_gb))
    return candidates


def _validate_inputs(
    machines: int,
    cards_per_machine: int,
    params_b: float,
    seq_len: int,
    layers: int,
    hidden: int,
    heads: int,
    tokens_t: float,
    peak_tflops: float,
    hbm_gb: float,
    mfu: float,
    micro_batches: int,
    max_tp: int,
    zero_stages: Sequence[int],
    recompute_modes: Sequence[str],
) -> None:
    """在枚举前拒绝畸形输入，避免返回一个看似合理的空结果。"""
    if machines < 1:
        raise ValueError("machines 必须 >= 1")
    if cards_per_machine < 1:
        raise ValueError("cards_per_machine 必须 >= 1")
    if params_b <= 0:
        raise ValueError("params_b 必须 > 0")
    if seq_len < 1 or layers < 1 or hidden < 1 or heads < 1:
        raise ValueError("seq_len / layers / hidden / heads 必须 >= 1")
    if heads > hidden:
        raise ValueError("heads 不能大于 hidden")
    if tokens_t <= 0:
        raise ValueError("tokens_t 必须 > 0")
    if peak_tflops <= 0:
        raise ValueError("peak_tflops 必须 > 0")
    if hbm_gb <= 0:
        raise ValueError("hbm_gb 必须 > 0")
    if not 0 < mfu <= 1:
        raise ValueError("mfu 必须在 (0, 1] 内")
    if micro_batches < 0:
        raise ValueError("micro_batches 必须 >= 0")
    if max_tp < 1:
        raise ValueError("max_tp 必须 >= 1")

    if not zero_stages:
        raise ValueError("zero_stages 不能为空")
    if not recompute_modes:
        raise ValueError("recompute_modes 不能为空")

    valid_zero = (0, 1, 2, 3)
    for stage in zero_stages:
        if stage not in valid_zero:
            raise ValueError(f"未知 ZeRO stage: {stage}")

    valid_recompute = ("none", "selective", "full")
    for mode in recompute_modes:
        if mode not in valid_recompute:
            raise ValueError(f"未知 recompute 模式: {mode}")
