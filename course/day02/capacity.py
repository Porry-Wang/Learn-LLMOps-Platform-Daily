#!/usr/bin/env python3
"""
Day 2 · 显存与算力账本计算器（完整阅读版）

这份脚本把「一个训练配置要多少显存、要跑多久」拆成三本账：

  ① 模型状态账：16Ψ 字节/参数；TP×PP 切模型，dp 只切 ZeRO 允许切的部分
  ② 激活账：每层 s·b·h·(34 + 5·a·s/h) 字节；只按 TP 切、按 PP 分层
  ③ 算力账：C = 6·N·D FLOPs（full recompute 用 8ND）

阅读顺序：
  1) 先读三个账本函数上方的注释；
  2) 再读 build_report()，看三本账怎么装配成一张单配置报告；
  3) 最后读 scan()，看平台如何枚举候选、先筛 OOM、再按有效时长排序。

配套文件：
  · debug.py              把 CLI 参数写成 Python 用例，方便断点单步
  · course/day02/README.md         三本账的完整推导与案例
  · course/day02/question.md 闭卷题库 + 标准答案

五个命门（闭卷只默这五条，不默代码）：
  ① 16 = 2+2+4+4+4 各对应什么
  ② 四个 ZeRO stage 的每卡模型状态公式
  ③ 激活为什么只切 TP 不切 dp；s² 项从哪来
  ④ 6ND 的单位陷阱（B×T 差 1e18）；full recompute 为什么把算力账改成 8ND
  ⑤ --scan 的排序键为什么不能是「是否 OOM」
"""
import argparse
import sys
from typing import List, Optional, Sequence

GB = 1e9  # 十进制 GB（NVIDIA / 论文口径）

# ---------- 结构参数速查（Llama 系，用于三个验证案例） ----------
#   7B : L=32,  H=4096,  A=32
#   70B: L=80,  H=8192,  A=64
#   405B: L=126, H=16384, A=64


# ============================================================================
# 账本① 阅读笔记（不影响运行，对着代码看）
# ============================================================================
# 问题：训练时每张卡为了记住“模型 + 优化器”，要占多少显存？
#
# 1) 一个参数占 16 字节，也就是 16Ψ
#
#      存什么              字节    干什么用
#      ----------------    ----    ------------------------------
#      权重（BF16）          2      前向计算
#      梯度（BF16）          2      反向传播：告诉权重往哪调
#      FP32 主权重           4      更新时用高精度，避免 BF16 抹掉小变化
#      Adam 动量 m           4      优化器历史
#      Adam 方差 v           4      优化器历史
#      ----------------    ----
#      合计                 16
#
#    7B 模型：16 × 7e9 = 112e9 字节 = 112 GB。
#
#    顺带回答题目卡 Q1 后半问：
#      FP16 指数范围窄（最大约 65504），大梯度/大激活容易溢出；
#      BF16 指数范围和 FP32 一样宽（约 3.4e38），不容易溢出，但尾数精度低。
#      所以训练常用“BF16 权重/梯度 + FP32 主权重”。
#
# 2) ZeRO-0 与 ZeRO-1 的区别
#
#      ZeRO-0：16 字节全留在每张卡
#          per_gpu = 16 * psi
#
#      ZeRO-1：把 16 字节拆成两堆
#          4 字节（权重+梯度）：每张卡都要用 → 不切 → 4 * psi
#          12 字节（FP32 主权重 + m + v）：只在更新时用 → 按 dp 切
#          per_gpu = 4 * psi + 12 * psi / dp
#
# 3) TP/PP 再切
#
#      TP 按宽度切每层矩阵，PP 按层分段；切走的部分不在这张卡上。
#      所以最后 return per_gpu / (tp * pp)。
#
# 4) 代入案例 1（7B, ZeRO-1, tp=1, pp=1, dp=8）
#
#      psi     = 7e9
#      per_gpu = 4*7e9 + 12*7e9/8
#              = 28e9 + 10.5e9
#              = 38.5e9 字节 = 38.5 GB
#      return  = 38.5 / (1*1) = 38.5 GB
#      → 与脚本输出 38.50 GB 对上。
#
#    再加 TP/PP（tp=2, pp=4, dp=2）：
#      per_gpu = 4*7 + 12*7/2 = 28 + 42 = 70 GB
#      每卡    = 70 / (2*4) = 8.75 GB
#
# 一句话：
#   4Ψ 每个 DP 副本都要用，所以不除 dp；
#   12Ψ 只在更新时用，所以 ZeRO-1 按 dp 切；
#   TP/PP 是物理切模型，最后整体除以 tp*pp。
# ============================================================================


def per_gpu_model_state_bytes(params_b: float, zero_stage: int,
                              tp: int, pp: int, dp: int) -> float:
    """账本①：每卡模型状态（字节）。

    每个参数 16 字节 = 参数 2 + 梯度 2 + fp32 主参数 4 + 动量 4 + 方差 4。

    切分规则（命门②，逐字读懂再动手）：
      先按 TP×PP 把模型切成 tp*pp 份，每卡只拿自己那片；
      dp 这一维只有 ZeRO 允许切的部分才能再除以 dp：
        stage 0：什么都不给 dp 切   → 每卡 16Ψ / (tp·pp)
        stage 1：优化器 12Ψ 给 dp 切 → 每卡 (4Ψ + 12Ψ/dp) / (tp·pp)
        stage 2：梯度 2Ψ 也给 dp 切  → 每卡 (2Ψ + 14Ψ/dp) / (tp·pp)
        stage 3：全部给 dp 切        → 每卡 16Ψ / (tp·pp·dp)

    params_b 的单位是 B（十亿个参数）。返回每卡字节数（float）。
    """
    psi = params_b * 1e9  # 7 → 70 亿个参数

    if zero_stage == 0:
        per_gpu = 16 * psi
    elif zero_stage == 1:
        per_gpu = 4 * psi + 12 * psi / dp
    elif zero_stage == 2:
        per_gpu = 2 * psi + 14 * psi / dp
    elif zero_stage == 3:
        per_gpu = 16 * psi / dp
    else:
        raise ValueError(f"未知 ZeRO stage: {zero_stage}（只支持 0/1/2/3）")

    return per_gpu / (tp * pp)


def per_gpu_activation_bytes(seq_len: int, micro_bs: int, hidden: int,
                             heads: int, layers: int, recompute: str,
                             tp: int, pp: int) -> float:
    """账本②：每卡激活（字节）。

    单层、切分前： s·b·h·(34 + 5·a·s/h) 字节
      · 34 项：∝ s 的一次方 —— 每层必须留存的输出/中间结果（线性部分）
      · 5·a·s/h 项：∝ s 的平方 —— attention scores/softmax，序列越长越贵
    切分：只按 TP 切（除以 tp），按 PP 分层（卡上只有 layers/pp 层）；
          ★ dp 不切激活 ★——每个 rank 跑自己的 batch，激活本来就只有 1/b 个 b（命门③）
    recompute 打折：
      none      → 原式
      selective → 只重算 s² 那部分，单层变 s·b·h·34
      full      → 只保留每层输入，单层变 2·s·b·h（同时算力账要改成 8ND，见账本③）

    简化说明：这里按「在飞 1 个 micro-batch」计；1F1B 调度下实际在飞 min(m, pp) 个，
    报告里注一句即可，不改代码。
    """
    sbh = seq_len * micro_bs * hidden  # 一份中间结果有多少个数

    if recompute == "none":
        per_layer = sbh * (34 + 5 * heads * seq_len / hidden)
    elif recompute == "selective":
        per_layer = sbh * 34
    else:  # full
        per_layer = 2 * sbh

    return per_layer * layers / pp / tp


def train_time_hours(params_b: float, tokens_t: float, gpus: int,
                     peak_tflops: float, mfu: float, recompute: str) -> float:
    """账本③：纯算力下界的训练时长（小时）。

    C = 6·N·D FLOPs（recompute=full → 8·N·D：前向重算多烧一遍）。
      N = 参数「个数」，D = token「个数」——params_b 单位 B、tokens_t 单位 T，
      先把两个单位都展开成「个」，这就是命门④的单位陷阱。
    有效算力 = gpus × peak_tflops×1e12 × mfu   （FLOPs/秒）
    """
    n = params_b * 1e9       # 参数「个数」：7B → 7e9
    d = tokens_t * 1e12      # token「个数」：1T → 1e12

    # 普通训练 ≈ 前向 2ND + 反向 4ND = 6ND；full recompute 多一次前向 → 8ND
    flops = (8 if recompute == "full" else 6) * n * d

    # 有效算力：卡数 × 单卡峰值（TFLOPS → FLOPs/s）× MFU
    effective = gpus * peak_tflops * 1e12 * mfu

    # 秒 → 小时
    return flops / effective / 3600


def pp_bubble_ratio(pp: int, micro_batches: int) -> float:
    """MFU 上界的第一个约束：流水线气泡占比 = (p-1)/(m+p-1)。

    m（micro-batch 数）太小，气泡吃掉一切——这就是为什么 scan 里
    要强制 m >= 2*pp（计划文档「坑」一节）。
    """
    return (pp - 1) / (micro_batches + pp - 1)


def tp_ok(tp: int) -> bool:
    """MFU 上界的第二个约束：TP 不出 NVLink 域（单机 8 卡）→ tp <= 8。
    跨机做 TP 不是不行，是慢到不值得——自检题 4 会要求你解释。
    """
    return tp <= 8


# ---------- 装配与打印：把三本账拼成一张报告 ----------

def build_report(args: argparse.Namespace) -> bool:
    """打印单配置的三本账明细，返回是否放得下。"""
    model_state = per_gpu_model_state_bytes(
        args.params, args.zero, args.tp, args.pp, args.dp,
    )
    activation = per_gpu_activation_bytes(
        args.seq_len, args.micro_bs, args.hidden, args.heads, args.layers,
        args.recompute, args.tp, args.pp,
    )
    # 教学版把通信缓冲区记 0；真实系统里它通常是数 GB 到十几 GB，不能忽略。
    comm_buffer = 0.0
    total = model_state + activation + comm_buffer

    hours = train_time_hours(
        args.params, args.tokens, args.gpus, args.peak_tflops, args.mfu,
        args.recompute,
    )
    m = args.micro_batches or 2 * args.pp
    bubble = pp_bubble_ratio(args.pp, m)
    effective_hours = hours / (1 - bubble)

    rows = [
        ("权重+梯度+优化器(每卡)", model_state),
        ("激活(每卡)", activation),
        ("通信缓冲区(每卡)", comm_buffer),
    ]
    print(f"\n== 每卡显存明细（{args.params}B, TP{args.tp} PP{args.pp} DP{args.dp}, "
          f"ZeRO-{args.zero}, recompute={args.recompute}）==")
    print(f"  拓扑：TP{args.tp} × PP{args.pp} × DP{args.dp} = "
          f"{args.tp * args.pp * args.dp} 卡；m={m}")
    for name, value in rows:
        print(f"  {name:<24s} {value / GB:8.2f} GB")
    print(f"  {'合计/卡':<24s} {total / GB:8.2f} GB   (HBM {args.hbm_gb} GB)")
    print(f"  结论：{'❌ OOM' if total / GB > args.hbm_gb else '✅ 放得下'}")
    print(f"  纯算力时长：{hours:.1f} 小时（MFU={args.mfu}）")
    print(f"  气泡占比：  {bubble:.1%} → 扣气泡后的有效时长：{effective_hours:.1f} 小时")
    print("  注：未计 TP/ZeRO 通信成本，真实值更长。")
    return total / GB <= args.hbm_gb


def scan(args: argparse.Namespace) -> None:
    """枚举 (tp, pp, dp, zero, recompute)：tp*pp*dp == gpus，tp<=8，
    tp 整除 heads，m>=2*pp。输出所有不 OOM 的配置，按预估吞吐排序（命门⑤）：
        吞吐 ∝ 1 / (时长 / (1 - 气泡占比))

    注意：scan 默认只枚举 ZeRO-0/1。ZeRO-2/3 的公式已在账本①里实现，
    但通信成本没有定价，放进排序会给出过于乐观的结论，所以先不混进来。
    """
    m_fixed = args.micro_batches  # 0 表示每个配置按 m = 2*pp 取
    m_desc = "m=2·pp" if not m_fixed else f"m={m_fixed} >= 2·pp"
    tried = 0
    candidates: List[dict] = []

    # ① 枚举拓扑：tp 不出单机 NVLink 域，pp/dp 必须整除剩余卡数
    for tp in range(1, 9):
        if tp > args.gpus or not tp_ok(tp) or args.gpus % tp != 0:
            continue
        if args.heads % tp != 0:
            continue  # TP 按 attention head 切，每卡必须分到整数个 head
        rest = args.gpus // tp
        for pp in range(1, rest + 1):
            if rest % pp != 0:
                continue
            if pp > args.layers:
                continue  # 物理下限：每个 stage 至少分到 1 层
            dp = rest // pp
            m = m_fixed or 2 * pp
            if m < 2 * pp:
                continue  # 气泡约束：micro-batch 太少，PP 吃不饱
            bubble = pp_bubble_ratio(pp, m)

            # ② 固定拓扑后，再枚举 ZeRO 与重算策略，逐本账累加
            zero_stages = (0,) if dp == 1 else (0, 1)  # dp=1 时 ZeRO-1 与 ZeRO-0 等价，去重
            for zero in zero_stages:
                model = per_gpu_model_state_bytes(args.params, zero, tp, pp, dp)
                for recompute in ("none", "selective", "full"):
                    tried += 1
                    act = per_gpu_activation_bytes(
                        args.seq_len, args.micro_bs, args.hidden, args.heads,
                        args.layers, recompute, tp, pp,
                    )
                    mem_gb = (model + act) / GB
                    if mem_gb > args.hbm_gb:
                        continue  # ③ 显存关：放不下直接淘汰

                    hours = train_time_hours(
                        args.params, args.tokens, args.gpus,
                        args.peak_tflops, args.mfu, recompute,
                    )
                    candidates.append(dict(
                        tp=tp, pp=pp, dp=dp, zero=zero, recompute=recompute,
                        mem_gb=mem_gb, bubble=bubble, hours=hours,
                        effective_hours=hours / (1 - bubble),
                    ))

    # ④ 排序：先看扣掉气泡后的时长；同分时优先小 TP、小 PP。
    #    不能把「是否 OOM」当排序键——走到这里的都已经不 OOM 了（命门⑤）。
    #    同分优先小 TP/PP 是工程启发式：本模型没给 TP/PP 通信息定价，
    #    少切一层就少一份未计入的成本，显存够用就别主动上 TP。
    candidates.sort(key=lambda r: (r["effective_hours"], r["tp"], r["pp"], r["mem_gb"]))

    print(f"\n== scan：{args.params}B / {args.gpus} 卡 / s={args.seq_len} / "
          f"b={args.micro_bs} / HBM={args.hbm_gb:g} GB ==")
    print(f"  约束：tp<=8、tp|heads 且 tp·pp·dp={args.gpus}，{m_desc}，ZeRO-0/1")
    print(f"  枚举 {tried} 种组合，显存放得下 {len(candidates)} 种")

    if not candidates:
        print("  没有可行配置：先加卡、加 PP/TP，或换 full 重算。")
        return

    top_n = 20
    print(f"\n  {'#':>2}  {'TP':>2} {'PP':>3} {'DP':>4}  {'ZeRO':>4}  "
          f"{'recompute':<9}  {'每卡GB':>8}  {'气泡':>6}  {'纯算力h':>9}  {'扣气泡h':>9}")
    for i, r in enumerate(candidates[:top_n], 1):
        print(f"  {i:>2}  {r['tp']:>2} {r['pp']:>3} {r['dp']:>4}  "
              f"{'Z' + str(r['zero']):>4}  {r['recompute']:<9}  "
              f"{r['mem_gb']:>8.2f}  {r['bubble']:>6.1%}  "
              f"{r['hours']:>9.1f}  {r['effective_hours']:>9.1f}")
    if len(candidates) > top_n:
        print(f"  …… 其余 {len(candidates) - top_n} 种可行配置未展示")

    print("\n  排序键 = 纯算力时长 / (1 - 气泡占比)，越小越快；同分时优先小 TP、小 PP。")
    print("  本模型未给 TP / ZeRO 通信定价：TP 越大不代表越快，只是每卡更省；")
    print("  真实系统里跨 NVLink 域的 TP 会继续掉 MFU，所以这里只硬性要求 tp<=8。")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Day 2 · 显存与算力账本计算器")
    p.add_argument("--params", type=float, required=True, help="参数量，单位 B")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--micro-bs", type=int, default=1)
    p.add_argument("--layers", type=int, default=32)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--dp", type=int, default=1)
    p.add_argument("--zero", type=int, choices=[0, 1, 2, 3], default=1,
                   help="单配置模式使用；--scan 目前只枚举 0/1")
    p.add_argument("--recompute", choices=["none", "selective", "full"], default="none")
    p.add_argument("--gpus", type=int, required=True)
    p.add_argument("--peak-tflops", type=float, default=989.0, help="H100 BF16 稠密峰值")
    p.add_argument("--hbm-gb", type=float, default=80.0)
    p.add_argument("--tokens", type=float, default=1.0, help="训练 token 总数，单位 T")
    p.add_argument("--mfu", type=float, default=0.4)
    p.add_argument("--micro-batches", type=int, default=0, help="m，0=取 2*pp")
    p.add_argument("--scan", action="store_true", help="枚举所有可行拓扑并排序")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.scan:
        # scan 自己枚举 tp/pp/dp，不再校验命令行里的那一组。
        scan(args)
        return 0

    if args.tp * args.pp * args.dp != args.gpus:
        print(f"约束不满足：tp*pp*dp={args.tp * args.pp * args.dp} "
              f"!= gpus={args.gpus}", file=sys.stderr)
        return 2

    build_report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# 三个验证案例（逐条跑，把输出表贴进 course/day02/README.md）：
#
# 案例1  7B / s=4096 / b=1 / TP1 PP1 DP8 / ZeRO-1 / H100
#   python3 capacity.py --params 7 --layers 32 --hidden 4096 --heads 32 \
#     --tp 1 --pp 1 --dp 8 --gpus 8 --zero 1 --recompute full --tokens 1
#   → 先看 recompute=none 会不会 OOM，再换 full——「激活占多少」就是这一问的答案
#
# 案例2  70B / TP8 PP1 DP16（128 卡）/ ZeRO-1
#   python3 capacity.py --params 70 --layers 80 --hidden 8192 --heads 64 \
#     --tp 8 --pp 1 --dp 16 --gpus 128 --zero 1 --recompute selective --tokens 1
#
# 案例3  405B / Llama3 论文配置 TP8 PP16 DP128（16384 卡）/ ZeRO-1
#   python3 capacity.py --params 405 --layers 126 --hidden 16384 --heads 64 \
#     --tp 8 --pp 16 --dp 128 --gpus 16384 --zero 1 --recompute full \
#     --micro-batches 32 --tokens 1
#
# 自检题5 先手算再验证：1000×H100、7B、1T token、MFU 0.4 → 约 29.5 小时
