#!/usr/bin/env python3
"""S2（下半）：checkpoint 里到底要存什么。

三个实验，对应三句要能讲清的话：

    实验 1  存了什么
        打开 ckpt 看结构：model 是“名字→张量”，optimizer 里是 m / v，
        step 是整数，config 是形状说明书。

    实验 2  为什么要存 optimizer
        A 组：存全套（model + optimizer + step）
        C 组：只存 model（optimizer 从零开始）
        对照组：中间完全不中断，一次跑完 60 步。
        比较续训后每一步的 loss 与“参数更新量 ‖Δθ‖”。
        A 组应当与对照组逐位一致；C 组第一步会明显变大（刹车没了）。

    实验 3  config 有什么用
        把 hidden 维度改一倍，再用同一个 state_dict 加载 —— 立刻 shape mismatch。
        这就是“必须先按 config 造出空模型，才有地方放权重”。

运行（单进程，不需要 torchrun）：
    ../../.venv/bin/python ckpt.py
"""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- 全局设定：一组固定的小数字，方便手算 ----
SEED = 0
TOTAL_STEPS = 60          # 计划训练总步数
FAULT_AT = 30             # 第 30 步后“机器挂了”
LR = 0.05
BATCH = 32

CONFIG = {"in_dim": 16, "hidden": 64, "out_dim": 1}


class TinyMLP(nn.Module):
    """三层 MLP。结构简单到可以手算，但足以演示优化器状态。"""

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(config: dict, seed: int = SEED) -> TinyMLP:
    """按 config 造一个空模型；同一 seed 保证每次初始化完全一致。"""
    torch.manual_seed(seed)
    return TinyMLP(**config)


def make_batches(num_steps: int, seed: int = SEED):
    """预先把所有 batch 生成好，保证三次实验读到的数据一模一样。"""
    torch.manual_seed(seed)
    batches = []
    for _ in range(num_steps):
        x = torch.randn(BATCH, CONFIG["in_dim"])
        # 目标是一个固定线性函数 + 少量噪声：简单、稳定、可复现
        w = torch.arange(CONFIG["in_dim"], dtype=torch.float32).reshape(-1, 1) * 0.1
        y = x @ w + 0.01 * torch.randn(BATCH, 1)
        batches.append((x, y))
    return batches


def flat_params(model: nn.Module) -> torch.Tensor:
    """把所有参数压成一维向量，用来量“这一步参数一共动了多远”。"""
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def run_steps(model, optimizer, batches, start_step: int, steps: int):
    """跑 steps 步，记录 (step, loss, ‖Δθ‖)。"""
    rows = []
    for i in range(steps):
        step = start_step + i
        x, y = batches[step]

        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x), y)
        loss.backward()

        before = flat_params(model)
        optimizer.step()
        delta = (flat_params(model) - before).norm().item()

        rows.append((step + 1, loss.item(), delta))
    return rows


def describe_ckpt(ckpt: dict, path: Path) -> None:
    """实验 1：把 ckpt 拆开，看看每一层到底是什么。"""
    print("\n=== 实验 1：ckpt 里到底有什么 ===")
    print(f"文件：{path}  大小：{path.stat().st_size / 1024:.1f} KB")
    print(f"顶层键：{list(ckpt)}")

    model_sd = ckpt["model"]
    numel = sum(t.numel() for t in model_sd.values())
    print(f"\n[model] 张量数={len(model_sd)}  参数总数={numel}")
    for name, tensor in list(model_sd.items())[:3]:
        print(f"  {name:<22} shape={tuple(tensor.shape)} dtype={tensor.dtype}")
    print("  ...（只列前 3 个）")

    opt_sd = ckpt["optimizer"]
    print(f"\n[optimizer] 顶层键：{list(opt_sd)}")
    first_key = next(iter(opt_sd["state"]))
    first_state = opt_sd["state"][first_key]
    print(f"  state 的 key 是参数序号，例如 {first_key}；里面是：")
    for k, v in first_state.items():
        shape = tuple(v.shape) if torch.is_tensor(v) else "标量"
        print(f"    {k:<12} shape={shape}")
    group = {k: v for k, v in opt_sd["param_groups"][0].items() if k != "params"}
    print(f"  param_groups[0]（超参本身也要存）：{group}")

    print(f"\n[step]    {ckpt['step']}  （int，决定 lr 调度和横轴）")
    print(f"[config]  {ckpt['config']}  （决定模型的张量骨架）")


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="s02_ckpt_"))
    print(f"ckpt 临时目录：{workdir}")
    print(f"模型 config：{CONFIG}   LR={LR}   BATCH={BATCH}")

    batches = make_batches(TOTAL_STEPS)

    # ---------- 对照组：一次跑完 60 步，中间不中断 ----------
    ref_model = build_model(CONFIG)
    ref_opt = torch.optim.Adam(ref_model.parameters(), lr=LR)
    ref_rows = run_steps(ref_model, ref_opt, batches, 0, TOTAL_STEPS)

    # ---------- 前 30 步：两次实验共用同一段“故障前”的历史 ----------
    def train_first_half():
        model = build_model(CONFIG)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        run_steps(model, opt, batches, 0, FAULT_AT)
        return model, opt

    # ---------- A 组：存全套 ----------
    model_a, opt_a = train_first_half()
    ckpt_full = {
        "model": model_a.state_dict(),
        "optimizer": opt_a.state_dict(),
        "step": FAULT_AT,
        "config": dict(CONFIG),
    }
    full_path = workdir / "ckpt-step-0000030-full.pt"
    torch.save(ckpt_full, full_path)
    describe_ckpt(ckpt_full, full_path)

    # 模拟“新进程”：重建空模型、重建优化器，再从文件恢复
    resume_model = build_model(CONFIG)
    resume_opt = torch.optim.Adam(resume_model.parameters(), lr=LR)
    loaded = torch.load(full_path)
    resume_model.load_state_dict(loaded["model"])
    resume_opt.load_state_dict(loaded["optimizer"])
    rows_full = run_steps(
        resume_model, resume_opt, batches, loaded["step"], TOTAL_STEPS - FAULT_AT
    )

    # ---------- C 组：只存 model ----------
    model_c, _ = train_first_half()
    ckpt_model_only = {"model": model_c.state_dict(), "step": FAULT_AT,
                       "config": dict(CONFIG)}
    model_only_path = workdir / "ckpt-step-0000030-modelonly.pt"
    torch.save(ckpt_model_only, model_only_path)

    resume_model_c = build_model(CONFIG)
    resume_opt_c = torch.optim.Adam(resume_model_c.parameters(), lr=LR)  # 全新，没有 m/v
    resume_model_c.load_state_dict(torch.load(model_only_path)["model"])
    rows_model_only = run_steps(
        resume_model_c, resume_opt_c, batches, FAULT_AT, TOTAL_STEPS - FAULT_AT
    )

    # ---------- 实验 2：对比 ----------
    print("\n=== 实验 2：续训后的前 8 步 ===")
    print(f"{'step':>5} {'对照 loss':>11} {'A组 loss':>11} {'A组Δθ':>8} "
          f"{'C组 loss':>11} {'C组Δθ':>8}")
    for i in range(8):
        ref_step, ref_loss, ref_delta = ref_rows[FAULT_AT + i]
        _, a_loss, a_delta = rows_full[i]
        _, c_loss, c_delta = rows_model_only[i]
        print(f"{ref_step:>5} {ref_loss:>11.6f} {a_loss:>11.6f} {a_delta:>8.3f} "
              f"{c_loss:>11.6f} {c_delta:>8.3f}")

    ref_tail = [r[1] for r in ref_rows[FAULT_AT:]]
    a_tail = [r[1] for r in rows_full]
    c_tail = [r[1] for r in rows_model_only]

    max_a_diff = max(abs(a - r) for a, r in zip(a_tail, ref_tail))
    max_c_diff = max(abs(c - r) for c, r in zip(c_tail, ref_tail))
    print(f"\nA 组与对照组的最大 loss 偏差：{max_a_diff:.2e}")
    print(f"C 组与对照组的最大 loss 偏差：{max_c_diff:.2e}")
    print(f"A 组第一步 Δθ = {rows_full[0][2]:.3f}，"
          f"对照第一步 Δθ = {ref_rows[FAULT_AT][2]:.3f}")
    print(f"C 组第一步 Δθ = {rows_model_only[0][2]:.3f}")

    # ---------- 实验 3：config 决定形状 ----------
    print("\n=== 实验 3：config 变了，同一个 state_dict 就装不进去 ===")
    bad_config = dict(CONFIG, hidden=CONFIG["hidden"] * 2)
    print(f"原 config hidden={CONFIG['hidden']}，改成 hidden={bad_config['hidden']} 后加载：")
    try:
        build_model(bad_config).load_state_dict(ckpt_full["model"])
    except RuntimeError as exc:
        for line in str(exc).splitlines()[:3]:
            print("  " + line)

    print(f"\n文件留在：{workdir}")
    print("想手动看：>>> ckpt = torch.load('...'); print(ckpt['optimizer'].keys())")


if __name__ == "__main__":
    main()
