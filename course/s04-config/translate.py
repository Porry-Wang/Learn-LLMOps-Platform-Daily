#!/usr/bin/env python3
"""S4 · 配置翻译链：五组字段 → 四层优先级 → RuntimeSpec → 容器。

这一课只回答一个问题：**页面上这个字段，最后怎么变成容器里的东西。**

平台侧实现在 `platform/src/porryllmops/config.py`。这份脚本把它跑一遍，
每一段对应九图第 3–4 节的一张表：

    1) 五组字段：页面为什么长           —— FIELD_GROUPS / EXECUTED_FIELDS
    2) 四层优先级：冲突时谁赢           —— merge_config / sources
    3) 四问：字段怎么走完全程           —— trace()
    4) RuntimeSpec：容器拿到什么        —— build_runtime_spec()
    5) init 与 start 是两个进程         —— render_commands()
    6) 私有变量：存真值、出口掩码       —— mask_config()
    7) 内容哈希：改备注不产生新版本     —— compute_content_hash()

跑法（仓库根目录）：

    .venv/bin/python course/s04-config/translate.py

边界：本 toy 不读 YAML、不生成 k8s 对象；merge 的输入是 dict。
真实系统的模板来自平台库表，注入来自控制器，落点是 PodSpec。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许直接 `python course/s04-config/translate.py`：把 platform/src 加进 sys.path。
_SRC = Path(__file__).resolve().parents[2] / "platform" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from porryllmops.config import (  # noqa: E402
    EXECUTED_FIELDS,
    FIELD_GROUPS,
    FIELD_TRACE,
    GROUPS,
    TEMPLATES,
    build_runtime_spec,
    compute_content_hash,
    mask_config,
    merge_config,
    render_commands,
    trace,
)

TASK = "finetune-7b"
RUN = "run-0001"


def _rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _demo_groups() -> None:
    _rule("1) 五组字段：页面为什么长")
    for key, label in GROUPS:
        fields = FIELD_GROUPS[key]
        executed = [f for f in fields if f in EXECUTED_FIELDS]
        print(f"{label:<8} {len(fields):>2} 个字段")
        print(f"         全部：{', '.join(fields)}")
        print(
            f"         进容器：{', '.join(executed) if executed else '（不直接进容器）'}"
        )
    print()
    print("身份组和保障组决定校验、通知、采集策略，但不改变这次容器的形状。")


def _demo_priority():
    _rule("2) 四层优先级：冲突时谁赢")
    merged, sources = merge_config(
        TEMPLATES["llm-ft-v3"],
        {"max_steps": 8000, "lr": 2e-4},
        {"lr": 1e-4, "data_path": "oss://team-a/corpus-v3"},
        {"image_digest": "sha256:9f2c"},
    )
    print(f"{'字段':<16}{'最终值':<30}{'来源'}")
    print("-" * 62)
    for key in ("lr", "max_steps", "batch_size", "data_path", "image_digest"):
        print(f"{key:<16}{str(merged[key]):<30}{sources[key]}")
    print()
    print("规则：template < task < run_override < platform_injected；")
    print("      None 视为「这层没写」，不覆盖上层。")
    return merged, sources


def _demo_trace(merged, sources) -> None:
    _rule("3) 四问：一个字段怎么走完全程")
    for field in ("data_path", "ckpt_path", "tf_events_path", "image", "start_cmd"):
        answered = trace(field, merged, sources)
        print(f"[{field}]  source={answered['source']}  value={answered['value']}")
        print(f"  哪里设  {answered['where']}")
        print(f"  怎么存  {answered['stored']}")
        print(f"  怎么注入 {answered['injected']}")
        print(f"  哪里读  {answered['read']}")
        print()
    print(f"可追踪字段共 {len(FIELD_TRACE)} 个：{', '.join(sorted(FIELD_TRACE))}")


def _demo_spec(merged) -> None:
    _rule("4) RuntimeSpec：容器拿到什么")
    spec = build_runtime_spec(
        merged, task_name=TASK, run_id=RUN, attempt=1, nodes=1, gpus_per_node=8
    )
    print(f"image     {spec.image}")
    print(f"nodes×gpu {spec.nodes} × {spec.gpus_per_node} = world_size {spec.world_size}")
    print("mounts")
    for inside, source in spec.mounts:
        print(f"  {inside:<8} ← {source}")
    print("env")
    for name in sorted(spec.env):
        print(f"  {name:<24} {spec.env[name]}")
    print()
    print("RANK / LOCAL_RANK 不在这里：那是 torchrun 在每个进程里设的，")
    print("平台只负责 WORLD_SIZE 这类「整组共享」的变量。")


def _demo_commands(merged) -> None:
    _rule("5) init 与 start 是两个独立进程")
    merged = dict(merged)
    merged["init_cmd"] = "export HTTP_PROXY=http://proxy:3128 && python prepare.py"
    merged["start_cmd"] = "torchrun --nproc_per_node=8 train.py"
    spec = build_runtime_spec(merged, task_name=TASK, run_id=RUN)
    for name, command in render_commands(spec):
        print(f"{name:<6} {command}")
    print()
    print("init 里 export 的 HTTP_PROXY 不会出现在 start 的环境里：")
    print("两个 `sh -c` 各起一个进程，环境只来自平台注入。")
    print(f"spec.env 里有 HTTP_PROXY 吗？{'HTTP_PROXY' in spec.env}")


def _demo_mask(merged) -> None:
    _rule("6) 私有变量：存储要真值，出口要掩码")
    merged = dict(merged)
    merged["hf_token"] = "hf_this_is_a_real_token"
    merged["env"] = {"WANDB_API_KEY": "wb-real-1234", "NCCL_DEBUG": "INFO"}

    shown = mask_config(merged)
    print("给页面 / 接口 / 日志看：")
    print(f"  hf_token          {shown['hf_token']}")
    print(f"  env.WANDB_API_KEY {shown['env']['WANDB_API_KEY']}")
    print(f"  env.NCCL_DEBUG    {shown['env']['NCCL_DEBUG']}")
    print()
    print("容器启动时用的是存储里的真值（否则训练连不上任何服务）；")
    print("掩码只发生在出口：打印、API 响应、事件 payload、复制任务。")


def _demo_hash() -> None:
    _rule("7) 内容哈希：改备注不产生新版本")
    base = {"lr": 1e-4, "seq_len": 2048, "data_path": "oss://team-a/corpus-v3"}
    same = dict(base, comment="换个说法重提一次", description="备注")
    other = dict(base, lr=2e-4)
    print(f"base            {compute_content_hash(base)[:23]}…")
    print(f"+ 备注          {compute_content_hash(same)[:23]}…  ← 同一个版本")
    print(f"lr 1e-4 → 2e-4  {compute_content_hash(other)[:23]}…  ← 新版本")


def _selfcheck() -> None:
    """四条断言：优先级、四问、命令隔离、掩码。"""
    merged, sources = merge_config(
        {"lr": 3e-4}, {"lr": 2e-4}, {"lr": 1e-4}, {"lr": 5e-5}
    )
    assert merged["lr"] == 5e-5 and sources["lr"] == "platform_injected"

    for field in FIELD_TRACE:
        answered = trace(field)
        assert all(answered[q] for q in ("where", "stored", "injected", "read"))

    spec = build_runtime_spec(
        {"init_cmd": "export FOO=1", "start_cmd": "train.py",
         "hf_token": "hf_x", "env": {"HF_TOKEN": "hf_x"}},
        task_name="t", run_id="run-1",
    )
    assert "FOO" not in spec.env
    assert spec.public()["env"]["HF_TOKEN"] == "***"
    assert spec.env["HF_TOKEN"] == "hf_x"

    print("四条断言全部成立：最高层赢 / 四问答全 / export 不跨进程 / 掩码只在出口")


if __name__ == "__main__":
    _demo_groups()
    merged, sources = _demo_priority()
    _demo_trace(merged, sources)
    _demo_spec(merged)
    _demo_commands(merged)
    _demo_mask(merged)
    _demo_hash()
    print()
    _selfcheck()
