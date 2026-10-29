"""PorryLLMOps 控制面 CLI（S3 演示）。

四个命令，对应六张表的四个日常动作：

    submit  写 Task / ConfigVersion / Run
    status  看 Run + Worker 的当前快照
    events  看事件流水
    retry   把 failed 的 Run 推进到 attempt+1

在仓库根目录下（PYTHONPATH 指向 platform/src）：

    export PYTHONPATH=platform/src
    .venv/bin/python -m porryllmops.cli --db demo.json submit --task finetune-7b \\
        --set lr=1e-4 --inject image_digest=sha256:9f2c --simulate ok
    .venv/bin/python -m porryllmops.cli --db demo.json status
    .venv/bin/python -m porryllmops.cli --db demo.json events run-0001
    .venv/bin/python -m porryllmops.cli --db demo.json submit --task broken-7b \\
        --simulate trainer-lost --gpus 8
    .venv/bin/python -m porryllmops.cli --db demo.json retry run-0002

retry 只对 failed 的 run 有效，所以演示里先制造一次失败再重试。

边界：--simulate 是演示驱动，按脚本改状态，不代表真实调度器。
生产里这些状态由 kubelet / 训练进程 / 控制器上报。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

# 直接 `python cli.py` 或在 PyCharm 里右键 Debug 时，本文件的 __package__ 是空的，
# 下面的相对导入会报 "no known parent package"。这里把包上下文补上：
# parents[1] 就是 platform/src，塞进 sys.path 后 porryllmops 才能被找到。
# 用 `python -m porryllmops.cli` 时 __package__ 已经有值，走不进这个分支。
if __package__ in (None, ""):
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "porryllmops"

from .models import (
    ROLE_COORDINATOR,
    ROLE_SIDECAR,
    ROLE_TRAINER,
    STATE_SCHEDULING,
    Conflict,
    ConfigVersion,
    IllegalTransition,
    NotFound,
    Run,
)
from .store import Store

__all__ = ["main", "build_parser"]

SCENARIOS = ("none", "ok", "sidecar-lost", "trainer-lost", "bad-ckpt")


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def _coerce(raw: str) -> Any:
    """1e-4 → float，true → bool，adamw → str。"""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _pairs(items: Optional[List[str]], flag: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"{flag} 需要 K=V 形式，收到 {item!r}")
        key, raw = item.split("=", 1)
        out[key.strip()] = _coerce(raw.strip())
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m porryllmops.cli",
        description="PorryLLMOps 训练控制面演示（六张表 + 事件流水）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "例子：\n"
            "  submit --task finetune-7b --set lr=1e-4 --simulate ok\n"
            "  submit --task finetune-7b --simulate trainer-lost --gpus 8\n"
            "  status / status run-0001\n"
            "  events run-0001 / events --type run.\n"
            "  retry run-0001\n"
        ),
    )
    parser.add_argument(
        "--db",
        default=os.environ.get("PORRYLLMOPS_DB", "porryllmops.json"),
        help="JSON 库文件路径（默认 $PORRYLLMOPS_DB 或 ./porryllmops.json）",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    submit = sub.add_parser("submit", help="提交一次运行")
    submit.add_argument("--task", required=True, help="任务名（不存在则新建）")
    submit.add_argument("--template", default="llm-ft-v3", help="平台模板")
    submit.add_argument(
        "--defaults", action="append", metavar="K=V",
        help="新建任务时写入的默认值（任务已存在则忽略）",
    )
    submit.add_argument(
        "--set", action="append", metavar="K=V", help="本次覆盖（run_override）",
    )
    submit.add_argument(
        "--inject", action="append", metavar="K=V", help="平台注入（镜像 digest 等）",
    )
    submit.add_argument("--idem", metavar="KEY", help="幂等键：同 key 只建一个 Run")
    submit.add_argument("--gpus", type=int, default=8, help="演示用 trainer 数")
    submit.add_argument(
        "--simulate", choices=SCENARIOS, default="none",
        help="演示场景：none(只建 Run) / ok / sidecar-lost / trainer-lost / bad-ckpt",
    )
    submit.set_defaults(handler=_cmd_submit)

    status = sub.add_parser("status", help="看 Run 当前快照")
    status.add_argument("run_id", nargs="?", help="不给则列出全部 Run")
    status.set_defaults(handler=_cmd_status)

    events = sub.add_parser("events", help="看事件流水")
    events.add_argument("run_id", nargs="?", help="只给这一个 Run 的事件")
    events.add_argument("--type", help="按前缀过滤，如 run.")
    events.add_argument("--corr", help="按 correlation_id 过滤")
    events.set_defaults(handler=_cmd_events)

    retry = sub.add_parser("retry", help="自动重试一个 failed 的 Run")
    retry.add_argument("run_id")
    retry.set_defaults(handler=_cmd_retry)

    return parser


# ---------------------------------------------------------------------------
# 输出小工具
# ---------------------------------------------------------------------------

def _print_config(cv: ConfigVersion) -> None:
    print(f"  config {cv.id}（{len(cv.merged)} 个字段）")
    for key in sorted(cv.merged):
        print(f"    {key:<18} {str(cv.merged[key]):<22} ← {cv.sources.get(key, '?')}")


def _print_run(store: Store, run: Run) -> None:
    task_name = store.get_task(run.task_id).name
    print(
        f"{run.id}  {task_name}  {run.state}  "
        f"attempt={run.attempt}  health={run.health}"
    )
    print(f"  config    {run.config_version_id}")
    print(
        f"  三个成功  exit={run.exit_code}  "
        f"ckpt={run.ckpt_state}  loadable={run.loadable}"
    )
    print(f"  可续跑    {'是' if run.fully_succeeded else '否'}")
    print(f"  起止      {run.started_at or '-'} / {run.ended_at or '-'}")

    workers = store.current_workers(run.id)
    if workers:
        print(f"  workers   {store.ready_workers(run.id)}/{len(workers)} running")
        missing = store.pending_trainers(run.id)
        if missing:
            detail = ", ".join(
                f"rank {m['rank']}@{m['node']}({m['state']})" for m in missing
            )
            print(f"  缺 trainer {detail}")


def _compact(payload: Any, width: int = 72) -> str:
    text = json.dumps(dict(payload), ensure_ascii=False)
    return text if len(text) <= width else text[: width - 1] + "…"


# ---------------------------------------------------------------------------
# 演示驱动
# ---------------------------------------------------------------------------

def _simulate(store: Store, run: Run, scenario: str, gpus: int) -> None:
    if gpus < 1:
        raise SystemExit("--gpus 必须 >= 1")

    for rank in range(gpus):
        store.add_worker(run.id, rank=rank, node=f"sh-a-gpu-{30 + rank:03d}")
    sidecar = store.add_worker(
        run.id, rank=gpus, role=ROLE_SIDECAR, node="sh-a-gpu-030"
    )
    coordinator = store.add_worker(
        run.id, rank=gpus + 1, role=ROLE_COORDINATOR, node="sh-a-gpu-030"
    )
    # sidecar / coordinator 先起，训练进程后起
    store.set_worker_state(sidecar.id, "running")
    store.set_worker_state(coordinator.id, "running")

    store.advance_run(run.id, STATE_SCHEDULING)
    trainers = [w for w in store.current_workers(run.id) if w.role == ROLE_TRAINER]

    # 先起 n-1 个，演示「数 trainer」而不是「等上游宣布」
    for worker in trainers[:-1]:
        store.set_worker_state(worker.id, "running")
    print(
        f"\n[simulate] {len(trainers) - 1}/{len(trainers)} trainer 就位 → "
        f"start_run_if_ready = {store.start_run_if_ready(run.id)}"
    )
    pending = store.pending_trainers(run.id)
    if pending:
        print(f"[simulate] 缺: {pending}")

    store.set_worker_state(trainers[-1].id, "running")
    print(
        f"[simulate] {len(trainers)}/{len(trainers)} trainer 就位 → "
        f"start_run_if_ready = {store.start_run_if_ready(run.id)}"
    )

    if scenario == "sidecar-lost":
        action = store.lose_worker(sidecar.id)
        print(
            f"[simulate] sidecar 掉线 → {action}；run 状态 "
            f"{store.get_run(run.id).state}（训练照跑）"
        )

    if scenario == "trainer-lost":
        victim = trainers[-1]
        action = store.lose_worker(victim.id)
        print(
            f"[simulate] trainer rank {victim.rank} 掉线 → {action}；"
            f"run 状态 {store.get_run(run.id).state}"
        )
        print(f"[simulate] 下一步：retry {run.id}")
        return

    task_name = store.get_task(run.task_id).name
    artifact = store.add_artifact(
        run.id,
        kind="checkpoint",
        path=f"s3://porry-ckpt/{task_name}/{run.id}/a{run.attempt}/step-1500",
        checksum="sha256:4b8e",
        manifest={"step": 1500, "parallel": {"tp": 1, "pp": 1, "dp": gpus}},
    )
    print(f"[simulate] ckpt {artifact.id} → {artifact.path}")

    if scenario == "bad-ckpt":
        store.finish_run(run.id, exit_code=0, ckpt_state="partial", loadable=True)
        print("[simulate] 退出 0、能加载，但只存了权重（Day 3 的 C 组）")
    else:
        store.finish_run(run.id, exit_code=0, ckpt_state="complete", loadable=True)


# ---------------------------------------------------------------------------
# 四个命令
# ---------------------------------------------------------------------------

def _cmd_submit(store: Store, args: argparse.Namespace) -> int:
    defaults = _pairs(args.defaults, "--defaults")
    overrides = _pairs(args.set, "--set")
    injected = _pairs(args.inject, "--inject")

    task = store.find_task_by_name(args.task)
    if task is None:
        task = store.create_task(args.task, defaults=defaults, template=args.template)
        print(f"新建任务 {task.id}  {task.name}（模板 {task.template}）")
    elif defaults:
        print(f"任务 {task.id} 已存在，忽略 --defaults")

    run = store.create_run(
        task.id,
        overrides=overrides,
        injected=injected,
        idempotency_key=args.idem,
    )
    print(f"run {run.id}  state={run.state}  attempt={run.attempt}")
    _print_config(store.run_config(run.id))

    if args.simulate != "none":
        _simulate(store, run, args.simulate, args.gpus)
        print()
        _print_run(store, store.get_run(run.id))
    return 0


def _cmd_status(store: Store, args: argparse.Namespace) -> int:
    if args.run_id:
        _print_run(store, store.get_run(args.run_id))
        return 0

    runs = store.list_runs()
    if not runs:
        print("还没有 run，先 submit 一个")
        return 0
    for run in runs:
        _print_run(store, run)
        print()
    return 0


def _cmd_events(store: Store, args: argparse.Namespace) -> int:
    items = store.events(entity_id=args.run_id, corr_id=args.corr)
    if args.type:
        items = [event for event in items if event.type.startswith(args.type)]
    if not items:
        print("没有匹配的事件")
        return 0

    for event in items:
        corr = f" [{event.corr_id}]" if event.corr_id else ""
        print(
            f"{event.ts}  {event.type:<16} {event.entity_id:<24}"
            f"{corr} {_compact(event.payload)}"
        )
    return 0


def _cmd_retry(store: Store, args: argparse.Namespace) -> int:
    run = store.retry_run(args.run_id)
    print(
        f"{run.id} → attempt {run.attempt}  state={run.state}  "
        f"config={run.config_version_id}（配置不变）"
    )
    print("提示：status 看 worker 是否重建；worker 要按新 attempt 重新 add")
    return 0


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    store = Store(args.db)
    try:
        return args.handler(store, args)
    except (NotFound, IllegalTransition, Conflict, ValueError) as exc:
        print(f"错误（{type(exc).__name__}）: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
