"""S3 对象模型：六张表的形状与规矩。

    Task ─1:N→ ConfigVersion ─1:N→ Run ─1:N→ Worker
                                     └─1:N→ Artifact

    Event ──(entity_type, entity_id)──→ 指向上面任意一行

四条不变量（tests/test_store.py 逐条钉住）：

    1. 配置冻结：ConfigVersion 不可变；Run 只存 config_version_id，
       不存 task_id 的实时值，也不存用户填的「原始输入」。
    2. 单向引用：下游存上游 id，上游不知道下游存在。
    3. 终态不复活：succeeded / failed / cancelled 不能直接改回 running；
       要再跑，只能 begin_retry()（同一个 Run，attempt+1）或新建 Run。
    4. 三个成功分开：exit_code / ckpt_state / loadable 三列，
       不能压成一个 status。进程退出 0 不等于训练状态存全了。

toy 边界（讲报告时主动说）：

    · 内容哈希靠应用层比对；生产是数据库唯一索引。
    · frozen=True 只挡住进程内重新赋值，挡不住有人直接改 JSON 文件。
    · 角色策略只有三条分支；真实系统还有抢占、驱逐、跨机房重调度。
    · Run 是「尝试容器」：attempt 递增而 Run 不换，是为了让「同一件事
      重试 8 次」在数据里可区分于「换配置跑了 8 回」。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

__all__ = [
    # 状态与角色
    "STATE_PENDING",
    "STATE_SCHEDULING",
    "STATE_RUNNING",
    "STATE_SUCCEEDED",
    "STATE_FAILED",
    "STATE_CANCELLED",
    "ALL_STATES",
    "TERMINAL_STATES",
    "TRANSITIONS",
    "ROLE_TRAINER",
    "ROLE_SIDECAR",
    "ROLE_COORDINATOR",
    "ROLES",
    "ROLE_ON_LOST",
    "ACTION_FAIL_RUN",
    "ACTION_RESTART_WORKER",
    "ACTION_DEGRADE_RUN",
    "CKPT_NONE",
    "CKPT_PARTIAL",
    "CKPT_COMPLETE",
    # 模板与合并
    "DEFAULT_TEMPLATE",
    "TEMPLATES",
    "LAYERS",
    "merge_config",
    "compute_content_hash",
    "normalize_for_hash",
    # 对象
    "Task",
    "ConfigVersion",
    "Run",
    "Worker",
    "Artifact",
    "Event",
    "to_dict",
    # 错误
    "IllegalTransition",
    "Conflict",
    "NotFound",
]

# ---------------------------------------------------------------------------
# 状态机
# ---------------------------------------------------------------------------

STATE_PENDING = "pending"
STATE_SCHEDULING = "scheduling"
STATE_RUNNING = "running"
STATE_SUCCEEDED = "succeeded"
STATE_FAILED = "failed"
STATE_CANCELLED = "cancelled"

ALL_STATES: Tuple[str, ...] = (
    STATE_PENDING,
    STATE_SCHEDULING,
    STATE_RUNNING,
    STATE_SUCCEEDED,
    STATE_FAILED,
    STATE_CANCELLED,
)

TERMINAL_STATES = frozenset({STATE_SUCCEEDED, STATE_FAILED, STATE_CANCELLED})

# 合法边。终态出边为空：要重试走 begin_retry()，不走状态迁移。
TRANSITIONS: Dict[str, frozenset] = {
    STATE_PENDING: frozenset({STATE_SCHEDULING, STATE_CANCELLED}),
    STATE_SCHEDULING: frozenset({STATE_RUNNING, STATE_FAILED, STATE_CANCELLED}),
    STATE_RUNNING: frozenset({STATE_SUCCEEDED, STATE_FAILED, STATE_CANCELLED}),
    STATE_SUCCEEDED: frozenset(),
    STATE_FAILED: frozenset(),
    STATE_CANCELLED: frozenset(),
}

# ---------------------------------------------------------------------------
# 角色：回答「这个进程挂了，这次训练还算不算活着」
# ---------------------------------------------------------------------------

ROLE_TRAINER = "trainer"          # 算梯度、做 all-reduce；少一个 gang 就不完整
ROLE_SIDECAR = "sidecar"          # 日志/指标上报；挂了不影响训练
ROLE_COORDINATOR = "coordinator"  # rendezvous、传 ckpt；挂了失去控制面

ROLES: Tuple[str, ...] = (ROLE_TRAINER, ROLE_SIDECAR, ROLE_COORDINATOR)

ACTION_FAIL_RUN = "fail_run"
ACTION_RESTART_WORKER = "restart_worker"
ACTION_DEGRADE_RUN = "degrade_run"

ROLE_ON_LOST: Dict[str, str] = {
    ROLE_TRAINER: ACTION_FAIL_RUN,
    ROLE_SIDECAR: ACTION_RESTART_WORKER,
    ROLE_COORDINATOR: ACTION_DEGRADE_RUN,
}

# 三个成功里的第二个：训练状态存到什么程度
CKPT_NONE = "none"
CKPT_PARTIAL = "partial"    # 只有权重，没有优化器状态 / global step
CKPT_COMPLETE = "complete"

# ---------------------------------------------------------------------------
# 配置合并与哈希
# ---------------------------------------------------------------------------

DEFAULT_TEMPLATE = "llm-ft-v3"

# 平台预置基线。升级模板 = 改这里；历史 ConfigVersion 不受影响。
TEMPLATES: Dict[str, Dict[str, Any]] = {
    DEFAULT_TEMPLATE: {
        "lr": 3e-4,
        "batch_size": 32,
        "max_steps": 5000,
        "seq_len": 2048,
        "optimizer": "adamw",
        "precision": "bf16",
    },
}

# 合并优先级从低到高；sources 里记的层级名。
LAYERS: Tuple[str, ...] = ("template", "task", "run_override", "platform_injected")

# 这些字段不影响执行，不参与内容哈希（改备注不该产生新版本）。
NON_EXECUTION_KEYS = frozenset(
    {"name", "display_name", "description", "comment", "notes", "tags"}
)


def merge_config(
    template: Mapping[str, Any],
    task_defaults: Mapping[str, Any],
    overrides: Mapping[str, Any],
    injected: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """四层叠加，返回（最终生效值, 每个字段的来源层）。

    空值（None）视为「这层没写」，不覆盖上层。
    例：模板 lr=3e-4 + 本次覆盖 lr=1e-4 → merged["lr"]=1e-4，
    sources["lr"]="run_override"。
    """
    merged: Dict[str, Any] = {}
    sources: Dict[str, str] = {}

    for layer_name, layer in zip(
        LAYERS, (template, task_defaults, overrides, injected)
    ):
        for key, value in layer.items():
            if value is None:
                continue
            merged[key] = value
            sources[key] = layer_name

    return merged, sources


def normalize_for_hash(merged: Mapping[str, Any]) -> Dict[str, Any]:
    """只保留「影响执行」的字段：去元信息、去空值。"""
    return {
        key: value
        for key, value in merged.items()
        if key not in NON_EXECUTION_KEYS and value is not None
    }


def compute_content_hash(merged: Mapping[str, Any]) -> str:
    """稳定哈希：同一份执行配置永远得到同一个值。

    created_at 之类的时间字段不进 merged，自然不会进哈希。
    """
    payload = json.dumps(
        normalize_for_hash(merged), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 错误
# ---------------------------------------------------------------------------

class IllegalTransition(ValueError):
    """状态机不允许的迁移。"""


class Conflict(RuntimeError):
    """乐观锁版本不匹配：别人先改了。"""


class NotFound(KeyError):
    """按 id 查不到记录。"""


# ---------------------------------------------------------------------------
# 序列化
# ---------------------------------------------------------------------------

def _to_plain(value: Any) -> Any:
    """把 MappingProxyType / tuple 转成 JSON 能吃的类型。"""
    if isinstance(value, MappingProxyType):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


def to_dict(obj: Any) -> Dict[str, Any]:
    """dataclass → 普通 dict（字段集合与构造参数严格一致）。"""
    return {f.name: _to_plain(getattr(obj, f.name)) for f in fields(obj)}


# ---------------------------------------------------------------------------
# 六张表
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """用户的意图：一个可编辑的模板。改它不影响任何已有 Run。"""

    id: str
    name: str
    defaults: Dict[str, Any] = field(default_factory=dict)
    template: str = DEFAULT_TEMPLATE
    created_at: str = ""
    updated_at: str = ""
    rev: int = 0

    def __post_init__(self) -> None:
        # 复制一份，避免调用方改到 store 里的那份 dict
        self.defaults = dict(self.defaults)


@dataclass(frozen=True)
class ConfigVersion:
    """一次冻结的参数快照：merged 是值，sources 是来源。

    frozen=True 挡住重新赋值；MappingProxyType 挡住原地修改 dict。
    """

    id: str
    task_id: str
    merged: Mapping[str, Any]
    sources: Mapping[str, str]
    content_hash: str
    created_at: str = ""

    def __post_init__(self) -> None:
        # dict(...) 复制一份，避免和调用方共享引用
        object.__setattr__(self, "merged", MappingProxyType(dict(self.merged)))
        object.__setattr__(self, "sources", MappingProxyType(dict(self.sources)))


@dataclass
class Run:
    """一次执行。state 只表达进程生命周期；其余成功由独立字段表达。"""

    id: str
    task_id: str
    config_version_id: str
    state: str = STATE_PENDING
    attempt: int = 1
    health: str = "ok"            # ok | degraded（coordinator 掉了）
    created_at: str = ""
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    exit_code: Optional[int] = None
    ckpt_state: str = CKPT_NONE
    loadable: Optional[bool] = None
    idempotency_key: Optional[str] = None
    rev: int = 0

    def advance(self, to: str, now: str) -> None:
        """沿合法边迁移；终态出边为空，所以终态不可复活。"""
        if to not in ALL_STATES:
            raise ValueError(f"未知状态: {to}")
        if self.state in TERMINAL_STATES:
            raise IllegalTransition(
                f"run {self.id} 已在终态 {self.state}；要再跑请用 retry_run()"
            )
        if to not in TRANSITIONS[self.state]:
            raise IllegalTransition(
                f"run {self.id}: {self.state} → {to} 不是合法边"
                f"（可选 {sorted(TRANSITIONS[self.state])}）"
            )

        self.state = to
        if to == STATE_RUNNING:
            self.started_at = self.started_at or now
        if to in TERMINAL_STATES:
            self.ended_at = now

    def begin_retry(self, now: str) -> None:
        """自动重试：同一个 Run，attempt+1，回到 pending。

        这不是「复活终态」——failed 是对第 N 次尝试的判定；第 N+1 次
        是新的尝试，旧的失败记录原样留在事件流里。
        """
        if self.state != STATE_FAILED:
            raise IllegalTransition(
                f"只有 failed 能重试，run {self.id} 当前 {self.state}"
            )

        self.attempt += 1
        self.state = STATE_PENDING
        self.health = "ok"
        self.started_at = None
        self.ended_at = None
        self.exit_code = None
        self.ckpt_state = CKPT_NONE
        self.loadable = None

    def finish(
        self, exit_code: int, ckpt_state: str, loadable: bool, now: str
    ) -> None:
        """收尾：state 只由进程退出码决定，另外两个事实单独记。"""
        if self.state != STATE_RUNNING:
            raise IllegalTransition(
                f"只有 running 能收尾，run {self.id} 当前 {self.state}"
            )
        if ckpt_state not in (CKPT_NONE, CKPT_PARTIAL, CKPT_COMPLETE):
            raise ValueError(f"未知 ckpt 状态: {ckpt_state}")

        self.exit_code = exit_code
        self.ckpt_state = ckpt_state
        self.loadable = loadable
        self.state = STATE_SUCCEEDED if exit_code == 0 else STATE_FAILED
        self.ended_at = now

    @property
    def fully_succeeded(self) -> bool:
        """三个成功同时成立，才算「可以接着跑」。"""
        return (
            self.state == STATE_SUCCEEDED
            and self.exit_code == 0
            and self.ckpt_state == CKPT_COMPLETE
            and self.loadable is True
        )


@dataclass
class Worker:
    """Run 里的一个进程。role 决定它挂了之后 Run 该怎么办。"""

    id: str
    run_id: str
    rank: int
    role: str = ROLE_TRAINER
    node: str = ""
    state: str = "pending"        # pending|starting|running|restarting|lost|exited|superseded
    attempt: int = 1
    last_heartbeat: Optional[str] = None
    exit_code: Optional[int] = None
    rev: int = 0


@dataclass(frozen=True)
class Artifact:
    """一件产物。独立对象，才能被多个后续 Run 引用。"""

    id: str
    run_id: str
    attempt: int
    kind: str
    path: str
    checksum: str
    manifest: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))


@dataclass(frozen=True)
class Event:
    """append-only 的流水。entity_type + entity_id 指向任意一行。"""

    id: str
    entity_type: str
    entity_id: str
    type: str
    ts: str
    source: str = "platform"
    payload: Mapping[str, Any] = field(default_factory=dict)
    corr_id: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))
