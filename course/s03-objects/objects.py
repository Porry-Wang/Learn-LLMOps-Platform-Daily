#!/usr/bin/env python3
"""S3 · 对象模型：六张表 + 状态机（完整阅读版）

这一课只回答一个问题：**训练平台到底在记什么。**
平台版实现在 `platform/src/porryllmops/models.py`（形状）和 `store.py`（落盘 + 事件）。
这份文件把「形状」和「规矩」抽出来单放一份：去掉持久化、去掉配置合并，
只留最不能忘的部分。读完它，回头读 models.py 就只剩细节。

对象图（四条边都是 1:N；下游存上游 id，上游不知道下游存在）：

    Task ─→ ConfigVersion ─→ Run ─→ Worker
                              └──→ Artifact
    Event ─(entity_type, entity_id)→ 指向上面任意一行

四条不变量（每条在文件末尾都有一条断言盯着）：

    1. 配置冻结      ConfigVersion 建好就改不动；Run 只存 config_version_id。
    2. 单向引用      下游存上游 id；Task 表里没有 run_ids 这一列。
    3. 终态不复活    终态出边为空。要再跑走 begin_retry()：同一个 Run，attempt+1。
    4. 三个成功分开   exit_code / ckpt_state / loadable 三列，不压成一个 status。

阅读顺序：

    1) 状态机 TRANSITIONS —— 只有 8 条边，值得背下来
    2) Run.finish() / Run.begin_retry() —— 两个改状态的入口，各自动哪些字段
    3) ConfigVersion —— 为什么 frozen=True 还不够，还要 MappingProxyType
    4) 文件末尾 _selfcheck() —— 六条断言，就是六条不变量

跑自检：  .venv/bin/python course/s03-objects/objects.py

边界（讲报告时主动说）：内容哈希的真实计算在 S4 的配置翻译链里，这里
content_hash 只是个必填字段；真实系统的落盘是数据库事务，不是 dataclass。
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple

# ---------------------------------------------------------------------------
# 1. 状态机：Run 的 state 只表达「进程活着还是死了」
# ---------------------------------------------------------------------------

PENDING = "pending"          # 已收单，还没轮到
SCHEDULING = "scheduling"    # 在找机器 / 等资源
RUNNING = "running"          # 进程起来了
SUCCEEDED = "succeeded"      # 进程退出码 0（注意：不代表存全了）
FAILED = "failed"
CANCELLED = "cancelled"

ALL_STATES: Tuple[str, ...] = (PENDING, SCHEDULING, RUNNING, SUCCEEDED, FAILED, CANCELLED)
TERMINAL_STATES: FrozenSet[str] = frozenset({SUCCEEDED, FAILED, CANCELLED})

# 只有 8 条边。终态刻意没有出边——「复活」不是状态迁移，是新的一次尝试。
TRANSITIONS: Dict[str, FrozenSet[str]] = {
    PENDING: frozenset({SCHEDULING, CANCELLED}),
    SCHEDULING: frozenset({RUNNING, FAILED, CANCELLED}),
    RUNNING: frozenset({SUCCEEDED, FAILED, CANCELLED}),
    SUCCEEDED: frozenset(),
    FAILED: frozenset(),
    CANCELLED: frozenset(),
}

# 三个成功里的第二个：断点续训要的东西存到什么程度
CKPT_NONE = "none"        # 什么都没存
CKPT_PARTIAL = "partial"  # 只有权重；优化器状态、global step 没了 → 不能续训
CKPT_COMPLETE = "complete"


class IllegalTransition(ValueError):
    """状态机不允许的迁移。"""


# ---------------------------------------------------------------------------
# 2. 角色：回答「这个进程挂了，这次训练还算不算活着」
# ---------------------------------------------------------------------------

TRAINER = "trainer"          # 算梯度、做 all-reduce；少一个，gang 就不完整
SIDECAR = "sidecar"          # 日志 / 指标上报；挂了不影响训练
COORDINATOR = "coordinator"  # rendezvous、传 ckpt；挂了失去控制面

# 角色不是给人看的备注，它直接决定「挂了怎么办」——这张表就是处置策略。
ROLE_ON_LOST: Dict[str, str] = {
    TRAINER: "fail_run",            # 训练没法继续，判失败，走重试
    SIDECAR: "restart_worker",      # 只重启这个进程，Run 不动
    COORDINATOR: "degrade_run",     # Run 降级继续跑，但记 health=degraded
}


# ---------------------------------------------------------------------------
# 3. 六张表
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """用户的意图：一条可编辑的模板。改它不影响任何已有 Run。"""

    id: str
    name: str
    defaults: Dict[str, Any] = field(default_factory=dict)  # 用户填的参数
    template: str = "llm-ft-v3"                             # 用哪个基线模板
    rev: int = 0                                            # 乐观锁计数

    def __post_init__(self) -> None:
        self.defaults = dict(self.defaults)   # 别和调用方共享同一个 dict


@dataclass(frozen=True)
class ConfigVersion:
    """一次冻结的参数快照。Run 只存它的 id，不存值。

    三个字段各有用途，缺一层就降一档：
        merged        最终生效值          → 复现
        sources       每个值来自哪一层     → 溯源（用户填的？平台注入的？）
        content_hash  执行相关字段的哈希   → 去重（同配置不产生新版本）
    """

    id: str
    task_id: str
    merged: Mapping[str, Any]
    sources: Mapping[str, str]
    content_hash: str

    def __post_init__(self) -> None:
        # frozen=True 只挡住 `cv.merged = {...}` 这种重新赋值，
        # 挡不住 `cv.merged["lr"] = 1`。所以 dict 还要包一层只读视图。
        object.__setattr__(self, "merged", MappingProxyType(dict(self.merged)))
        object.__setattr__(self, "sources", MappingProxyType(dict(self.sources)))


@dataclass
class Run:
    """一次执行 —— 更准确地说，是「一次尝试的容器」。

    整个对象里只有 state 在说进程死活；「存全了吗」「能用吗」各占一列。
    """

    id: str
    task_id: str
    config_version_id: str        # ← 配置只存 id，这是不变量 1
    state: str = PENDING
    attempt: int = 1              # 第几次尝试；重试不换 Run
    health: str = "ok"            # ok | degraded（coordinator 掉了）
    exit_code: Optional[int] = None
    ckpt_state: str = CKPT_NONE
    loadable: Optional[bool] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    rev: int = 0

    # -- 入口 A：正常推进 ----------------------------------------------------
    def advance(self, to: str, now: str) -> None:
        if to not in ALL_STATES:
            raise ValueError(f"未知状态: {to}")
        if self.state in TERMINAL_STATES:
            raise IllegalTransition(
                f"run {self.id} 已在终态 {self.state}；要再跑请用 begin_retry()"
            )
        if to not in TRANSITIONS[self.state]:
            raise IllegalTransition(
                f"run {self.id}: {self.state} → {to} 不是合法边"
                f"（可选 {sorted(TRANSITIONS[self.state])}）"
            )
        self.state = to
        if to == RUNNING:
            self.started_at = self.started_at or now
        if to in TERMINAL_STATES:
            self.ended_at = now

    # -- 入口 B：收尾 --------------------------------------------------------
    def finish(self, exit_code: int, ckpt_state: str, loadable: bool, now: str) -> None:
        """state 只由退出码决定；另外两个事实单独记，不许互相推断。"""
        if self.state != RUNNING:
            raise IllegalTransition(f"只有 running 能收尾，当前 {self.state}")
        if ckpt_state not in (CKPT_NONE, CKPT_PARTIAL, CKPT_COMPLETE):
            raise ValueError(f"未知 ckpt 状态: {ckpt_state}")
        self.exit_code = exit_code
        self.ckpt_state = ckpt_state
        self.loadable = loadable
        self.state = SUCCEEDED if exit_code == 0 else FAILED
        self.ended_at = now

    # -- 入口 C：重试 --------------------------------------------------------
    def begin_retry(self, now: str) -> None:
        """同一个 Run，attempt+1，回到 pending。

        不是「复活终态」：failed 是对第 N 次尝试的判定，旧事实留在事件流里；
        这一行的使命是装「当前这次尝试」，所以上一轮的事实必须清空，
        否则一个字段同时装着上次的 137 和这次的进行中。
        """
        if self.state != FAILED:
            raise IllegalTransition(f"只有 failed 能重试，当前 {self.state}")
        self.attempt += 1
        self.state = PENDING
        self.health = "ok"
        self.started_at = None
        self.ended_at = None
        self.exit_code = None
        self.ckpt_state = CKPT_NONE
        self.loadable = None

    @property
    def fully_succeeded(self) -> bool:
        """四个条件同时成立，才算「这次的产物可以拿来接着训」。"""
        return (
            self.state == SUCCEEDED
            and self.exit_code == 0
            and self.ckpt_state == CKPT_COMPLETE
            and self.loadable is True
        )


@dataclass
class Worker:
    """Run 里的一个进程。role 决定它挂了之后 Run 怎么办。"""

    id: str
    run_id: str
    rank: int
    role: str = TRAINER
    node: str = ""
    state: str = "pending"   # pending|starting|running|restarting|lost|exited|superseded
    attempt: int = 1         # 和所属 Run 的 attempt 对齐；旧一轮的会被标 superseded
    exit_code: Optional[int] = None


@dataclass(frozen=True)
class Artifact:
    """一件产物。独立成行，才能被多个后续 Run 引用。"""

    id: str
    run_id: str
    attempt: int             # 第几次尝试存下的——重试不覆盖上一轮的产物
    kind: str                # checkpoint | log | metrics | ...
    path: str
    checksum: str
    manifest: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))


@dataclass(frozen=True)
class Event:
    """append-only 的流水：谁都不改，只往后加。

    不建六条外键，靠 (entity_type, entity_id) 两个字符串指向任意一行，
    所以一张表能记全部对象的全部历史。
    """

    id: str
    entity_type: str          # task | config | run | worker | artifact
    entity_id: str
    type: str                 # run.created / run.failed / ...
    ts: str
    source: str = "platform"
    payload: Mapping[str, Any] = field(default_factory=dict)
    corr_id: Optional[str] = None   # 串起「一次用户操作」引发的多条事件

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))


# ---------------------------------------------------------------------------
# 4. 自检：六条断言，就是四条不变量 + 两个边界
# ---------------------------------------------------------------------------

def _selfcheck() -> None:
    now = "2026-10-29T00:00:00Z"

    # 不变量 4：退出码 0 但只存了权重 → state 是 succeeded，但不可续训
    ok_run = Run(id="run-1", task_id="t-1", config_version_id="cv-1")
    ok_run.advance(SCHEDULING, now)
    ok_run.advance(RUNNING, now)
    ok_run.finish(exit_code=0, ckpt_state=CKPT_PARTIAL, loadable=True, now=now)
    assert ok_run.state == SUCCEEDED and not ok_run.fully_succeeded

    # 不变量 3：终态不能直接改回 running
    try:
        ok_run.advance(RUNNING, now)
        raise AssertionError("终态居然被复活了")
    except IllegalTransition:
        pass

    # 不变量 3（续）：succeeded 不能重试 —— 重试是给 failed 用的
    try:
        ok_run.begin_retry(now)
        raise AssertionError("succeeded 被重试了")
    except IllegalTransition:
        pass

    # 不变量 3（续）：failed 重试 = 同一个 Run，attempt+1，清空上一轮事实
    bad_run = Run(id="run-2", task_id="t-1", config_version_id="cv-1")
    bad_run.advance(SCHEDULING, now)
    bad_run.advance(RUNNING, now)
    bad_run.finish(exit_code=137, ckpt_state=CKPT_PARTIAL, loadable=False, now=now)
    assert bad_run.state == FAILED and bad_run.exit_code == 137
    bad_run.begin_retry(now)
    assert (bad_run.attempt, bad_run.state, bad_run.exit_code,
            bad_run.ckpt_state, bad_run.loadable) == (2, PENDING, None, CKPT_NONE, None)

    # 非法边：pending 不能直接 succeeded（必须先 scheduling → running）
    fresh = Run(id="run-3", task_id="t-1", config_version_id="cv-1")
    try:
        fresh.advance(SUCCEEDED, now)
        raise AssertionError("非法边被放行了")
    except IllegalTransition:
        pass

    # 不变量 1：ConfigVersion 冻结 —— 重新赋值和原地改都要被拦住
    cv = ConfigVersion(id="cv-1", task_id="t-1", merged={"lr": 1e-4},
                       sources={"lr": "run_override"}, content_hash="sha256:x")
    for mutate in (lambda: setattr(cv, "merged", {}),
                   lambda: cv.merged.__setitem__("lr", 9.9)):
        try:
            mutate()
            raise AssertionError("ConfigVersion 被改了")
        except (AttributeError, TypeError):
            pass

    # 不变量 2：Task 没有 run_ids 这一列 —— 关系只能从下游往上游查
    assert "run_ids" not in {f.name for f in fields(Task)}

    print("六条断言全部成立：三个成功分开 / 终态不复活 / 重试不换 Run /")
    print("             配置冻结 / 单向引用 / 非法边拒绝")


if __name__ == "__main__":
    _selfcheck()
