"""S3 持久化：六张表落一个 JSON 文件。

它只做四件事：

    1. 把 models.py 的对象存下来、按 id 取出来（每次都返回副本）。
    2. 每次改动追加 Event；events 只增不改。
    3. 写盘用 temp → fsync → rename，避免半个文件（和 ckpt 一样）。
    4. 三个护栏：content_hash 去重、idempotency_key 幂等、rev 乐观锁。

最小用法：

    >>> store = Store("db.json")
    >>> task = store.create_task("finetune-7b", defaults={"max_steps": 8000})
    >>> run = store.create_run(task.id, overrides={"lr": 1e-4})
    >>> store.advance_run(run.id, "scheduling")
    >>> store.advance_run(run.id, "running")
    >>> store.finish_run(run.id, exit_code=0, ckpt_state="complete", loadable=True)
    >>> store.get_run(run.id).fully_succeeded
    True

toy 边界：

    · 全库读进内存；生产是数据库 + 分页 + 归档。
    · 没有并发控制：乐观锁只在单进程里演示「版本不匹配就拒绝」。
    · JSON 文件可以被手工改坏；生产靠数据库约束和备份。
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .models import (
    ACTION_DEGRADE_RUN,
    ACTION_FAIL_RUN,
    ACTION_RESTART_WORKER,
    LAYERS,
    ROLE_ON_LOST,
    ROLE_TRAINER,
    ROLES,
    STATE_FAILED,
    STATE_PENDING,
    STATE_RUNNING,
    STATE_SCHEDULING,
    TEMPLATES,
    TERMINAL_STATES,
    Artifact,
    ConfigVersion,
    Conflict,
    Event,
    IllegalTransition,
    NotFound,
    Run,
    Task,
    Worker,
    compute_content_hash,
    merge_config,
    to_dict,
)

__all__ = ["Store", "ACTIVE_WORKER_STATES"]

ACTIVE_WORKER_STATES = frozenset({"pending", "starting", "running", "restarting"})

_SEQ_KEYS = ("task", "run", "artifact", "event")
_ID_PREFIX = {"task": "task", "run": "run", "artifact": "art", "event": "ev"}


def _utcnow() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _empty_db() -> Dict[str, Any]:
    return {
        "seq": {key: 0 for key in _SEQ_KEYS},
        "tasks": {},
        "config_versions": {},
        "runs": {},
        "workers": {},
        "artifacts": {},
        "events": [],
        "idem": {},
    }


class Store:
    """六张表的容器 + 事件流 + 写盘。

    clock 可注入，测试里用来拿确定的时间戳。
    """

    def __init__(self, path: str, clock: Optional[Callable[[], str]] = None) -> None:
        self.path = Path(path)
        self._clock = clock or _utcnow
        self._db = self._load()

    # ------------------------------------------------------------------
    # 磁盘
    # ------------------------------------------------------------------

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return _empty_db()

        with self.path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)

        db = _empty_db()
        for key, value in raw.items():
            if key == "seq" and isinstance(value, dict):
                db["seq"].update(value)
            elif key in db:
                db[key] = value
        return db

    def _persist(self) -> None:
        """temp → fsync → rename：要么旧文件，要么完整新文件。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(self._db, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)

    # ------------------------------------------------------------------
    # 内部小工具
    # ------------------------------------------------------------------

    def _next_id(self, kind: str) -> str:
        self._db["seq"][kind] += 1
        return f"{_ID_PREFIX[kind]}-{self._db['seq'][kind]:04d}"

    def _require(self, table: str, row_id: str) -> Dict[str, Any]:
        row = self._db[table].get(row_id)
        if row is None:
            raise NotFound(f"{table} 里没有 {row_id}")
        return row

    def _emit(
        self,
        entity_type: str,
        entity_id: str,
        type: str,
        payload: Optional[Dict[str, Any]] = None,
        source: str = "platform",
        corr_id: Optional[str] = None,
    ) -> Event:
        """只写内存；由公开方法统一 _persist()。"""
        event = Event(
            id=self._next_id("event"),
            entity_type=entity_type,
            entity_id=entity_id,
            type=type,
            ts=self._clock(),
            source=source,
            payload=payload or {},
            corr_id=corr_id,
        )
        self._db["events"].append(to_dict(event))
        return event

    # ------------------------------------------------------------------
    # Task
    # ------------------------------------------------------------------

    def create_task(
        self,
        name: str,
        defaults: Optional[Dict[str, Any]] = None,
        template: str = "llm-ft-v3",
    ) -> Task:
        if template not in TEMPLATES:
            raise ValueError(f"未知模板: {template}")

        now = self._clock()
        task = Task(
            id=self._next_id("task"),
            name=name,
            defaults=dict(defaults or {}),
            template=template,
            created_at=now,
            updated_at=now,
        )
        self._db["tasks"][task.id] = to_dict(task)
        self._emit("task", task.id, "task.created", {"name": name})
        self._persist()
        return task

    def update_task(
        self,
        task_id: str,
        changes: Dict[str, Any],
        expected_rev: Optional[int] = None,
    ) -> Task:
        """只改 Task 的默认值；已有 Run 的配置引用不动。"""
        row = self._require("tasks", task_id)
        if expected_rev is not None and row["rev"] != expected_rev:
            raise Conflict(
                f"task {task_id} 已被改过：现在是 rev={row['rev']}，"
                f"你拿到的是 rev={expected_rev}"
            )

        defaults = dict(row["defaults"])
        defaults.update(changes)
        row["defaults"] = defaults
        row["rev"] += 1
        row["updated_at"] = self._clock()

        self._emit("task", task_id, "task.updated", {"changes": dict(changes)})
        self._persist()
        return Task(**row)

    def get_task(self, task_id: str) -> Task:
        return Task(**self._require("tasks", task_id))

    def list_tasks(self) -> List[Task]:
        return sorted(
            (Task(**row) for row in self._db["tasks"].values()),
            key=lambda task: task.id,
        )

    def find_task_by_name(self, name: str) -> Optional[Task]:
        for task in self.list_tasks():
            if task.name == name:
                return task
        return None

    # ------------------------------------------------------------------
    # ConfigVersion
    # ------------------------------------------------------------------

    def save_config(
        self,
        task_id: str,
        overrides: Optional[Dict[str, Any]] = None,
        injected: Optional[Dict[str, Any]] = None,
    ) -> ConfigVersion:
        """四层合并 → 冻结。同任务同内容复用同一版本。"""
        task = self.get_task(task_id)

        merged, sources = merge_config(
            TEMPLATES[task.template],
            task.defaults,
            overrides or {},
            injected or {},
        )
        content_hash = compute_content_hash(merged)

        for row in self._db["config_versions"].values():
            if row["task_id"] == task_id and row["content_hash"] == content_hash:
                return ConfigVersion(**row)

        digest = content_hash.split(":", 1)[1]
        task_short = task_id.rsplit("-", 1)[-1]
        cv = ConfigVersion(
            id=f"cv-{task_short}-{digest[:8]}",
            task_id=task_id,
            merged=merged,
            sources=sources,
            content_hash=content_hash,
            created_at=self._clock(),
        )
        self._db["config_versions"][cv.id] = to_dict(cv)
        self._emit(
            "config", cv.id, "config.created",
            {"task_id": task_id, "content_hash": content_hash},
        )
        self._persist()
        return cv

    def get_config(self, config_version_id: str) -> ConfigVersion:
        return ConfigVersion(**self._require("config_versions", config_version_id))

    def run_config(self, run_id: str) -> ConfigVersion:
        """Run 当初用的那份配置，不随 Task 编辑而变。"""
        run = self.get_run(run_id)
        return self.get_config(run.config_version_id)

    def list_configs(self, task_id: Optional[str] = None) -> List[ConfigVersion]:
        rows = self._db["config_versions"].values()
        return [
            ConfigVersion(**row)
            for row in rows
            if task_id is None or row["task_id"] == task_id
        ]

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def create_run(
        self,
        task_id: str,
        overrides: Optional[Dict[str, Any]] = None,
        injected: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
    ) -> Run:
        """建一次执行。idempotency_key 相同 → 返回同一个 Run。"""
        if idempotency_key is not None:
            existing = self._db["idem"].get(idempotency_key)
            if existing is not None:
                return self.get_run(existing)

        config = self.save_config(task_id, overrides, injected)
        run = Run(
            id=self._next_id("run"),
            task_id=task_id,
            config_version_id=config.id,
            created_at=self._clock(),
            idempotency_key=idempotency_key,
        )
        self._db["runs"][run.id] = to_dict(run)
        if idempotency_key is not None:
            self._db["idem"][idempotency_key] = run.id

        self._emit(
            "run", run.id, "run.created",
            {"config_version_id": config.id, "overrides": dict(overrides or {})},
        )
        self._persist()
        return run

    def get_run(self, run_id: str) -> Run:
        return Run(**self._require("runs", run_id))

    def list_runs(self, task_id: Optional[str] = None) -> List[Run]:
        rows = self._db["runs"].values()
        runs = [
            Run(**row) for row in rows if task_id is None or row["task_id"] == task_id
        ]
        return sorted(runs, key=lambda r: r.id)

    def advance_run(
        self, run_id: str, to: str, expected_rev: Optional[int] = None
    ) -> Run:
        row = self._require("runs", run_id)
        if expected_rev is not None and row["rev"] != expected_rev:
            raise Conflict(
                f"run {run_id} 已被改过：rev={row['rev']}，期望 {expected_rev}"
            )

        run = Run(**row)
        previous = run.state
        run.advance(to, now=self._clock())
        run.rev += 1

        self._db["runs"][run_id] = to_dict(run)
        self._emit(
            "run", run_id, f"run.{to}", {"from": previous, "to": to}
        )
        self._persist()
        return run

    def finish_run(
        self,
        run_id: str,
        exit_code: int,
        ckpt_state: str = "complete",
        loadable: bool = True,
    ) -> Run:
        """三个成功分别落库：exit_code / ckpt_state / loadable。"""
        row = self._require("runs", run_id)
        run = Run(**row)
        run.finish(exit_code, ckpt_state, loadable, now=self._clock())
        run.rev += 1

        self._db["runs"][run_id] = to_dict(run)
        # 事件类型统一为 run.<新状态>：查「失败过哪些」只需要一个名字
        self._emit(
            "run", run_id, f"run.{run.state}",
            {
                "exit_code": exit_code,
                "ckpt_state": ckpt_state,
                "loadable": loadable,
            },
        )
        self._persist()
        return run

    def retry_run(self, run_id: str) -> Run:
        """自动重试：同一个 Run，attempt+1；旧 attempt 的 worker 让位。"""
        row = self._require("runs", run_id)
        run = Run(**row)
        previous_attempt = run.attempt
        run.begin_retry(now=self._clock())
        run.rev += 1

        for worker_row in self._db["workers"].values():
            if worker_row["run_id"] != run_id:
                continue
            if worker_row["attempt"] != previous_attempt:
                continue
            if worker_row["state"] in ACTIVE_WORKER_STATES:
                worker_row["state"] = "superseded"

        self._db["runs"][run_id] = to_dict(run)
        self._emit(
            "run", run_id, "run.retried",
            {"attempt": run.attempt, "config_version_id": run.config_version_id},
        )
        self._persist()
        return run

    def resume_run(
        self,
        run_id: str,
        overrides: Optional[Dict[str, Any]] = None,
        injected: Optional[Dict[str, Any]] = None,
    ) -> Run:
        """人工续跑：新 Run，配置可以换（换了就是新 ConfigVersion）。"""
        parent = self.get_run(run_id)
        child = self.create_run(parent.task_id, overrides, injected)
        self._emit(
            "run", child.id, "run.resumed",
            {"parent_run_id": parent.id, "parent_attempt": parent.attempt},
        )
        self._persist()
        return child

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def add_worker(
        self,
        run_id: str,
        rank: int,
        role: str = ROLE_TRAINER,
        node: str = "",
    ) -> Worker:
        if role not in ROLES:
            raise ValueError(f"未知角色: {role}（可选 {ROLES}）")

        run = self.get_run(run_id)
        worker = Worker(
            id=f"w-{run.id}-a{run.attempt}-r{rank}",
            run_id=run.id,
            rank=rank,
            role=role,
            node=node,
            attempt=run.attempt,
        )
        self._db["workers"][worker.id] = to_dict(worker)
        self._emit(
            "worker", worker.id, "worker.created",
            {"run_id": run.id, "rank": rank, "role": role, "node": node},
        )
        self._persist()
        return worker

    def get_worker(self, worker_id: str) -> Worker:
        return Worker(**self._require("workers", worker_id))

    def list_workers(self, run_id: Optional[str] = None) -> List[Worker]:
        rows = self._db["workers"].values()
        workers = [
            Worker(**row)
            for row in rows
            if run_id is None or row["run_id"] == run_id
        ]
        return sorted(workers, key=lambda w: (w.attempt, w.rank))

    def set_worker_state(self, worker_id: str, state: str) -> Worker:
        row = self._require("workers", worker_id)
        if row["state"] == state:
            return Worker(**row)

        row["state"] = state
        row["rev"] += 1
        if state == "running":
            row["last_heartbeat"] = self._clock()

        self._emit("worker", worker_id, f"worker.{state}", {"run_id": row["run_id"]})
        self._persist()
        return Worker(**row)

    def heartbeat(self, worker_id: str) -> Worker:
        row = self._require("workers", worker_id)
        row["last_heartbeat"] = self._clock()
        self._persist()
        return Worker(**row)

    def current_workers(self, run_id: str) -> List[Worker]:
        """只看当前 attempt 的 worker；旧 attempt 的留在历史里。"""
        run = self.get_run(run_id)
        return [
            w for w in self.list_workers(run_id) if w.attempt == run.attempt
        ]

    def trainers_ready(self, run_id: str) -> bool:
        """「数出来」而不是「被告知」：所有 trainer 都 running 才算齐。"""
        trainers = [
            w for w in self.current_workers(run_id) if w.role == ROLE_TRAINER
        ]
        return bool(trainers) and all(w.state == "running" for w in trainers)

    def pending_trainers(self, run_id: str) -> List[Dict[str, Any]]:
        """差哪几个：给页面显示「7/8，缺 sh-a-gpu-031」。"""
        return [
            {"rank": w.rank, "node": w.node, "state": w.state}
            for w in self.current_workers(run_id)
            if w.role == ROLE_TRAINER and w.state != "running"
        ]

    def start_run_if_ready(self, run_id: str) -> bool:
        """scheduling + 所有 trainer 就位 → running。"""
        run = self.get_run(run_id)
        if run.state != STATE_SCHEDULING:
            return False
        if not self.trainers_ready(run_id):
            return False
        self.advance_run(run_id, STATE_RUNNING)
        return True

    def lose_worker(
        self, worker_id: str, corr_id: Optional[str] = None
    ) -> str:
        """进程丢失 → 按 role 决定 Run 的命运。返回执行的动作名。"""
        row = self._require("workers", worker_id)
        worker = Worker(**row)
        action = ROLE_ON_LOST[worker.role]
        corr = corr_id or f"c-{worker.id}"

        row["state"] = "lost"
        row["rev"] += 1
        self._emit(
            "worker", worker_id, "worker.lost",
            {"role": worker.role, "node": worker.node, "rank": worker.rank},
            corr_id=corr,
        )

        run = self.get_run(worker.run_id)

        if action == ACTION_FAIL_RUN:
            # gang 不完整：只有还在进行中的 run 需要转 failed
            if run.state in (STATE_SCHEDULING, STATE_RUNNING):
                previous = run.state
                run.advance(STATE_FAILED, now=self._clock())
                run.rev += 1
                self._db["runs"][run.id] = to_dict(run)
                self._emit(
                    "run", run.id, "run.failed",
                    {"reason": "gang_incomplete", "missing_rank": worker.rank,
                     "from": previous},
                    corr_id=corr,
                )

        elif action == ACTION_DEGRADE_RUN:
            # 控制面丢了，但训练进程还活着：只降级，不杀 rank
            if run.state not in TERMINAL_STATES:
                run.health = "degraded"
                run.rev += 1
                self._db["runs"][run.id] = to_dict(run)
                self._emit(
                    "run", run.id, "run.degraded",
                    {"reason": "control_plane_lost", "role": worker.role},
                    corr_id=corr,
                )

        elif action == ACTION_RESTART_WORKER:
            # sidecar 掉了不影响训练：只重启它自己
            row["state"] = "restarting"
            self._emit(
                "worker", worker_id, "worker.restart",
                {"role": worker.role},
                corr_id=corr,
            )

        self._persist()
        return action

    def ready_workers(self, run_id: str) -> int:
        return sum(1 for w in self.current_workers(run_id) if w.state == "running")

    # ------------------------------------------------------------------
    # Artifact
    # ------------------------------------------------------------------

    def add_artifact(
        self,
        run_id: str,
        kind: str,
        path: str,
        checksum: str,
        manifest: Optional[Dict[str, Any]] = None,
    ) -> Artifact:
        run = self.get_run(run_id)
        artifact = Artifact(
            id=self._next_id("artifact"),
            run_id=run.id,
            attempt=run.attempt,
            kind=kind,
            path=path,
            checksum=checksum,
            manifest=manifest or {},
            created_at=self._clock(),
        )
        self._db["artifacts"][artifact.id] = to_dict(artifact)
        self._emit(
            "artifact", artifact.id, "artifact.created",
            {"run_id": run.id, "kind": kind, "path": path, "checksum": checksum},
        )
        self._persist()
        return artifact

    def get_artifact(self, artifact_id: str) -> Artifact:
        return Artifact(**self._require("artifacts", artifact_id))

    def list_artifacts(
        self, run_id: Optional[str] = None, kind: Optional[str] = None
    ) -> List[Artifact]:
        rows = self._db["artifacts"].values()
        return [
            Artifact(**row)
            for row in rows
            if (run_id is None or row["run_id"] == run_id)
            and (kind is None or row["kind"] == kind)
        ]

    def latest_artifact(
        self, run_id: str, kind: str = "checkpoint"
    ) -> Optional[Artifact]:
        items = [
            a for a in self.list_artifacts(kind=kind)
            if a.run_id == run_id and a.attempt == self.get_run(run_id).attempt
        ]
        return items[-1] if items else None

    # ------------------------------------------------------------------
    # Event
    # ------------------------------------------------------------------

    def events(
        self,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        corr_id: Optional[str] = None,
    ) -> List[Event]:
        out = []
        for row in self._db["events"]:
            if entity_type is not None and row["entity_type"] != entity_type:
                continue
            if entity_id is not None and row["entity_id"] != entity_id:
                continue
            if corr_id is not None and row["corr_id"] != corr_id:
                continue
            out.append(Event(**row))
        return out

    def append_event(
        self,
        entity_type: str,
        entity_id: str,
        type: str,
        payload: Optional[Dict[str, Any]] = None,
        source: str = "platform",
        corr_id: Optional[str] = None,
    ) -> Event:
        """外部组件（kubelet、训练进程）追加一条事件。"""
        event = self._emit(
            entity_type, entity_id, type, payload, source=source, corr_id=corr_id
        )
        self._persist()
        return event
