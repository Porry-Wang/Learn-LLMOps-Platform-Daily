"""S3 验收：六张表的规矩，一条测试钉一条。

    · 终态不可复活；重试 = 同一个 Run，attempt+1
    · 改 Task 不改已有 Run 的配置
    · merged 存值、sources 存来源
    · content_hash 去重，且不受备注影响
    · Event 只追加，落盘后顺序不变
    · idempotency_key：同一个请求只建一个 Run
    · role 决定「worker 挂了 run 算不算失败」
    · Run 由「数 trainer」决定启动，不由上游宣布
    · 三个成功分开记
    · 人工 resume 是新 Run
    · 乐观锁拒绝过期写入
    · Artifact 是独立对象，可被后续 Run 指向
"""

import pytest
from dataclasses import FrozenInstanceError

from porryllmops.models import (
    ACTION_DEGRADE_RUN,
    ACTION_FAIL_RUN,
    ACTION_RESTART_WORKER,
    ROLE_COORDINATOR,
    ROLE_SIDECAR,
    STATE_FAILED,
    STATE_PENDING,
    STATE_RUNNING,
    STATE_SCHEDULING,
    STATE_SUCCEEDED,
    Conflict,
    IllegalTransition,
)
from porryllmops.store import Store


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------

def make_clock():
    tick = {"n": 0}

    def now() -> str:
        tick["n"] += 1
        return f"2026-10-27T10:{tick['n'] // 60:02d}:{tick['n'] % 60:02d}Z"

    return now


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "db.json", clock=make_clock())


def task_and_run(store, **overrides):
    task = store.create_task("finetune-7b", defaults={"max_steps": 8000})
    run = store.create_run(
        task.id,
        overrides=overrides or {"lr": 1e-4},
        injected={"image_digest": "sha256:9f2c"},
    )
    return task, run


def to_running(store, run):
    store.advance_run(run.id, STATE_SCHEDULING)
    return store.advance_run(run.id, STATE_RUNNING)


# ---------------------------------------------------------------------------
# 1. 状态机
# ---------------------------------------------------------------------------

def test_terminal_state_cannot_revive(store):
    _, run = task_and_run(store)

    with pytest.raises(IllegalTransition):
        store.advance_run(run.id, STATE_SUCCEEDED)      # pending 不能直接成功

    to_running(store, run)
    store.finish_run(run.id, exit_code=1)

    with pytest.raises(IllegalTransition):
        store.advance_run(run.id, STATE_RUNNING)        # 终态不可复活

    retried = store.retry_run(run.id)
    assert (retried.id, retried.state, retried.attempt) == (run.id, STATE_PENDING, 2)

    types = [e.type for e in store.events(entity_type="run", entity_id=run.id)]
    assert "run.failed" in types
    assert "run.retried" in types


def test_retry_keeps_run_resume_creates_new(store):
    task = store.create_task("finetune-7b")
    run = store.create_run(task.id, overrides={"lr": 1e-4})
    to_running(store, run)
    store.finish_run(run.id, exit_code=1, ckpt_state="none", loadable=False)

    retried = store.retry_run(run.id)
    assert (retried.id, retried.attempt) == (run.id, 2)
    assert retried.config_version_id == run.config_version_id

    store.advance_run(run.id, STATE_SCHEDULING)
    store.advance_run(run.id, STATE_RUNNING)
    store.finish_run(run.id, exit_code=1)

    child = store.resume_run(run.id, overrides={"lr": 2e-4})
    assert child.id != run.id
    assert child.attempt == 1
    assert child.config_version_id != run.config_version_id
    assert any(e.type == "run.resumed" for e in store.events(entity_type="run"))


# ---------------------------------------------------------------------------
# 2. 配置冻结
# ---------------------------------------------------------------------------

def test_task_edit_does_not_change_existing_run(store):
    task = store.create_task("finetune-7b", defaults={"max_steps": 8000})
    run = store.create_run(
        task.id,
        overrides={"lr": 1e-4},
        injected={"image_digest": "sha256:9f2c"},
    )
    before = dict(store.run_config(run.id).merged)

    store.update_task(task.id, {"max_steps": 6000, "lr": 1e-2})

    after = store.run_config(run.id)
    assert dict(after.merged) == before
    assert after.merged["max_steps"] == 8000
    assert after.merged["lr"] == 1e-4

    fresh = store.save_config(task.id)
    assert fresh.id != after.id
    assert fresh.merged["max_steps"] == 6000


def test_sources_records_which_layer_gave_the_value(store):
    task = store.create_task("finetune-7b", defaults={"max_steps": 8000})
    run = store.create_run(
        task.id,
        overrides={"lr": 1e-4},
        injected={"image_digest": "sha256:9f2c"},
    )

    sources = store.run_config(run.id).sources
    assert sources["lr"] == "run_override"
    assert sources["max_steps"] == "task"
    assert sources["batch_size"] == "template"
    assert sources["image_digest"] == "platform_injected"


def test_content_hash_dedupes_and_ignores_metadata(store):
    task = store.create_task("finetune-7b")

    cv1 = store.save_config(task.id, overrides={"lr": 1e-4})
    cv2 = store.save_config(task.id, overrides={"lr": 1e-4})
    assert cv1.id == cv2.id
    assert len(store.list_configs(task.id)) == 1

    cv3 = store.save_config(task.id, overrides={"lr": 2e-4})
    assert cv3.id != cv1.id

    # 备注不参与执行 → 不该产生新版本
    cv4 = store.save_config(task.id, overrides={"lr": 2e-4, "comment": "retry"})
    assert cv4.id == cv3.id


# ---------------------------------------------------------------------------
# 3. Event
# ---------------------------------------------------------------------------

def test_events_are_append_only(store):
    task = store.create_task("finetune-7b")
    run = store.create_run(task.id)
    first = store.events()
    assert first

    store.advance_run(run.id, STATE_SCHEDULING)
    second = store.events()
    assert len(second) > len(first)
    assert [e.id for e in second[: len(first)]] == [e.id for e in first]

    with pytest.raises(FrozenInstanceError):
        first[0].type = "rewritten"

    reloaded = Store(store.path)
    assert [e.id for e in reloaded.events()] == [e.id for e in second]


# ---------------------------------------------------------------------------
# 4. 幂等与乐观锁
# ---------------------------------------------------------------------------

def test_create_run_is_idempotent(store):
    task = store.create_task("finetune-7b")

    r1 = store.create_run(task.id, overrides={"lr": 1e-4}, idempotency_key="req-001")
    r2 = store.create_run(task.id, overrides={"lr": 1e-4}, idempotency_key="req-001")
    assert r1.id == r2.id
    assert len(store.list_runs()) == 1

    # 没有幂等键：点两次就是两次运行，但复用同一个配置版本
    r3 = store.create_run(task.id, overrides={"lr": 1e-4})
    r4 = store.create_run(task.id, overrides={"lr": 1e-4})
    assert r3.id != r4.id
    assert r3.config_version_id == r4.config_version_id


def test_optimistic_lock_rejects_stale_write(store):
    task = store.create_task("finetune-7b")
    updated = store.update_task(task.id, {"lr": 1e-4}, expected_rev=task.rev)
    assert updated.rev == task.rev + 1

    with pytest.raises(Conflict):
        store.update_task(task.id, {"lr": 9e-4}, expected_rev=task.rev)

    assert store.get_task(task.id).defaults["lr"] == 1e-4


# ---------------------------------------------------------------------------
# 5. Worker 与 role
# ---------------------------------------------------------------------------

def test_worker_role_decides_whether_run_fails(store):
    task = store.create_task("finetune-7b")
    run = store.create_run(task.id)

    for rank in range(8):
        store.add_worker(run.id, rank=rank, node=f"sh-a-gpu-0{30 + rank}")
    store.add_worker(run.id, rank=100, role=ROLE_SIDECAR, node="sh-a-gpu-030")
    store.add_worker(run.id, rank=101, role=ROLE_COORDINATOR, node="sh-a-gpu-030")

    store.advance_run(run.id, STATE_SCHEDULING)
    for worker in store.current_workers(run.id):
        if worker.role == "trainer":
            store.set_worker_state(worker.id, "running")
    assert store.start_run_if_ready(run.id) is True

    sidecar = [w for w in store.current_workers(run.id) if w.role == ROLE_SIDECAR][0]
    assert store.lose_worker(sidecar.id) == ACTION_RESTART_WORKER
    assert store.get_run(run.id).state == STATE_RUNNING
    assert store.get_worker(sidecar.id).state == "restarting"

    coordinator = [
        w for w in store.current_workers(run.id) if w.role == ROLE_COORDINATOR
    ][0]
    assert store.lose_worker(coordinator.id) == ACTION_DEGRADE_RUN
    degraded = store.get_run(run.id)
    assert degraded.state == STATE_RUNNING
    assert degraded.health == "degraded"

    trainer5 = [w for w in store.current_workers(run.id) if w.rank == 5][0]
    assert store.lose_worker(trainer5.id) == ACTION_FAIL_RUN
    assert store.get_run(run.id).state == STATE_FAILED


def test_run_starts_only_when_all_trainers_running(store):
    task = store.create_task("finetune-7b")
    run = store.create_run(task.id)
    for rank in range(8):
        store.add_worker(run.id, rank=rank, node=f"node-{rank}")
    store.advance_run(run.id, STATE_SCHEDULING)

    for worker in store.current_workers(run.id)[:7]:
        store.set_worker_state(worker.id, "running")

    assert store.trainers_ready(run.id) is False
    assert store.start_run_if_ready(run.id) is False
    assert store.pending_trainers(run.id) == [
        {"rank": 7, "node": "node-7", "state": "pending"}
    ]

    last = store.current_workers(run.id)[7]
    store.set_worker_state(last.id, "running")
    assert store.start_run_if_ready(run.id) is True
    assert store.get_run(run.id).state == STATE_RUNNING


# ---------------------------------------------------------------------------
# 6. 三个成功
# ---------------------------------------------------------------------------

def test_three_successes_are_independent(store):
    task = store.create_task("finetune-7b")

    # S2 的 C 组：退出 0、能加载，但只存了权重
    run = store.create_run(task.id)
    to_running(store, run)
    partial = store.finish_run(
        run.id, exit_code=0, ckpt_state="partial", loadable=True
    )
    assert partial.state == STATE_SUCCEEDED
    assert partial.fully_succeeded is False

    # 三个都成立
    full = store.create_run(task.id)
    to_running(store, full)
    complete = store.finish_run(
        full.id, exit_code=0, ckpt_state="complete", loadable=True
    )
    assert complete.fully_succeeded is True


# ---------------------------------------------------------------------------
# 7. Artifact
# ---------------------------------------------------------------------------

def test_artifact_is_a_standalone_object(store):
    task = store.create_task("finetune-7b")
    run = store.create_run(task.id)
    to_running(store, run)

    artifact = store.add_artifact(
        run.id,
        kind="checkpoint",
        path="s3://porry-ckpt/finetune-7b/run-01/a1/step-1500",
        checksum="sha256:4b8e",
        manifest={"step": 1500, "parallel": {"tp": 1, "pp": 1, "dp": 4}},
    )
    assert artifact.run_id == run.id
    assert artifact.manifest["step"] == 1500
    assert store.latest_artifact(run.id).id == artifact.id

    # 后续 Run 能指回同一件产物（lineage）
    child = store.resume_run(run.id, overrides={"lr": 2e-4})
    assert child.id != run.id
    assert store.get_artifact(artifact.id).path.endswith("step-1500")
