"""S4 配置翻译链：优先级、四问、RuntimeSpec、掩码。

测的是「谁覆盖谁」和「翻译到哪一步」，不是 YAML 解析——
本 toy 不读 YAML，字段就是 merged dict 里的键。
"""

import pytest

from porryllmops.config import (
    FIELD_TRACE,
    LAYERS,
    TEMPLATES,
    ConfigError,
    build_runtime_spec,
    compute_content_hash,
    mask_config,
    merge_config,
    render_commands,
    trace,
)


def _merge(template=None, task=None, override=None, injected=None):
    return merge_config(
        template or {}, task or {}, override or {}, injected or {}
    )


# ---------------------------------------------------------------------------
# 1. 优先级：四层，上面赢，None 不算赢
# ---------------------------------------------------------------------------

def test_four_layers_precedence():
    merged, sources = _merge(
        template={"lr": 3e-4, "seq_len": 2048},
        task={"lr": 2e-4, "max_steps": 6000},
        override={"lr": 1e-4},
        injected={"lr": 5e-5, "image_digest": "sha256:9f2c"},
    )

    assert merged["lr"] == 5e-5                 # 四层都有 lr 时，最高层赢
    assert sources["lr"] == LAYERS[-1]
    assert sources["seq_len"] == "template"
    assert sources["max_steps"] == "task"
    assert sources["image_digest"] == "platform_injected"


def test_none_does_not_override_lower_layer():
    merged, sources = _merge(
        template={"lr": 3e-4},
        task={"lr": 2e-4},
        override={"lr": None},
    )

    assert merged["lr"] == 2e-4
    assert sources["lr"] == "task"


# ---------------------------------------------------------------------------
# 2. 内容哈希：改备注不产生新版本
# ---------------------------------------------------------------------------

def test_comment_does_not_change_content_hash():
    base = {"lr": 1e-4, "seq_len": 2048}
    with_comment = dict(base, comment="retry", description="顺手改的")

    assert compute_content_hash(base) == compute_content_hash(with_comment)
    assert compute_content_hash(base) != compute_content_hash(dict(base, lr=2e-4))


# ---------------------------------------------------------------------------
# 3. 四问：每个关键字段都要答得全
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", sorted(FIELD_TRACE))
def test_trace_answers_four_questions(field):
    answered = trace(field)
    for question in ("where", "stored", "injected", "read"):
        assert answered[question].strip()


def test_trace_carries_value_and_source():
    merged, sources = _merge(
        template=TEMPLATES["llm-ft-v3"], override={"lr": 1e-4}
    )
    answered = trace("ckpt_path", merged, sources)

    assert answered["source"] == "derived"       # 没填 → 平台推导
    assert answered["value"] is None

    with pytest.raises(ConfigError):
        trace("no_such_field")


# ---------------------------------------------------------------------------
# 4. RuntimeSpec：路径、镜像、并行度、挂载
# ---------------------------------------------------------------------------

def test_runtime_spec_derives_paths_from_task_and_run():
    merged, _ = _merge(template=TEMPLATES["llm-ft-v3"])
    spec = build_runtime_spec(
        merged, task_name="finetune-7b", run_id="run-0001", attempt=2
    )

    assert spec.env["PORRY_DATA_PATH"] == "s3://porry-data/finetune-7b"
    assert spec.env["PORRY_CKPT_PATH"].endswith(
        "s3://porry-ckpt/finetune-7b/run-0001/a2"
    )
    assert spec.env["PORRY_TF_EVENTS_PATH"].endswith("/a2/tfevents")
    assert ("/data", spec.env["PORRY_DATA_PATH"]) in spec.mounts
    assert spec.world_size == 8


def test_user_paths_flow_into_env_and_can_be_overridden():
    merged, sources = _merge(
        template=TEMPLATES["llm-ft-v3"],
        override={"data_path": "oss://team-a/corpus-v3"},
        injected={"ckpt_path": "s3://porry-ckpt/team-a"},
    )
    spec = build_runtime_spec(merged, task_name="t", run_id="run-0009")

    assert spec.env["PORRY_DATA_PATH"] == "oss://team-a/corpus-v3"
    assert spec.env["PORRY_CKPT_PATH"] == "s3://porry-ckpt/team-a"
    assert sources["data_path"] == "run_override"
    assert sources["ckpt_path"] == "platform_injected"


def test_user_mounts_are_appended_but_reserved_paths_rejected():
    merged, _ = _merge(
        override={"mounts": [["/scratch", "s3://scratch/team-a"]]}
    )
    spec = build_runtime_spec(merged, task_name="t", run_id="run-0001")
    assert ("/scratch", "s3://scratch/team-a") in spec.mounts
    assert ("/data", "s3://porry-data/t") in spec.mounts   # 平台挂载仍在

    with pytest.raises(ConfigError, match="保留挂载点"):
        build_runtime_spec(
            {"mounts": [["/data", "s3://tampered"]]},
            task_name="t", run_id="run-0001",
        )


def test_image_digest_is_pinned_and_reserved_env_is_rejected():
    merged, _ = _merge(
        template=TEMPLATES["llm-ft-v3"],
        override={"image": "registry.local/porry/trainer:24.10"},
        injected={"image_digest": "sha256:9f2c"},
    )
    spec = build_runtime_spec(merged, task_name="t", run_id="run-0001")
    assert spec.image == "registry.local/porry/trainer:24.10@sha256:9f2c"

    with pytest.raises(ConfigError, match="PORRY_"):
        build_runtime_spec(
            {"env": {"PORRY_DATA_PATH": "/tampered"}},
            task_name="t",
            run_id="run-0001",
        )


# ---------------------------------------------------------------------------
# 5. init 与 start 是两个独立进程：export 不会传过去
# ---------------------------------------------------------------------------

def test_init_export_does_not_reach_start():
    merged, _ = _merge(
        override={
            "init_cmd": "export FOO=1 && echo prepared",
            "start_cmd": "torchrun train.py",
        }
    )
    spec = build_runtime_spec(merged, task_name="t", run_id="run-0001")
    commands = dict(render_commands(spec))

    assert "export FOO=1" in commands["init"]
    assert commands["init"] != commands["start"]
    # 环境只来自平台注入；不会回读 init 那个 shell 的 export
    assert "FOO" not in spec.env


# ---------------------------------------------------------------------------
# 6. 私有变量：执行要真值，出口要掩码
# ---------------------------------------------------------------------------

def test_secrets_masked_at_exit_but_kept_for_execution():
    merged, _ = _merge(
        override={"hf_token": "hf_real_value", "env": {"WANDB_API_KEY": "wb-123"}}
    )

    assert mask_config(merged)["hf_token"] == "***"
    assert mask_config(merged)["env"]["WANDB_API_KEY"] == "***"
    assert "lr" not in mask_config(merged)  # 掩码是出口视图，不凭空造键

    spec = build_runtime_spec(merged, task_name="t", run_id="run-0001")
    assert spec.env["WANDB_API_KEY"] == "wb-123"          # 容器里要真值
    assert spec.public()["env"]["WANDB_API_KEY"] == "***"  # 页面/接口只给掩码
