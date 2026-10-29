"""S4 配置翻译链：五组字段 → 四层优先级 → RuntimeSpec → 容器能用的东西。

页面上的字段不是「容器参数」，是**意图**。中间必须有一条可复核的翻译链：

    表单五组字段
        │  用户在哪设、谁覆盖谁
        ▼
    merge_config（template < task < run_override < platform_injected）
        │  每个字段记来源（sources）
        ▼
    ConfigVersion（冻结快照，content_hash 去重）
        │
        ▼
    build_runtime_spec()  ← 本文件的新增部分
        │  镜像 / 环境变量 / 挂载 / 命令 / 并行度
        ▼
    容器

四个问题（每个关键字段都要答）：用户在哪里设置 → 平台怎样保存 → 运行时怎样注入 → 训练代码在哪里读取。
`trace(field)` 就是这四个问题的结构化答案；`build_runtime_spec()` 只实现第三问。

toy 边界（讲报告时主动说）：

    · 只实现「影响执行」的字段翻译；身份组、保障组只登记、不驱动容器。
    · 挂载是常量（/data、/ckpt），生产要从存储系统解析出真实 volume。
    · 秘密值只做整值掩码；生产要按类型做 last4 / 前缀展示，并在 API 层统一出口。
    · init 与 start 是两个独立 `sh -c`；容器编排细节（initContainer / 探针）不在本文件。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple

__all__ = [
    "GROUPS",
    "FIELD_GROUPS",
    "EXECUTED_FIELDS",
    "TEMPLATES",
    "DEFAULT_TEMPLATE",
    "LAYERS",
    "NON_EXECUTION_KEYS",
    "merge_config",
    "normalize_for_hash",
    "compute_content_hash",
    "FIELD_TRACE",
    "trace",
    "is_secret",
    "mask_value",
    "mask_config",
    "RuntimeSpec",
    "build_runtime_spec",
    "render_commands",
    "ConfigError",
    "SECRET_SUFFIXES",
]

# ---------------------------------------------------------------------------
# 1. 五组字段：页面很长，是因为它同时配置了五件不同的事
# ---------------------------------------------------------------------------

GROUPS: Tuple[Tuple[str, str], ...] = (
    ("identity", "任务身份"),
    ("program", "训练程序"),
    ("input", "训练输入"),
    ("env_resources", "执行环境与资源"),
    ("safeguards", "运行保障与观测"),
)

# 九图第 3 节的字段清单。这里登记它是为了「答得出页面为什么长」，
# 不代表每个字段都会进容器——见 EXECUTED_FIELDS。
FIELD_GROUPS: Dict[str, Tuple[str, ...]] = {
    "identity": ("stage", "name", "task_type", "description", "scenario", "tags"),
    "program": (
        "framework", "config_file", "git_repo", "git_branch", "git_commit",
        "submodules", "init_cmd", "start_cmd", "stop_cmd",
    ),
    "input": ("base_model", "tokenizer", "dataset", "data_path"),
    "env_resources": (
        "image", "image_digest", "deps", "env",
        "app_group", "pool", "gpu_type", "nodes", "gpus_per_node", "mounts",
    ),
    "safeguards": (
        "compute_check", "fault_tolerance",
        "metrics", "trace", "notify", "metrics_push",
    ),
}

# 真正会改变 RuntimeSpec 的字段（本 toy 只翻译这些）。
EXECUTED_FIELDS: frozenset = frozenset({
    "image", "image_digest", "env", "nodes", "gpus_per_node", "mounts",
    "data_path", "ckpt_path", "tf_events_path",
    "init_cmd", "start_cmd", "framework",
})

# ---------------------------------------------------------------------------
# 2. 四层优先级与内容哈希（S3 就在用，S4 把实现搬到这里独立成模块）
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
# 计划里写成 default < template < task < runtime override，本 toy 的
# 「runtime override」拆成两层：run_override（用户本次改）与
# platform_injected（平台注入，优先级最高，保证镜像摘要不被用户覆盖）。
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
    """稳定哈希：同一份执行配置永远得到同一个值。"""
    payload = json.dumps(
        normalize_for_hash(merged), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 3. 四问：一个字段怎么从页面走到训练代码
# ---------------------------------------------------------------------------

# field → 四个问题的答案（顺序固定：where / stored / injected / read）
FIELD_TRACE: Dict[str, Dict[str, str]] = {
    "data_path": {
        "where": "训练输入 · 训练数据集（或自定义数据地址）",
        "stored": "ConfigVersion.merged['data_path']；未填则平台按 task 推导",
        "injected": "环境变量 PORRY_DATA_PATH + 挂载 /data",
        "read": "训练脚本自己从 env 读；平台只保证「给了」，不保证「用了」",
    },
    "ckpt_path": {
        "where": "执行环境 · 产出存储目录；未填则由 run_id / attempt 推导",
        "stored": "ConfigVersion.merged['ckpt_path']",
        "injected": "环境变量 PORRY_CKPT_PATH + 挂载 /ckpt",
        "read": "训练框架的 save_checkpoint；平台之后从这里采集 Artifact",
    },
    "tf_events_path": {
        "where": "运行保障 · TensorBoard 产出路径",
        "stored": "ConfigVersion.merged['tf_events_path']",
        "injected": "环境变量 PORRY_TF_EVENTS_PATH（默认落在 ckpt 同目录）",
        "read": "SummaryWriter / 框架日志目录；平台轮询该目录出曲线",
    },
    "image": {
        "where": "执行环境 · 镜像",
        "stored": "merged['image'] + merged['image_digest']（摘要单独存）",
        "injected": "容器的 image 字段；有摘要时写成 repo@sha256:…",
        "read": "容器运行时（kubelet / containerd）；训练代码不感知",
    },
    "init_cmd": {
        "where": "训练程序 · 初始化命令",
        "stored": "merged['init_cmd']",
        "injected": "独立的一次 `sh -c`，不是 start 命令的前缀",
        "read": "只在 init 那个进程里执行；它 export 的变量不会进 start",
    },
    "start_cmd": {
        "where": "训练程序 · 启动命令",
        "stored": "merged['start_cmd']",
        "injected": "独立的一次 `sh -c`；环境只含平台注入的变量",
        "read": "torchrun / 训练入口",
    },
}


def trace(
    field: str,
    merged: Optional[Mapping[str, Any]] = None,
    sources: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """返回一个字段的四问；带 merged / sources 时附上当前值与来源。"""
    if field not in FIELD_TRACE:
        raise ConfigError(
            f"没有 {field!r} 的追踪记录；可选 {sorted(FIELD_TRACE)}"
        )
    out = {"field": field, **FIELD_TRACE[field]}
    if merged is not None:
        out["value"] = merged.get(field)
    if sources is not None:
        out["source"] = sources.get(field, "derived")
    return out


# ---------------------------------------------------------------------------
# 4. 私有变量：存储要真值，出口要掩码
# ---------------------------------------------------------------------------

SECRET_SUFFIXES: Tuple[str, ...] = (
    "_token", "_key", "_secret", "_password", "_passwd", "_credential",
)
SECRET_NAMES = frozenset({
    "hf_token", "wandb_api_key", "data_access_key", "docker_password",
})

MASK = "***"


def is_secret(name: str) -> bool:
    """按名字判断是不是私有变量。显式名单优先，后缀其次。"""
    lowered = name.lower()
    return lowered in SECRET_NAMES or lowered.endswith(SECRET_SUFFIXES)


def mask_value(value: Any) -> str:
    """整值掩码。生产应按类型做 last4 / 前缀展示，但绝不整段回显。"""
    return MASK if value not in (None, "") else value


def mask_config(values: Mapping[str, Any]) -> Dict[str, Any]:
    """出口用的只读视图：私有字段掩码，env 里按变量名逐个判断。"""
    out: Dict[str, Any] = {}
    for key, value in values.items():
        if key == "env" and isinstance(value, Mapping):
            out[key] = {
                name: (mask_value(val) if is_secret(name) else val)
                for name, val in value.items()
            }
        elif is_secret(key):
            out[key] = mask_value(value)
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# 5. RuntimeSpec：容器真正能用上的那一小份东西
# ---------------------------------------------------------------------------

DEFAULT_IMAGE = "registry.local/porry/trainer:24.10"
CONTAINER_DATA = "/data"
CONTAINER_CKPT = "/ckpt"
PLATFORM_ENV_PREFIX = "PORRY_"


class ConfigError(ValueError):
    """配置翻译不下去：缺字段、非法覆盖、未知追踪项。"""


@dataclass(frozen=True)
class RuntimeSpec:
    """一次运行要交给容器运行时的东西。字段少，但每个都能对上页面。"""

    task_name: str
    run_id: str
    attempt: int
    image: str
    env: Mapping[str, str]
    mounts: Tuple[Tuple[str, str], ...]       # (容器内路径, 源路径)
    init_cmd: str
    start_cmd: str
    nodes: int
    gpus_per_node: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "env", MappingProxyType(dict(self.env)))

    @property
    def world_size(self) -> int:
        return self.nodes * self.gpus_per_node

    def public(self) -> Dict[str, Any]:
        """接口 / 页面 / 日志用的视图：私有变量一律掩码。"""
        return {
            "task_name": self.task_name,
            "run_id": self.run_id,
            "attempt": self.attempt,
            "image": self.image,
            "env": dict(mask_config({"env": self.env})["env"]),
            "mounts": list(self.mounts),
            "init_cmd": self.init_cmd,
            "start_cmd": self.start_cmd,
            "nodes": self.nodes,
            "gpus_per_node": self.gpus_per_node,
            "world_size": self.world_size,
        }


def _as_env_str(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def build_runtime_spec(
    merged: Mapping[str, Any],
    *,
    task_name: str,
    run_id: str,
    attempt: int = 1,
    nodes: Optional[int] = None,
    gpus_per_node: Optional[int] = None,
) -> RuntimeSpec:
    """把 merged 配置翻译成容器能用的 RuntimeSpec。

    四条规则要能讲出来：

        1. 路径：用户填了就用用户的；没填由 task/run/attempt 推导（可复现）。
        2. 镜像：摘要单独存、拼在镜像名后面；有摘要就不用浮动 tag。
        3. 环境：平台变量独占 PORRY_ 前缀，用户 env 撞上就报错——
           「谁覆盖谁」必须有确定答案，不能靠字典插入顺序。
        4. 挂载：/data 与 /ckpt 是平台保留挂载点；用户自定义挂载只能追加。
    """
    data_path = str(
        merged.get("data_path") or f"s3://porry-data/{task_name}"
    )
    ckpt_path = str(
        merged.get("ckpt_path") or f"s3://porry-ckpt/{task_name}/{run_id}/a{attempt}"
    )
    tf_events_path = str(merged.get("tf_events_path") or f"{ckpt_path}/tfevents")

    nodes = int(nodes if nodes is not None else merged.get("nodes", 1))
    gpus_per_node = int(
        gpus_per_node if gpus_per_node is not None else merged.get("gpus_per_node", 8)
    )
    if nodes < 1 or gpus_per_node < 1:
        raise ConfigError(f"并行度非法：nodes={nodes}, gpus_per_node={gpus_per_node}")

    image = str(merged.get("image") or DEFAULT_IMAGE)
    digest = merged.get("image_digest")
    if digest and "@" not in image:
        image = f"{image}@{digest}"

    user_env: Dict[str, Any] = dict(merged.get("env") or {})
    reserved = sorted(k for k in user_env if k.startswith(PLATFORM_ENV_PREFIX))
    if reserved:
        raise ConfigError(
            f"{reserved} 用了平台保留前缀 {PLATFORM_ENV_PREFIX}；"
            "平台变量由合并链注入，用户不能覆盖"
        )

    env: Dict[str, str] = {
        "PORRY_TASK_NAME": task_name,
        "PORRY_RUN_ID": run_id,
        "PORRY_ATTEMPT": str(attempt),
        "PORRY_DATA_PATH": data_path,
        "PORRY_CKPT_PATH": ckpt_path,
        "PORRY_TF_EVENTS_PATH": tf_events_path,
        "PORRY_WORLD_SIZE": str(nodes * gpus_per_node),
    }
    for name, value in user_env.items():
        env[name] = _as_env_str(value)

    init_cmd = str(merged.get("init_cmd") or "")
    start_cmd = str(
        merged.get("start_cmd")
        or f"torchrun --nproc_per_node={gpus_per_node} train.py"
    )

    mounts: List[Tuple[str, str]] = [
        (CONTAINER_DATA, data_path),
        (CONTAINER_CKPT, ckpt_path),
    ]
    for item in merged.get("mounts") or ():
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ConfigError(f"mounts 每项必须是 (容器内路径, 源路径)，收到 {item!r}")
        inside, source = str(item[0]), str(item[1])
        if inside in (CONTAINER_DATA, CONTAINER_CKPT):
            raise ConfigError(
                f"{inside} 是平台保留挂载点（数据 / 产物），用户不能覆盖"
            )
        mounts.append((inside, source))

    return RuntimeSpec(
        task_name=task_name,
        run_id=run_id,
        attempt=attempt,
        image=image,
        env=env,
        mounts=tuple(mounts),
        init_cmd=init_cmd,
        start_cmd=start_cmd,
        nodes=nodes,
        gpus_per_node=gpus_per_node,
    )


def render_commands(spec: RuntimeSpec) -> List[Tuple[str, str]]:
    """init 与 start 各自是一次 `sh -c`。

    这是「init_cmd 里 export 的变量传不到 start_cmd」的实现层原因：
    两个独立进程，环境只从平台注入，不回读上一个进程的 shell。
    """
    out: List[Tuple[str, str]] = []
    if spec.init_cmd:
        out.append(("init", f"sh -c {spec.init_cmd!r}"))
    out.append(("start", f"sh -c {spec.start_cmd!r}"))
    return out
