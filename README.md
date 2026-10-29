# PorryLLMOps · 训推平台学习记录与 MVP

**本项目是什么**——Porry 对 LLMOps 训推平台的日常学习记录，外加一个跟着课程一起长出来的可运行 MVP。

**PorryLLMOps 平台是什么**——一句话：它不是发明新技术的产品，而是**把 Infra 团队反复要做的事抽成统一流程 + 可视化界面 + API**，让算法同学不用懂 K8s 也能自助跑分布式训练，让 Infra 同学不用天天救火、能专注做真正的工程优化。

它诞生的动机非常朴素：Infra 团队要维护的东西越来越多——调度、启动、容错、监控、产物管理、多租户隔离、热更新、权限、配置分层……每一件如果都靠人肉做，规模一大就崩。做一层抽象把这些动作固化成「页面填参数 → 点提交 → 看结果」，是必然的产品形态。

> 说明：平台名一律用代号 **PorryLLMOps**。问答里的第一人称是「项目作者」口吻，对标公开可查的行业做法，不虚构任职经历。

---

## 目录结构

仓库只有两个顶层目录，一条学习线、一条产品线：

```
course/       学习线：一个 session 一个目录，代码、日志、笔记、问答都在一起
platform/     产品线：PorryLLMOps MVP 的设计、源码与验收
```

### course/ —— 按 session 组织

目录名只有两种形态，看名字就知道它算不算 20 个 session 之一：

```
pre-environment/    pre- 前缀 = 前置，不带编号，不占 20 个 session 的额度
pre-reconcile/
s01-capacity/       s 前缀 + 两位数字 = session 验收单元，S1…S20，和计划文档的 S 号一一对应
s02-distributed/
```

前置和 session 的**判据只有一条**：2.0 §5.3 那张日历表里排了日期的是 `s`，没排日期的是前置。
S1 之所以是 `s` 而不是前置，是因为 2.0 给它派了「打包 + 验收」的活（`capacity.py` 进平台、写测试），
不只是读一遍笔记。用 Git 也能验证：`2fc31de`（10-20）是前置，`9fb538e`（10-23，2.0 生效日）才是 S1。

> 1.0 计划用的是「第 N 天」编号（D1 Reconcile、D2 容量、D3 分布式、D8 kind）。2.0 把单位换成了
> session，两套编号不同源（D2→S1、D3→S2、D8→S5），且学习进度已和日历脱钩，所以仓库只用 session 号。
> **D 号只活在 Obsidian 的资料库和 Git 历史里，不再进目录名。**

每个 session 目录里是固定的四件套（有则放，没有不占位）：

| 文件 | 是什么 |
| --- | --- |
| `README.md` | 本次实验记录：做了什么 / 看到什么 / 为什么 / 结论 / 业务映射 |
| `question.md` | 闭卷问题卡（主动回忆用） |
| `answer.md` | 标准答案，第一人称 STAR 版 |
| `*.py` | 本次可运行的教学脚本 |
| `logs/` | 实验原始输出，`README.md` 的每条结论都要能回指到这里 |

当前内容：

| 目录 | = 计划里的 | 主题 | 状态 |
| --- | --- | --- | --- |
| `course/pre-environment/` | 前置 | 环境冒烟：`smoke_gloo.py`（无 GPU，用 gloo 后端） | 完成 |
| `course/pre-reconcile/` | 前置 | Reconcile Loop：声明式、level vs edge、幂等收敛 | 完成（含 5 份对照实验日志） |
| `course/s01-capacity/` | S1 | 显存与算力三本账：TP/PP/DP/ZeRO/recompute | 完成 |
| `course/s02-distributed/` | S2 | 分布式最小闭环：4 rank all-reduce + checkpoint 三段实验 | 完成 |
| `course/s05-k8s-min/` | S5 | kind 本地集群：`kind-config.yaml` | 配置就绪（只有一个 yaml，尚未跑过） |

S3 之后的学习材料目录（`s03-objects/`、`s04-config/`）还没建，等排到再建，不预先占位；它们的状态记在下面的对照表里。

### platform/ —— 按模块组织

标准 Python 布局，包名 `porryllmops`：

```
platform/
├── design/                 设计文档（暂空，S1/S3 的设计说明目前写在模块 docstring 里）
├── src/porryllmops/
│   ├── capacity.py         S1 三本账 + feasible_placements()
│   ├── models.py           S3 六张表的形状与规矩（状态机、配置合并、content_hash）
│   ├── store.py            S3 持久化（JSON 落盘、事件流水、乐观锁、幂等）
│   ├── cli.py              S3 演示入口：submit / status / events / retry
│   └── tests/              test_capacity.py · test_store.py · test_cli.py
└── acceptance/             验收标准与极端测试用例（暂空）
```

---

## 学习线 ↔ 产品线怎么对上

两条线用同一个 session 号，所以对照关系不需要翻译：**`course/sNN-xxx/` 是学习记录，`platform/src/porryllmops/*.py` 是同一 session 的产品侧产出**。

| Session | 平台模块 | 学习材料 | 状态 |
| --- | --- | --- | --- |
| S1 容量与可行放置 | `capacity.py` | `course/s01-capacity/` | 完成 |
| S2 分布式最小闭环 | 无（教学实验，验证 rank / collective / ckpt 语义） | `course/s02-distributed/` | 完成 |
| S3 对象模型与存储 | `models.py` `store.py` `cli.py` | `course/s03-objects/`（待建） | 代码完成，18 项测试通过；`design/object-model.md` 未写 |
| S4 配置翻译链与优先级 | `config.py` | `course/s04-config/`（待建） | 半成品：`models.py::merge_config` 已做 4 层优先级 + `sources`，`test_store.py` 已断言，`cli.py submit` 会打印每个字段的「← 来源」；缺独立 `config.py`、五组字段 → RuntimeSpec 的翻译、私有变量掩码 |
| S5 K8s 最小面与三类卡住 | 无（读 + 记录，不落模块） | `course/s05-k8s-min/` | 未开始：只有 `kind-config.yaml`，`kind` / `kubectl` 本机尚未安装，缺 Pod 生命周期观察记录 |
| S6 准入 / 排队 / Gang | `scheduler.py` | `course/s06-scheduler/`（待建） | 未开始 |
| S7 rank 分配 / 拒绝原因 / top-3 | `placement.py` | `course/s07-placement/`（待建） | 未开始 |

前置 `pre-environment` / `pre-reconcile` 不在表里：它们是 2.0 §3.3 列的「保留（已完成）」，不占 20 个 session 的额度。

S2 和 S5 故意不落平台模块：前者的产出是「rank / collective / ckpt 的语义验证过了」，后者的产出是「看清了 K8s 的 Pod 生命周期和三类卡住长什么样」——都不是「平台多了一个功能」。

四步法（每个 session 都走一遍）：**人设锚定 → STAR 问答 + 源码下钻 → 设计文档 + 可运行代码 → 极端测试验收**。前三步的产出分别落在 `course/sNN-xxx/{question,answer}.md`、`platform/design/`、`platform/src/`，第四步落在 `platform/acceptance/`。

---

## 怎么跑

环境：macOS / Apple Silicon / Python 3.9 venv（仓库根 `.venv`）/ **无 GPU**，分布式一律用 gloo 后端。

```bash
# 全部测试（pytest.ini 已把 platform/src 加进 pythonpath）
.venv/bin/python -m pytest -q          # → 18 passed

# session 的教学脚本，直接跑
.venv/bin/python course/s01-capacity/capacity.py --scan --params 70 --gpus 128 \
    --layers 80 --hidden 8192 --heads 64

# S3 控制面 CLI 演示（PYTHONPATH 指向 platform/src；用仓库根的 venv 解释器，不依赖 activate）
export PYTHONPATH=platform/src

# happy path：submit → 7/8 就位 → 8/8 就位 → succeeded，三个成功分开记
.venv/bin/python -m porryllmops.cli --db /tmp/demo.json submit --task finetune-7b \
    --set lr=1e-4 --inject image_digest=sha256:9f2c --simulate ok
.venv/bin/python -m porryllmops.cli --db /tmp/demo.json status
.venv/bin/python -m porryllmops.cli --db /tmp/demo.json events run-0001

# trainer 掉线 → 重试（retry 只对 failed 状态的 run 有效，所以这里用 run-0002）
.venv/bin/python -m porryllmops.cli --db /tmp/demo.json submit --task broken-7b \
    --simulate trainer-lost --gpus 8
.venv/bin/python -m porryllmops.cli --db /tmp/demo.json retry run-0002
```

`cli.py` 也支持直接 `python platform/src/porryllmops/cli.py`（文件头有 4 行补包上下文的引导代码，PyCharm 里能右键 Debug）。

`--simulate` 是演示驱动，按脚本改状态，不代表真实调度器；真实系统里这些状态由 kubelet / 训练进程 / 控制器上报。
