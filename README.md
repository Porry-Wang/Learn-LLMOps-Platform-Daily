# PorryLLMOps · 训推平台学习记录与 MVP

**本项目是什么**——Porry 对 LLMOps 训推平台的日常学习记录，外加一个跟着课程一起长出来的可运行 MVP。

**PorryLLMOps 平台是什么**——一句话：它不是发明新技术的产品，而是**把 Infra 团队反复要做的事抽成统一流程 + 可视化界面 + API**，让算法同学不用懂 K8s 也能自助跑分布式训练，让 Infra 同学不用天天救火、能专注做真正的工程优化。

它诞生的动机非常朴素：Infra 团队要维护的东西越来越多——调度、启动、容错、监控、产物管理、多租户隔离、热更新、权限、配置分层……每一件如果都靠人肉做，规模一大就崩。做一层抽象把这些动作固化成「页面填参数 → 点提交 → 看结果」，是必然的产品形态。

> 说明：平台名一律用代号 **PorryLLMOps**。问答里的第一人称是「项目作者」口吻，对标公开可查的行业做法，不虚构任职经历。

---

## 目录结构

仓库只有两个顶层目录，一条学习线、一条产品线：

```
course/       学习线：一天一个目录，当天的代码、日志、笔记、问答都在一起
platform/     产品线：PorryLLMOps MVP 的设计、源码与验收
```

### course/ —— 按天组织

每个 `dayNN/` 目录里是固定的四件套（有则放，没有不占位）：

| 文件 | 是什么 |
| --- | --- |
| `README.md` | 当天实验记录：做了什么 / 看到什么 / 为什么 / 结论 / 业务映射 |
| `question.md` | 闭卷问题卡（主动回忆用） |
| `answer.md` | 标准答案，第一人称 STAR 版 |
| `*.py` | 当天可运行的教学脚本 |
| `logs/` | 实验原始输出，`README.md` 的每条结论都要能回指到这里 |

当前内容：

| 目录 | 主题 | 状态 |
| --- | --- | --- |
| `course/day00/` | 环境冒烟：`smoke_gloo.py`（无 GPU，用 gloo 后端） | 完成 |
| `course/day01/` | Reconcile Loop：声明式、level vs edge、幂等收敛 | 完成（含 5 份对照实验日志） |
| `course/day02/` | 显存与算力三本账：TP/PP/DP/ZeRO/recompute | 完成 |
| `course/day03/` | 分布式最小闭环：4 rank all-reduce + checkpoint 三段实验 | 完成 |
| `course/day04/` | 控制面对象模型（对应平台 S3） | **代码已落地，学习材料待补** |
| `course/day08/` | kind 本地集群：`kind-config.yaml` | 配置就绪 |

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

课程按天推进，平台按 session 累积。对照关系：

| Session | 平台模块 | 对应课程 | 状态 |
| --- | --- | --- | --- |
| S1 容量与可行放置 | `capacity.py` | `course/day02/` | 完成 |
| S3 对象模型与存储 | `models.py` `store.py` `cli.py` | `course/day04/`（待补） | 代码完成，18 项测试通过 |
| S6 准入 / 排队 / Gang | `scheduler.py` | — | 未开始 |
| S7 rank 分配 / 拒绝原因 / top-3 | `placement.py` | — | 未开始 |

S2、S4、S5 尚未定义，等排到再补，不预先占目录。

四步法（每个 session 都走一遍）：**人设锚定 → STAR 问答 + 源码下钻 → 设计文档 + 可运行代码 → 极端测试验收**。前三步的产出分别落在 `course/dayNN/{question,answer}.md`、`platform/design/`、`platform/src/`，第四步落在 `platform/acceptance/`。

---

## 怎么跑

环境：macOS / Apple Silicon / Python 3.9 venv（仓库根 `.venv`）/ **无 GPU**，分布式一律用 gloo 后端。

```bash
# 全部测试（pytest.ini 已把 platform/src 加进 pythonpath）
.venv/bin/python -m pytest -q          # → 18 passed

# 当天的教学脚本，直接跑
.venv/bin/python course/day02/capacity.py --scan --params 70 --gpus 128 \
    --layers 80 --hidden 8192 --heads 64

# S3 控制面 CLI 演示
export PYTHONPATH=platform/src
python -m porryllmops.cli --db /tmp/demo.json submit --task finetune-7b \
    --set lr=1e-4 --inject image_digest=sha256:9f2c --simulate ok
python -m porryllmops.cli --db /tmp/demo.json status
python -m porryllmops.cli --db /tmp/demo.json events run-0001
```

`--simulate` 是演示驱动，按脚本改状态，不代表真实调度器；真实系统里这些状态由 kubelet / 训练进程 / 控制器上报。
