# S4 · 配置翻译链（2026-10-29 · Day 2 · 九图第 3–4 节）

- 代码：`platform/src/porryllmops/config.py`（五组字段 / 四层合并 / trace / RuntimeSpec / 掩码）
- 演示：`translate.py`（跑一遍七段，末尾四条断言自检）
- 测试：`platform/src/porryllmops/tests/test_config.py`（16 条）；全仓 `34 passed in 0.09s`
- 证据：`logs/translate.log`（七段完整输出）、`logs/cli_submit_sources.log`（真实 CLI：来源 + 掩码）
- 问答：`question.md`（9 张问题卡）、`answer.md`（9 题 + 五条命门）

## 这一课的整体知识（人话版）

> 这一节回答「页面上这个字段，最后怎么变成容器里的东西」。先读这里，再看两份日志。

**一句话**：配置不是一堆键值，是**一条从「意图」到「容器」的翻译链**——用户填五组字段，平台按四层优先级合并并记来源，冻结成 `ConfigVersion`，再翻译成 `RuntimeSpec`（镜像 / 环境变量 / 挂载 / 命令 / 并行度），最后交给容器。每一步都要能回答：值从哪来、到哪去、冲突时谁赢。

### 1. 五组字段：页面为什么长

| 组 | 用户填什么 | 平台拿它干什么 | 本 toy 进容器吗 |
| --- | --- | --- | --- |
| 任务身份 | 阶段、名称、类型、描述、场景、标签 | 分类、校验、检索、通知策略 | 否 |
| 训练程序 | 框架、配置文件、Git、初始化/启动/退出命令 | 决定跑什么代码、怎么起 | 是（`init_cmd` / `start_cmd` / `framework`） |
| 训练输入 | 基座模型、tokenizer、数据集 / 数据地址 | 解析成路径并注入 | 是（`data_path`） |
| 执行环境与资源 | 镜像、依赖、环境变量、资源池、GPU 型号、机器数×卡数、挂载 | 生成容器规格 | 是（image / env / nodes / gpus_per_node / mounts） |
| 运行保障与观测 | 算力检测、容错、指标、轨迹、通知、推送 | 采集与告警策略 | 否（不改变这次容器形状） |

两个容易混的点：**「调试任务」是任务分类，「Debug 模式」会改变启动流程**，不是一回事；**身份组里的字段看着只是展示，但可能影响可选框架、校验规则和通知**——这是九图留下的待核实项，本 toy 只登记、不实现。

### 2. 四层优先级：冲突时谁赢

```
template < task < run_override < platform_injected
```

| 层 | 谁写 | 典型例子 | 能不能被覆盖 |
| --- | --- | --- | --- |
| `template` | 平台预置基线 | `lr=3e-4`、`seq_len=2048` | 能 |
| `task` | 任务默认值 | `max_steps=8000` | 能 |
| `run_override` | 本次提交改的 | `lr=1e-4`、`data_path=…` | 能 |
| `platform_injected` | 平台注入 | `image_digest=sha256:…` | 不能（最高） |

三条规则：**None 视为「这层没写」**，不覆盖上层；每个字段的赢家记进 `sources`，所以 CLI 能打出 `← run_override` 这种行；**内容哈希只算执行相关字段**，改备注不产生新版本（`NON_EXECUTION_KEYS`）。

为什么 `platform_injected` 最高：镜像是安全边界，用户改 `image` 可以，但不能绕开平台固定的 digest——「谁覆盖谁」必须有确定答案，不能靠字典插入顺序。

### 3. 四问：一个字段怎么走完全程

每个关键字段都答四个问题：**用户在哪里设置 → 平台怎样保存 → 运行时怎样注入 → 训练代码在哪里读取**。

| 字段 | 怎么存 | 怎么注入 | 哪里读 |
| --- | --- | --- | --- |
| 数据路径 | `merged['data_path']`，没填按 task 推导 | `PORRY_DATA_PATH` + 挂载 `/data` | 训练脚本自己读；平台只保证「给了」 |
| 产物路径 | `merged['ckpt_path']`，没填按 run/attempt 推导 | `PORRY_CKPT_PATH` + 挂载 `/ckpt` | 框架的 save_checkpoint；平台从这里采集 Artifact |
| 指标路径 | `merged['tf_events_path']` | `PORRY_TF_EVENTS_PATH`（默认落在 ckpt 同目录） | SummaryWriter；平台轮询该目录出曲线 |
| 镜像 | `image` + 单独存 `image_digest` | 容器 image 字段，写成 `repo@sha256:…` | 容器运行时，训练代码不感知 |
| 初始化命令 | `merged['init_cmd']` | 独立的一次 `sh -c` | 只在 init 进程里执行 |
| 启动命令 | `merged['start_cmd']` | 独立的一次 `sh -c` | torchrun / 训练入口 |

**「平台提供了路径」和「训练代码用了路径」是两个步骤**——平台能证明前半句（env 里有值），证不了后半句（代码有没有读）。这是九图第 3 节留的坑，答的时候要主动说。

### 4. RuntimeSpec：容器真正拿到什么

`build_runtime_spec()` 把 merged 翻译成一份冻结对象：镜像（含 digest）、`env`、`mounts`、`init_cmd` / `start_cmd`、`nodes × gpus_per_node`。日志里的实例：

```
image     registry.local/porry/trainer:24.10@sha256:9f2c
nodes×gpu 1 × 8 = world_size 8
mounts    /data ← oss://team-a/corpus-v3 ; /ckpt ← s3://porry-ckpt/…
```

两条边界：`RANK` / `LOCAL_RANK` **不在配置里**——那是 torchrun 在每个进程里设的，平台只给「整组共享」的 `PORRY_WORLD_SIZE`；`/data` 和 `/ckpt` 是平台保留挂载点，用户自定义挂载只能追加，不能顶掉。

### 5. init 与 start 是两个独立进程

这是今天最容易答错的一题：**`init_cmd` 里 `export` 的变量传不到 `start_cmd`**。因为它们是两次独立的 `sh -c`，各起一个进程；环境变量只从平台注入，不回读上一个 shell。日志里直接做了实验：init 里 `export HTTP_PROXY=…`，`spec.env` 里 `HTTP_PROXY` 为 `False`。

所以「初始化时设置的环境变量」只有两条正路：写进配置的 `env` 字段（进 merged、进哈希、能溯源），或者直接写进 `start_cmd`。**靠上一个脚本的 shell 状态**是最常见也最难查的坑——本地跑得好好的，平台上就是没有。

### 6. 私有变量：存储要真值，出口要掩码

- 存储必须是真值，否则容器启动连不上任何服务。
- 掩码发生在**出口**：页面 / API 响应、日志与事件 payload、复制任务。CLI 里 `hf_token *** ← run_override` 就是 `mask_config(cv.merged)` 的效果。
- 本 toy 做整值掩码；生产要按类型做 last4 / 前缀展示，并且**在 API 层统一出口**，而不是每个调用方各自记得掩码。
- 复制任务时私有变量应**清空并要求重填**，而不是复制一份——这是「复制」和「泄露」的分界线。

### 7. 保存 ≠ 保存并运行

`store.create_task()` 是保存意图；`store.create_run()` 才是运行。平台只要把这两个动作分开，就能表达三种结果：保存成功且运行成功 / 保存成功但运行失败（配置合法、资源不够）/ 保存失败（字段非法）。压成一个按钮动作，用户永远分不清「是我填错了」还是「平台没资源」。

### 8. 业务映射（九图第 3–4 节）

- **页面为什么长**：五组字段对应五种责任，不是把参数堆在一起；用户填的是意图，翻译是平台的责任。
- **配置优先级**：用户问「我改了参数为什么训练没变化」，答案要么是被更高层覆盖（看 `sources`），要么是代码根本没读这个变量（看第 3 节的两步区别）。
- **Debug 模式 vs 调试任务**：一个改启动流程，一个只是分类；混在一起会让「为什么这个任务行为不一样」无法回答。
- **私有变量**：平台的价值是「能看任务、看不到凭据」——这条做不到，多租户就不成立。

## 结论与下一步

**结论（面试可用的三句）**

1. 配置是一条链不是一堆键：五组字段 → 四层合并（记 `sources`）→ 冻结快照 → `RuntimeSpec` → 容器。
2. 冲突有确定答案：`template < task < run_override < platform_injected`，None 不算赢，镜像摘要平台说了算。
3. 三个最常被问的坑：init 的 `export` 不跨进程、平台给路径 ≠ 代码用了路径、私有变量只在出口掩码。

**已知缺口（主动交代）**

- 不解析 YAML / 不生成 PodSpec；`mounts` 的源是字符串，生产要解析成 volume。
- 未知字段不报错（拼错的键会安静地躺在 merged 里），生产要 schema 校验。
- 掩码只有整值与后缀启发式；没有 RBAC、没有按字段的展示策略。
- `env` 里的秘密只做出口掩码，不进事件 payload 的脱敏测试——Day 6 观测那一课再补。

**下一步（Day 3 · S5 + S6）**：配置翻译完，下一步是「先拿到资源」——队列 / 配额 / Gang，以及卡住时看哪里。产出 `scheduler.py` + 「三类卡住」一张表。
