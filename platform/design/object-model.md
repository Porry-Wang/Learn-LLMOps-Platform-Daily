# 对象模型（S3 · Day 1 · 九图第 1–2 节）

> 这份文档回答两个问题：**系统里有哪些东西**（静态结构），以及**一次提交在系统里留下什么痕迹**（动态写入）。
> 前者对应九图第 2 节的对象模型，后者对应第 1 节那句「配置如何变成一次**可追溯**的训练执行」。
> 代码依据：`platform/src/porryllmops/models.py`、`store.py`；证据：`course/s03-objects/logs/cli_submit.log`、`cli_retry_events.log`。

---

## 0. 一句话与边界

**一句话**：Task 存意图，ConfigVersion 存冻结的参数，Run 存一次尝试，Worker 存进程，Artifact 存产物，Event 存流水——六张表，四条不变量，一个状态机。

**先划线**（面试时必须主动说）：

| 这里是真的 | 这里是 toy |
| --- | --- |
| 六张表的字段与关系、四条不变量、状态机的合法边、角色决定命运、三个成功分开 | JSON 文件持久化（生产是数据库 + 唯一索引） |
| 内容哈希去重、幂等键、乐观锁 `rev` | 单进程内存字典 + 每次全量落盘 |
| 事件 append-only + `corr_id` 因果链 | 事件的 `source` 只区分 `platform` / `user`，没有采集侧 |
| 「配齐才启动」（gang 的语义） | Worker 是 subprocess / 脚本改状态，没有 kubelet |

---

## 1. 六张表：一行代表什么

| 表 | 一行 = | 关键字段 | 谁写 | 能不能改 |
| --- | --- | --- | --- | --- |
| `tasks` | 一份可编辑的训练意图 | `name` `defaults` `template` `rev` | 用户 | **能改**（只改默认值，`rev+1`） |
| `config_versions` | 一次冻结的参数快照 | `merged` `sources` `content_hash` | 平台（合并四层） | **不能改**（`frozen=True` + `MappingProxyType`） |
| `runs` | 对某份配置的一次尝试序列 | `config_version_id` `state` `attempt` `health` `exit_code` `ckpt_state` `loadable` `rev` | 平台 | 能改状态，但**终态出边为空** |
| `workers` | Run 里的一个进程 | `run_id` `attempt` `rank` `role` `state` `last_heartbeat` | 运行时上报 | 能改（心跳/状态） |
| `artifacts` | 一件产物 | `run_id` `attempt` `kind` `path` `checksum` `manifest` | 产物采集 | **不能改**（`frozen=True`） |
| `events` | 流水里的一行 | `entity_type` `entity_id` `type` `ts` `payload` `corr_id` | 任何写入方 | **只追加**（`test_events_are_append_only`） |

三个容易混的点：

1. **`Run` 里没有 `task_id` 的实时值，也没有用户填的原始表单**。只有 `config_version_id`。想知道「当时用什么参数」，必须顺着这条指针去 `config_versions`——这是「配置冻结」能成立的前提。
2. **`Run` 不存 `template` 名**。模板升级（改 `TEMPLATES`）不会影响历史 `ConfigVersion`，因为历史版本已经把值烘进去了。
3. **`Artifact` 自带 `attempt`**（`models.py:391`），不是只有 `run_id`。否则 attempt 1 和 attempt 2 的 ckpt 分不清谁是谁。

---

## 2. 关系图（含基数）

```
Task ──1:N──→ ConfigVersion ──1:N──→ Run ──1:N──→ Worker
  │                                   │
  │                                   └──1:N──→ Artifact

Event ──(entity_type, entity_id)──→ 指向上面任意一行（多对一）
```

实测基数（来自 `cli_submit.log` / `cli_retry_events.log`）：

```
表行数: {'artifacts': 1, ..., 'runs': 1, 'workers': 10}       ← 1 个 Run 对 10 个 Worker
workers by (attempt,state): {(1,'superseded'): 9, (1,'lost'): 1}  ← 重试后仍是 10 行，只是状态变了
```

为什么是 1:N 而不是 1:1——三条硬理由：

- **Worker 1:N**：一个 Run 至少 8 个 rank（trainer），多机就是 8×机器数；再加 sidecar / coordinator。id 格式 `w-{run_id}-a{attempt}-r{rank}`（`store.py:444`）本身就说明一个 Run 下有多行。
- **Artifact 1:N**：一次训练会产出多个 ckpt（step-500、step-1000…），外加导出模型和评估结果，`kind` 字段就是为这个留的。
- **ConfigVersion 1:N Run**：同一份配置可以跑多次（重试、手工再跑一遍），配置不因此变多。

**单向引用**：下游存上游 id，上游不知道下游存在——`Task` 里没有 `run_ids` 字段（`objects.py` 里有一条断言专门检查这点）。好处是删一个 Run 不用回头改 Task；代价是「一个任务跑过几次」要反查。

---

## 3. 状态机：6 个状态，8 条边，终态没有出边

```
                 ┌─────────── cancel ───────────┐
                 │                              ▼
pending ──→ scheduling ──→ running ──→ succeeded / failed / cancelled
   │            │              │
   └─ cancel ───┴── fail ──────┘
```

| 从 | 能到 |
| --- | --- |
| `pending` | `scheduling`、`cancelled` |
| `scheduling` | `running`、`failed`、`cancelled` |
| `running` | `succeeded`、`failed`、`cancelled` |
| `succeeded` / `failed` / `cancelled` | **（空）** |

三个设计选择，都要能说出理由：

- **`running` 不能直接回 `pending`**。想再跑只有两条路：`begin_retry()`（同一个 Run，`attempt+1`）或新建 Run。这样「同一件事重试 8 次」和「换配置跑了 8 回」在数据里长得不一样——前者一个 Run 8 个 attempt，后者 8 个 Run。
- **`scheduling → failed` 这条边存在**，因为「一直抢不到资源 / gang 凑不齐」是失败，不是「还在等」。
- **`cancelled` 是终态**，取消之后不能复活；要重跑，同样走 retry（从头开始，不是接着跑）。

`Run.state` 只表达**进程生命周期**。训练好坏由另外三列说（见 §5）。这是九图第 6 节「平台状态和程序状态要分开」在数据层的落点。

---

## 4. 主链时序：每步哪张表动、发什么事件

### 4.1 happy path（真实输出见 `logs/cli_submit.log`）

| # | 谁触发 | 表变化 | 事件 | 用户看到 |
| --- | --- | --- | --- | --- |
| 1 | 用户 submit | `tasks` +1 行 | `task.created` | 任务创建成功 |
| 2 | 平台合并四层 | `config_versions` +1 行（同任务同哈希则复用） | `config.created` | 「7 个字段」，每行标来源 |
| 3 | 平台 | `runs` +1 行（`state=pending`, `attempt=1`） | `run.created` | 实例已建，未启动 |
| 4 | 调度/运行时 | `workers` +10 行（8 trainer + 1 sidecar + 1 coordinator） | `worker.created` ×10 | 「7/8 就位，缺 rank 7」 |
| 5 | 运行时 | sidecar / coordinator → `running` | `worker.running` ×2 | — |
| 6 | 平台 | `runs.state: pending → scheduling` | `run.scheduling` | 「等待资源」 |
| 7 | 运行时 | 8 个 trainer → `running` | `worker.running` ×8 | 「8/8 就位」 |
| 8 | 平台 `start_run_if_ready` | `runs.state: scheduling → running` | `run.running` | 「训练中」 |
| 9 | 产物采集 | `artifacts` +1 行（`kind=checkpoint`） | `artifact.created` | ckpt 路径 |
| 10 | 平台 `finish_run` | `exit_code` `ckpt_state` `loadable` 三列 + `state` | `run.succeeded`（带三个字段） | 「三个成功：exit=0 / complete / True」 |

两个细节值得单独记：

- **第 4→8 步是「数出来」不是「被告知」**：`trainers_ready()` 遍历当前 attempt 的所有 trainer，全部 `running` 才转；`pending_trainers()` 顺手给出「缺谁、在哪个节点」，这就是页面上那句 `7/8 就位 → 缺 rank 7@sh-a-gpu-037` 的来源。
- **第 8 步只从 `scheduling` 触发**（`store.py:513`）。已经是 `running` 再调一次什么都不做，所以重放不会把状态搞乱——这是 reconcile 循环幂等性在这里的具体形态。

### 4.2 掉线分支（真实输出见 `logs/cli_retry_events.log`）

| # | 表变化 | 事件 | 关键字段 |
| --- | --- | --- | --- |
| 1 | `workers[rank7].state = lost` | `worker.lost` | `corr_id = c-w-run-0001-a1-r7` |
| 2 | `runs.state: running → failed` | `run.failed` | `reason=gang_incomplete`、`missing_rank=7`、**同一个 `corr_id`** |
| 3 | 同一 Run：`attempt 1→2`、`state→pending`、清空 `started_at/ended_at/exit_code/ckpt_state/loadable`；**`config_version_id` 一个字没变** | `run.retried` | `attempt=2`、**没有 `corr_id`** |
| 4 | 旧 attempt 里还活着的 9 个 worker → `superseded` | （不单独发事件） | — |

**为什么第 3 行没有 `corr_id`**：`worker.lost` 和 `run.failed` 是同一次事故的两行，共享 `c-w-run-0001-a1-r7`（= `"c-" + worker.id`，`store.py:530`）；而 retry 是人（或自动策略）**新发起**的动作，不属于那次掉线的因果链。混进来会让「这次事故影响了几行」这个查询失真。

**为什么不是「重启 rank 7」而是「整个 Run failed」**：`ROLE_ON_LOST` 按角色分流（`models.py:124`）——

| 角色 | 挂了之后 | 理由 |
| --- | --- | --- |
| trainer | `fail_run` | gang 少一个就不完整，all-reduce 卡死 |
| sidecar | `restart_worker` | 日志/指标上报，不参与计算 |
| coordinator | `degrade_run` | 控制面没了但训练还活着：`health=degraded`，不杀 rank |

---

## 5. 四条不变量

每条都写清「定义 → 违反会怎样 → 谁钉住它」。

### 5.1 配置冻结

- **定义**：`ConfigVersion` 不可变（`frozen=True` + `merged`/`sources` 包进 `MappingProxyType`）；`Run` 只存 `config_version_id`。
- **违反会怎样**：改一次任务默认值，所有历史 Run 的「当时用了什么」全部被改写，复盘和审计归零；同一条 loss 曲线再也对不上参数。
- **谁钉住**：`test_task_edit_does_not_change_existing_run`、`test_content_hash_dedupes_and_ignores_metadata`；`objects.py` 自检里对 `setattr` 和 `merged.__setitem__` 各断言一次。
- **为什么**：可追溯的成本是「多存一份快照」，不可追溯的成本是「永远不知道为什么这次更好」。

### 5.2 单向引用

- **定义**：下游存上游 id，上游不认识下游。Task 里没有 `run_ids`，ConfigVersion 里没有 `run_ids`。
- **违反会怎样**：删除/迁移变成连锁反应；两个控制器同时改同一对父子关系，谁先谁后变成竞态。
- **谁钉住**：`objects.py` 的字段断言（`"run_ids" not in Task`）；`test_artifact_is_a_standalone_object`（产物独立存在，才能被后续 Run 引用）。
- **为什么**：一对多天然是「多」的那一侧存外键；反过来存就要维护一个不断变长的数组。

### 5.3 终态不复活

- **定义**：`succeeded / failed / cancelled` 的出边为空；要再跑只能 `begin_retry()`（同 Run，`attempt+1`）或 `resume_run()`（新 Run）。
- **违反会怎样**：「这个 Run 到底失败过没有」取决于你什么时候看它——历史不可信，重试次数无法统计，告警无法去重。
- **谁钉住**：`test_terminal_state_cannot_revive`、`test_retry_keeps_run_resume_creates_new`。
- **为什么**：状态字段只有一个值，所以要靠 attempt 和事件流来保留「第 1 次失败过」这个事实。

### 5.4 三个成功分开

- **定义**：`exit_code`（进程退出）、`ckpt_state`（none/partial/complete）、`loadable`（能不能加载）各占一列；`fully_succeeded` 要求四件事同时成立（`state==succeeded` 且 `exit_code==0` 且 `ckpt_state=="complete"` 且 `loadable is True`）。
- **违反会怎样**：退出码 0 但只存了权重（`objects.py` 自检里 `finish(0, partial, True)` 就是这种：`state=succeeded` 但 `fully_succeeded=False`）被判成「成功」，下一次续训从半截 ckpt 起步，loss 突然跳变，排查三天找不到原因。
- **谁钉住**：`test_three_successes_are_independent`、`test_bad_ckpt_is_not_success`；`objects.py` 里 `finish(0, partial, True)` 之后断言 `state==succeeded` 但 `not fully_succeeded`。
- **为什么**：退出码是**进程**的事实，ckpt 完整性是**文件**的事实，可加载是**框架**的事实。三个不同来源的事实压进一个枚举，必然丢信息。

---

## 6. 事件流：怎么把「当时发生了什么」查出来

### 6.1 事件目录（当前代码会发这些）

| 前缀 | 事件 | 何时 |
| --- | --- | --- |
| `task.` | `created` / `updated` | 建任务、改默认值 |
| `config.` | `created` | 新的内容哈希落库（复用旧版本时**不发**） |
| `run.` | `created` / `scheduling` / `running` / `succeeded` / `failed` / `cancelled` / `retried` / `resumed` / `degraded` | 状态迁移与重试、续跑、降级 |
| `worker.` | `created` / `pending` / `starting` / `running` / `restarting` / `lost` / `exited` | worker 生命周期；`lost` 带 `role/node/rank` |
| `artifact.` | `created` | 发现一件产物 |

两条约定：**事件类型跟着新状态命名**（`run.<新状态>`），所以「查失败过哪些」只要一个名字；**复用旧 ConfigVersion 时不发 `config.created`**，否则事件流里的「配置变了几次」会被去重逻辑污染。

### 6.2 三种查法，对应三个真实问题

| 你想问 | 查法 | 这次的数据 |
| --- | --- | --- |
| 这个 Run 经历了什么 | `events(entity_type="run", entity_id="run-0001")` | 5 行（28 行里只有 5 行属于这个 Run） |
| 这次事故影响了哪些对象 | `events(corr_id="c-w-run-0001-a1-r7")` | 2 行：`worker.lost` + `run.failed`，跨了两张表 |
| 全局最近发生了什么 | 不筛，按时间扫 | 28 行（含 task / config / 10 个 worker） |

两列的分工：`entity_id` 定位「哪一个对象」，`entity_type` 支持「按类型筛」（查所有 `run.` 事件、所有 `worker.` 事件）。本项目 id 全局唯一且带前缀（`run-0001` / `w-run-0001-a1-r7`），所以 CLI 的 `events run-0001` 目前只传 `entity_id` 也能工作；`entity_type` 是为「按类型扫」和未来表内 id 可能重复留的口子——这是缺口，不是多余设计。

### 6.3 另外三个「防重复」机制

| 机制 | 位置 | 解决什么 |
| --- | --- | --- |
| 内容哈希去重 | `save_config` 检查同 `task_id` + 同 `content_hash`（`store.py:257`） | 同样的参数跑 10 次只有 1 个 cv |
| 幂等键 | `create_run(idempotency_key=…)` 命中就返回同一个 Run（`store.py:307`） | 网络重试/用户狂点「提交」不产生 10 个 Run |
| 乐观锁 `rev` | `update_task` / `advance_run` 传 `expected_rev`，不匹配抛 `Conflict`（`store.py:205`、`345`） | 两个控制器同时改同一个 Run，后写的那个必须知道「我读到的是旧值」 |

`rev` 的边界要说清：只有 `update_task` 和 `advance_run` 支持 `expected_rev`；`retry_run` / `resume_run` / `finish_run` 目前没接，所以「重试」这一步还挡不住并发。这是 toy 缺口，不是设计意图。

---

## 7. 九图第 2 节留下的四个问题，现在的答案

| 原图里的待核实问题 | 本设计的选择 | 依据 |
| --- | --- | --- |
| 运行实例 I1 是始终不变，还是某些操作产生新实例？ | **实例不变，attempt 变**。自动重试不换 Run；人工续跑换 Run（`run.resumed` 记 `parent_run_id`） | `retry_run` vs `resume_run` |
| 热更新时训练进程是否重启？容器是否重建？ | 本月不实现。设计上：改配置 → 新 `ConfigVersion`；是否中断进程由模式决定（Day 8 三模式表） | Day 8 |
| 自动容错恢复时用哪份配置？ | **用原 `config_version_id`**（retry 不改它） | `run.retried` 的 payload 里 `config_version_id` 不变 |
| 最终产出的 ckpt 能否对应当时生效的版本？ | 能：`Artifact.run_id` + `Artifact.attempt` → 反查 `Run.config_version_id` → `cv` → `merged` + `sources` | `models.py:391` |

---

## 8. 读顺序（复习用）

1. 关系图（§2）→ 记住四个 1:N 和「单向引用」。
2. 状态机（§3）→ 记住 6 态 8 边、终态出边为空。
3. 主链时序（§4.1）→ 能把 10 步里的「表变化 / 事件」对上。
4. 掉线分支（§4.2）→ 能说清为什么 retry 的 `corr_id` 是空的。
5. 四条不变量（§5）→ 每条能配一个反例。
6. 事件查法（§6.2）→ 能说出「按 Run 查」和「按 corr_id 查」的区别。

自检一条命令：`.venv/bin/python course/s03-objects/objects.py`，六条断言全过即为「结构没记错」；但断言只证明模型自洽，不证明你能讲清——讲清靠 §8 的六步。
