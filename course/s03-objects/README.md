# S3 · 对象模型与事件流（2026-10-29 · Day 1 · 九图第 1–2 节）

- 代码：`objects.py`（核心阅读版，330 行，末尾 `_selfcheck()` 六条断言）
- 平台侧：`platform/src/porryllmops/{models,store,cli}.py`
- 设计：`platform/design/object-model.md`（六张表 / 四条不变量 / 状态机 / 事件目录 / 读顺序）
- 测试：`platform/src/porryllmops/tests/test_store.py`（12 条）；全仓 `18 passed in 0.07s`
- 证据：`logs/cli_submit.log`（happy path）、`logs/cli_retry_events.log`（掉线 → failed → retry）
- 问答：`question.md`（10 张问题卡）、`answer.md`（10 题 + 五条命门）

## 这一课的整体知识（人话版）

> 先读这一节，再回看两份日志。日志里的每一行数字，下面都能找到它对应哪张表。

**一句话**：平台不是「记住一个任务在跑」，而是**把六种不同的事实分开记**——意图、参数快照、尝试、进程、产物、流水；再用四条不变量保证这六种事实不互相污染。九图第 1 节那句「配置如何变成一次可追溯的训练执行」，落到磁盘上就是这六张表。

### 1. 六张表 = 六种事实

| 表 | 一行 = | 能不能改 | 容易搞混的地方 |
| --- | --- | --- | --- |
| `tasks` | 一份可编辑的训练意图（模板 + 默认值） | 能改，`rev+1` | 改它**不影响**任何历史 Run |
| `config_versions` | 一次冻结的参数快照 | 不能改 | 存的是值，不是「用户填的表单」 |
| `runs` | 对某份配置的一次尝试序列 | 能改状态，终态不可复活 | 只存 `config_version_id`，不存参数值 |
| `workers` | Run 里的一个进程 | 能改（心跳/状态） | 重试后行数不变，旧行标 `superseded` |
| `artifacts` | 一件产物 | 不能改 | 自带 `attempt`，否则分不清哪轮的 ckpt |
| `events` | 流水里的一行 | 只追加 | 靠 `(entity_type, entity_id)` 指向任意一行 |

关键设计：**Run 只存指针，不存副本**。`cli_submit.log` 里落盘的 run-0001 JSON 有 `config_version_id: "cv-0001-662f6354"`，但没有任何 `lr` / `seq_len`——想知道当时用了什么，必须顺着指针去 cv。这样改任务默认值不会改写历史，同配置跑十次也只有一份快照（`store.py:257` 按 `task_id + content_hash` 去重）。

`ConfigVersion` 的三个字段各有分工：`merged` 复现、`sources` 溯源（就是日志里那行 `lr 0.0001 ← run_override`）、`content_hash` 去重（`cv-…-662f6354` 的后八位就是它）。光有 `frozen=True` 不够——它只挡住重新赋值，挡不住 `cv.merged["lr"] = 1`，所以 dict 还要包一层 `MappingProxyType`。

### 2. 四条边都是 1:N，上游不认识下游

```
Task ─1:N→ ConfigVersion ─1:N→ Run ─1:N→ Worker
                              └──1:N→ Artifact

Event ─(entity_type, entity_id)→ 指向上面任意一行
```

实测数字（来自两份日志）：

```
表行数: {'artifacts': 1, ..., 'runs': 1, 'workers': 10}            ← 1 个 Run 对 10 个进程
workers by (attempt,state): {(1,'superseded'): 9, (1,'lost'): 1}   ← 重试后还是 10 行
```

一个 Run 至少 8 个 trainer（多机就是 8×机器数），加 sidecar / coordinator；一次训练又会存多个 ckpt。所以没有一条边能画成 1:1。外键放在「多」的一侧——Worker 存 `run_id`，Artifact 存 `run_id`，Run 存 `config_version_id`；Task 里**没有** `run_ids`（`objects.py` 有一条断言专盯这个）。省掉的是「删一个 Run 要回头改 Task」和「父表上不断变长的数组」。

### 3. 状态机：6 态 8 边，终态没有出边

| 从 | 能到 |
| --- | --- |
| `pending` | `scheduling`、`cancelled` |
| `scheduling` | `running`、`failed`、`cancelled` |
| `running` | `succeeded`、`failed`、`cancelled` |
| `succeeded` / `failed` / `cancelled` | **（空）** |

2 + 3 + 3 = 8 条边。`scheduling → failed` 是必要的：「抢不到资源 / gang 凑不齐」是失败，不是继续等。终态出边为空，是因为 `state` 只有一个格子——要让「失败过」这个事实留下来，就不能让下一次尝试把它擦掉。所以：

- **retry**（`store.py:388`）：同一个 Run，`attempt+1`，回 `pending`，清空上一轮的时间/退出码/ckpt 状态，`config_version_id` 不改；旧 attempt 的 worker 标 `superseded`，掉线那个仍是 `lost`。
- **resume**（`store.py:412`）：新建 Run，可换配置，发 `run.resumed` 带 `parent_run_id`。
- **`succeeded` 不能 retry**：`begin_retry()` 要求 `state == failed`。「成功了再跑一次」是新的实验，走 resume。

### 4. 四条不变量

1. **配置冻结**：cv 不可变，Run 只存 id。违反 → 历史 Run 的「当时参数」被改写，复盘归零。
2. **单向引用**：下游存上游 id，上游没有 `*_ids`。违反 → 删除变连锁反应，父子关系变成竞态。
3. **终态不复活**：终态出边为空，再跑走 retry / resume。违反 → 「失败过没有」取决于你看的时间，告警无法去重。
4. **三个成功分开**：`exit_code`（进程）/ `ckpt_state`（文件）/ `loadable`（框架）各占一列。违反 → 退出码 0 但只存了权重被当成成功，续训从半截 ckpt 起步。

每条都有测试钉住：`test_task_edit_does_not_change_existing_run`、`test_artifact_is_a_standalone_object`、`test_terminal_state_cannot_revive` / `test_retry_keeps_run_resume_creates_new`、`test_three_successes_are_independent` / `test_bad_ckpt_is_not_success`。

### 5. 两次实跑看到了什么

**happy path**（`cli_submit.log`）：7 个字段、每个带来源；7/8 → 8/8 才启动（`start_run_if_ready` 只从 `scheduling` 触发，`store.py:513`）；落一句 `三个成功 exit=0 ckpt=complete loadable=True`，`rev=3`，run 可续跑。

**掉线路径**（`cli_retry_events.log`）：rank 7 lost → `run.failed {reason: gang_incomplete, missing_rank: 7}`；retry 后 `attempt=2 state=pending rev=4`，配置仍是 `cv-0001-9fbdfff5`；9 个还活着的 worker 变 `superseded`，掉线那个保持 `lost`。attempt 2 的 worker **一个都没有**——`retry_run()` 只把 Run 打回 pending，重建进程是运行时的活。

为什么 trainer 掉了是整个 Run 失败，而不是「重启 rank 7」：8 个 rank 在同一个 all-reduce 通信组里，少一个其余就**挂住**；新进程拿不到旧通信组，必须整组重新 rendezvous。处置按角色分流（`models.py:124`）：trainer → `fail_run`，sidecar → `restart_worker`，coordinator → `degrade_run`（`health=degraded`）。

### 6. 事件流：28 行里只有 5 行属于这个 Run

| 你想问 | 查法 | 这次的数据 |
| --- | --- | --- |
| 这个 Run 经历了什么 | `events run-0001` | 5 行 |
| 这次事故影响了哪些对象 | `events --corr c-w-run-0001-a1-r7` | 2 行（`worker.lost` + `run.failed`，跨两张表） |
| 全局最近发生了什么 | 不筛 | 28 行 |

`corr_id` 串的是**一条因果链**，不是同一个对象；它由 `"c-" + worker.id` 生成（`store.py:530`）。`run.retried` 刻意没有：retry 是新发起的动作，混进事故链会让「这次事故影响几行」失真。

另外三层防重复：内容哈希（同配置不产生新 cv）、幂等键（连点不产生多个 Run）、乐观锁 `rev`（并发写不互相覆盖）。**缺口**：`rev` 只接在 `update_task` / `advance_run` 上，`retry_run` / `resume_run` / `finish_run` 没接——讲的时候主动说，这是 toy 缺口。

### 7. 业务映射（九图第 1–2 节）

- **第 1 节「配置如何变成可追溯的执行」**：`Task → merge 四层 → ConfigVersion(merged/sources/hash) → Run(config_version_id/attempt) → Worker + Artifact → Event`。可追溯 = 三跳：`Artifact → Run → cv → (merged, sources)`。给一个 ckpt，四步内能答「什么参数、谁定的、是不是同一份配置」。
- **第 2 节四个待核实问题**：运行实例**不变**（自动重试改 attempt；人工续跑才换 Run）；热更新是否重启进程留给 Day 8 的模式表；自动容错**用原 cv**；ckpt 能对回配置（`Artifact.attempt` → `Run.config_version_id`）。
- **和后面几节的接口**：`Run.state` 只表达进程生命周期，训练好坏由三个成功字段说——这就是九图第 6 节「平台状态和程序状态要分开」在数据层的落点；Day 5 的九阶段生命周期只能做成派生 `phase` 视图，**不能**改这 6 个状态，否则 `TRANSITIONS` 和 `test_terminal_state_cannot_revive` 都要破。

## 结论与下一步

**结论（面试可用的三句）**

1. 平台记六种事实，不是一个 `status` 字符串；`Run` 只存 `config_version_id`，可追溯靠指针不靠副本。
2. 状态机 6 态 8 边、终态出边为空；重试改 attempt、续跑换 Run——两种「再跑一次」在数据里必须长得不一样。
3. 一次事故靠 `corr_id` 还原，一个对象靠 `(entity_type, entity_id)` 还原，全局靠时间扫；三种查询共一张 append-only 表。

**已知缺口（主动交代）**

- 持久化是 JSON 全量落盘，没有事务和唯一索引；`events` 没有外键校验。
- `rev` 没接全（见上）；演示没传幂等键（CLI 有 `--idem`，`cli_submit.log` 里是 `null`）。
- 事件的 `source` 只有 `platform` / `user`，没有真正的采集侧。

**下一步（Day 2 · S4）**：五组字段 → RuntimeSpec 的翻译、四层优先级（已有 `merge_config`）、私有变量掩码；把今天这根 `cv` 指针接到「配置到底怎么来」上。
