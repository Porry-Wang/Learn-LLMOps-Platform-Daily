# S3 · 标准答案（Q1–Q10 + 五条命门 · 第一人称 STAR 版）

> 用法：先看 `question.md`，遮住这里，用「我遇到什么 → 我要解决什么 → 我做了什么 → 结果如何」讲一遍；
> 讲不下去的地方再回来看对应那一段。
>
> 视角说明：S 段里的数字全部来自仓库里两次**真实执行**的日志——`logs/cli_submit.log`（happy path）
> 和 `logs/cli_retry_events.log`（掉线 → failed → retry → attempt 2）。行号指向
> `platform/src/porryllmops/{models,store}.py` 与 `course/s03-objects/objects.py`。
> 不虚构任职经历：对外讲时把这套账说成「对照公开的 K8s / Volcano / Megatron 做法做的设计」，不说成某公司的线上系统。

## 第一组：对象与基数

### Q1｜六张表各存什么？为什么 Run 只存 `config_version_id`？

**S｜我遇到的情况**

最初设计时，我想挑最省事的那条路：把参数直接塞进 Run——`run.lr`、`run.batch_size`、`run.seq_len`……看着挺自然。但顺着这条思路推一步就出问题：任务默认值一改，所有指向同一个 dict 的历史 Run 会跟着一起变——**我在用「当前值」冒充「当时值」**。这个假设在单机脚本里没人管，在平台上就是审计事故。

**T｜我要解决的问题**

平台的六张表，每一行到底代表哪个「事实」？哪些事实必须冻结、哪些必须能改？

**A｜我做了什么**

1. **按「这是什么事实」切表**：
   - `tasks`：用户的**意图**（可编辑的模板 + 默认值 + `rev`）。
   - `config_versions`：一次**冻结的参数快照**（`merged` 最终值、`sources` 来源层、`content_hash` 去重键）。
   - `runs`：对某份配置的**一次尝试序列**（`config_version_id`、`state`、`attempt`、`health`、三个成功字段）。
   - `workers`：Run 里的**一个进程**（`rank`、`role`、`node`、`state`）。
   - `artifacts`：**一件产物**（`kind`、`path`、`checksum`、`attempt`）。
   - `events`：**流水**（`entity_type` + `entity_id` 指向任意一行，只追加）。
2. **把「值」从 Run 里全部拿掉**，只留 `config_version_id` 一根指针。
3. **验证**：`cli_submit.log` 里落盘的 run-0001 原始 JSON 只有 `config_version_id: "cv-0001-662f6354"`，没有任何 `lr` / `seq_len` 字段；参数值在 `config_versions` 里，一份。

**R｜结果**

改 Task 的默认值不会再影响任何历史 Run（`test_task_edit_does_not_change_existing_run` 钉住这条）。代价是「这次参数是什么」要多一跳：Run → cv → `merged`。这一跳正是「可追溯」的物理形态，不是额外成本。

**可能被追问**

- 那同配置跑 10 次会存 10 份快照吗？——不会，`save_config` 先按 `task_id + content_hash` 查重（`store.py:257`），命中就复用同一个 cv。
- Run 里为什么还留 `task_id`？——为了「按任务列所有 Run」这一条查询不用先 join cv；它只是便利字段，权威路径仍是 cv。

**核句**

> Run 存的是「我用的是哪一版配置」，不是「我用了什么配置」——前者一根指针，后者一份会腐烂的副本。

### Q2｜四条边都是 1:N，为什么没有 1:1？为什么 Task 里没有 `run_ids`？

**S｜我遇到的情况**

画对象图时我第一反应是「一个任务对应一次训练」，想画成 1:1。然后我数了一下 `cli_submit.log` 的最后一行：

```
表行数: {'artifacts': 1, ..., 'runs': 1, 'workers': 10}
```

一个 Run，十个 Worker。再看重试之后的落盘：`workers by (attempt,state): {(1,'superseded'): 9, (1,'lost'): 1}`——还是 10 行，一行没多。基数这件事不能凭直觉画。

**T｜我要解决的问题**

每条边的「多」是哪一侧存的？为什么是这一侧？

**A｜我做了什么**

1. **逐条数多的一侧**：
   - Task → ConfigVersion：改一次参数就是一份新快照。
   - ConfigVersion → Run：同一份配置可以重试、可以手工再跑一遍。
   - Run → Worker：一个 Run 至少 8 个 trainer（多机就是 8×机器数），再加 sidecar / coordinator；id 格式 `w-{run_id}-a{attempt}-r{rank}`（`store.py:444`）本身就说明一个 Run 下有多行。
   - Run → Artifact：一次训练会存多个 ckpt（step-500、step-1000…），外加导出模型和评估结果。
2. **确认没有 1:1**：连「Run → Worker」都不是 1:1，因为 gang 训练的最小单位就是一个进程组。
3. **把外键放在「多」的一侧**：Worker 存 `run_id`，Artifact 存 `run_id`，Run 存 `config_version_id`，ConfigVersion 存 `task_id`。
4. **反过来验证**：Task 表里没有 `run_ids`，`objects.py` 的自检断言 `"run_ids" not in {f.name for f in fields(Task)}`。

**R｜结果**

四条边全是 1:N，上游不认识下游。省掉的麻烦是三件：删一个 Run 不用回头改 Task；两个控制器不会同时维护同一个不断变长的数组；「这个任务跑过几次」变成一次带过滤的查询，而不是一个需要同步的计数字段。

**可能被追问**

- 反查会不会很慢？——真实系统给 `runs.config_version_id`、`workers.run_id` 建索引即可；本项目是 JSON 表，全表扫，是 toy 缺口。
- 那「一个任务最近一次 Run」怎么查？——按 `task_id` 过滤再按创建时间取最后一个，不靠父表存指针。

**核句**

> 一对多，永远在「多」的那一侧存外键；在上游存数组，等于给自己养一个要同步的缓存。

### Q3｜`frozen=True` 为什么还不够？三个字段各解决什么

**S｜我遇到的情况**

我给 `ConfigVersion` 加了 `@dataclass(frozen=True)`，以为改不动了。写自检时顺手试了一句 `cv.merged["lr"] = 9.9`——**没报错**。冻结的是「这个属性指向哪个 dict」，不是「dict 里面的键」。

**T｜我要解决的问题**

怎么让「一次冻结的快照」真的冻结？这三个字段为什么缺一不可？

**A｜我做了什么**

1. **补一层只读视图**：`__post_init__` 里 `object.__setattr__(self, "merged", MappingProxyType(dict(self.merged)))`，`sources` 同样处理（`objects.py:128–129`）。现在 `cv.merged = {...}` 抛 `AttributeError`，`cv.merged["lr"] = 9.9` 抛 `TypeError`，两条断言分别盯着。
2. **把三个字段的用途分开**：
   - `merged`：最终生效值 → 复现。想知道「当时 lr 是多少」，看这里。
   - `sources`：每个值来自哪一层 → 溯源。`cli_submit.log` 里那七行 `← template / ← run_override / ← platform_injected` 就是它的渲染结果。
   - `content_hash`：只算执行相关字段 → 去重。`NON_EXECUTION_KEYS`（`models.py:157`）把 `name` / `description` / `tags` 排除在外，改备注不该产生新版本。
3. **让 id 由哈希推出来**：`cv-{task_short}-{digest[:8]}`（`store.py:264`），所以 `cv-0001-662f6354` 里的 `662f6354` 就是内容哈希前八位——同一个配置第二次提交，拿到的 id 一模一样。

**R｜结果**

「配置冻结」从一句愿望变成一个能被断言的事实；`test_content_hash_dedupes_and_ignores_metadata` 和 `test_sources_records_which_layer_gave_the_value` 各钉住一半。顺带得到一个好处：cv 的 id 天然可比较，不用查库就知道两份配置是不是同一份。

**可能被追问**

- 哈希撞了怎么办？——toy 用 sha256 前 8 位，理论可撞；生产应存完整哈希并加唯一索引。
- `sources` 为什么不干脆存一份四层的原始 dict？——存原始层会把「没写」和「写了 None」混起来，`merged + sources` 是压缩后的等价信息。

**核句**

> `frozen=True` 冻结的是字段，不是字段指向的那个 dict；哈希去重省的是行数，`sources` 省的是「谁改的」这场架。

## 第二组：状态与重试

### Q4｜6 态 8 边，终态为什么没有出边？

**S｜我遇到的情况**

我一度想给 Run 加一个 `resumed` 状态，让 failed 的 Run「回到」running——听起来比多一个 attempt 字段简单。顺着这个方案往下推，立刻撞上两件怪事：一是同一个 Run 的 `ended_at` 不知道该留哪一次；二是「这个 Run 失败过几次」这个查询没法写——失败被下一次状态覆盖掉了。

**T｜我要解决的问题**

状态字段到底该表达什么？重试算不算状态迁移？

**A｜我做了什么**

1. **只让 state 表达进程生命周期**，六个值：`pending → scheduling → running → succeeded / failed / cancelled`。
2. **数边**：`pending` 出 2（`scheduling`、`cancelled`）+ `scheduling` 出 3（`running`、`failed`、`cancelled`）+ `running` 出 3（`succeeded`、`failed`、`cancelled`）= **8 条**（`models.py:101`，`objects.py:56`）。
3. **终态出边写成空集**，而不是「不推荐」：`advance()` 第一步就检查 `if self.state in TERMINAL_STATES: raise`。
4. **重试不走 `advance()`**，走 `begin_retry()`，并且它只接受 `state == failed`（`objects.py:192`）——**`succeeded` 连重试都不能**，因为「成功了再跑一次」是另一次实验，不是第 N 次尝试的延续，应该 `resume_run()` 新建 Run。

**R｜结果**

`test_terminal_state_cannot_revive` 和 `test_retry_keeps_run_resume_creates_new` 都过；`cli_retry_events.log` 里第 1 次尝试的失败事实完整留在 `run.failed` 那一行，state 被 attempt 2 复用也没有覆盖它。

**可能被追问**

- 那「这个 Run 失败过吗」怎么查？——查事件流里有没有 `run.failed`，或者看 `attempt > 1`。状态字段只存当前值，历史在流水里。
- `scheduling → failed` 这条边为什么存在？——「抢不到资源 / gang 一直凑不齐」是失败，不是继续等；没有这条边，任务会永远卡在 scheduling。

**核句**

> 终态出边为空，不是因为「不想改」，而是因为 state 只有一个格子：要让「失败过」这件事留下来，就不能让下一次尝试把它擦掉。

### Q5｜retry 和 resume 的区别

**S｜我遇到的情况**

掉线那条日志跑完，我盯着两个数字看了很久：

```
run-0001 → attempt 2  state=pending  config=cv-0001-9fbdfff5（配置不变）
workers by (attempt,state): {(1, 'superseded'): 9, (1, 'lost'): 1}
```

attempt 变了，配置没变，worker 一行没多。那「人工从 ckpt 接着训」该长什么样？如果也用 retry，就说不清「这次是不是换了参数」。

**T｜我要解决的问题**

两种「再跑一次」——故障重试 vs 人工续跑——在数据里怎么区分？

**A｜我做了什么**

1. **retry = 同一个 Run，`attempt+1`**（`store.py:388`）：`state` 回 `pending`，清空 `started_at` / `ended_at` / `exit_code` / `ckpt_state` / `loadable` / `health`，`rev + 1`；旧 attempt 里还活着的 worker 标 `superseded`；`config_version_id` **一个字不改**。
2. **resume = 新 Run**（`store.py:412`）：直接 `create_run()`，可带新的 overrides（换了参数就是新的 cv），并且发一条 `run.resumed`，payload 里带 `parent_run_id` / `parent_attempt`。
3. **用日志验证分界**：retry 之后 `config=cv-0001-9fbdfff5（配置不变）`；attempt 2 的 worker **一个都没有**——因为 `retry_run()` 只负责把 Run 打回 pending，重建进程是运行时（真实系统里是控制器 / kubelet）的活。

**R｜结果**

「重试 8 次」= 1 个 Run 8 个 attempt；「换了 8 组参数各跑一次」= 8 个 Run。两种运维事实在数据里长得不一样，统计重试率不用猜。

**可能被追问**

- retry 为什么不清空 `id` / `task_id`？——它们标识「这件事」，不标识「这一次尝试」；清掉的只有属于单次尝试的事实。
- 为什么旧 worker 是 `superseded` 而不是删掉？——删了就查不到「第 1 次尝试用过哪些机器」。`lost` 那个也没有被改写成 superseded：终态事实不被改写。

**核句**

> retry 是同一件事再试一次，resume 是从产物出发的新一次；前者改 attempt，后者换 Run。

### Q6｜角色决定命运：为什么 trainer 掉了是整个 Run 失败

**S｜我遇到的情况**

`cli_retry_events.log` 里那句：

```
run.failed  run-0001  [c-w-run-0001-a1-r7] {"from": "running", "missing_rank": 7, "reason": "gang_incomplete"}
```

七个 rank 还活着，为什么要判整个 Run 失败？如果掉的是日志 sidecar，还会这样吗？

**T｜我要解决的问题**

「一个进程挂了」到「这次训练算不算活着」，中间靠什么判断？

**A｜我做了什么**

1. **把 role 当成处置策略表**，不是备注（`models.py:124`，`objects.py:84`）：

   | 角色 | 挂了之后 | 理由 |
   | --- | --- | --- |
   | `trainer` | `fail_run` | 算梯度、做 all-reduce；少一个 gang 就不完整 |
   | `sidecar` | `restart_worker` | 日志/指标上报，不参与计算 |
   | `coordinator` | `degrade_run` | 控制面没了但训练还活着：`health=degraded`，不杀 rank |

2. **搞清 trainer 为什么不能「只重启那一个」**：8 个 rank 在同一个 NCCL 通信组里，rank 7 一死，其余 7 个卡在 all-reduce 上**挂住而不是报错**；新起的进程拿不到旧通信组，必须全部重新 rendezvous。所以恢复的最小单位是整个 gang，不是单个进程。
3. **让失败带证据**：`run.failed` 的 payload 里写 `reason=gang_incomplete` 和 `missing_rank=7`，而不是只写一句 `failed`。

**R｜结果**

`test_worker_role_decides_whether_run_fails` 覆盖三种角色；CLI 的 status 能直接打出 `缺 trainer rank 7@sh-a-gpu-037(lost)`，不用翻日志猜是哪个 rank。

**可能被追问**

- coordinator 掉了为什么不失败？——它只管 rendezvous 和传 ckpt，训练进程之间的 all-reduce 不经过它；但「以后再也拿不到新 ckpt 元数据」，所以要降级并告警。
- `health=degraded` 算不算第四种成功状态？——不算，它是 Run 上的一个附加列，和 state 正交。

**核句**

> 进程挂了怎么办，不取决于「挂了几个」，取决于「这个进程在训练里干什么」。

## 第三组：事件与可追溯

### Q7｜28 行里只有 5 行属于这个 Run，Event 表该怎么设计？

**S｜我遇到的情况**

第一次查 `events run-0001`，只出来 5 行；不加过滤再查，是 28 行。差的 23 行属于 task、config、10 个 worker。我原本以为「查一个 Run 的历史」就是查一张 run 事件表。

**T｜我要解决的问题**

一张表怎么同时支持「按对象查」「按一次事故查」「看全局」？

**A｜我做了什么**

1. **不用六条外键，用 `(entity_type, entity_id)` 两列**指向任意一行（`models.py:408`）。`entity_id` 定位「哪一个」，`entity_type` 支持「按类型筛」。本项目 id 全局唯一且带前缀，所以 CLI 只传 `entity_id` 也能用；类型列是为「扫所有 run 事件」和未来 id 可能重复留的口子。
2. **只追加**：没有任何 `update` / `delete` 入口，`test_events_are_append_only` 钉住。
3. **三种查法对应三个真实问题**：

   | 你想问 | 查法 | 这次的数据 |
   | --- | --- | --- |
   | 这个 Run 经历了什么 | `entity_type=run, entity_id=run-0001` | 5 行 |
   | 这次事故影响了哪些对象 | `corr_id=c-w-run-0001-a1-r7` | 2 行，跨两张表 |
   | 全局最近发生了什么 | 不筛，按时间扫 | 28 行 |

**R｜结果**

同一张表支撑三种视图，不用为每个对象建一张历史表；新增对象类型（比如 Day 8 的 artifact 链）不需要改表结构，只要约定新的 `entity_type`。

**可能被追问**

- 没有外键，怎么保证 `entity_id` 有效？——toy 不校验，这是缺口；生产上要么加软外键校验任务，要么接受「指向已删除对象」并保留快照字段。
- 28 行会不会太大？——事件是追加型，靠时间分区和冷热分层，不靠删。

**核句**

> 一张流水表 + 两个字符串，就能记下所有对象的全部历史；代价是查询要带类型。

### Q8｜`corr_id` 是什么，为什么 retry 没有

**S｜我遇到的情况**

同一秒里出现了三行：

```
worker.lost  w-run-0001-a1-r7  [c-w-run-0001-a1-r7]
run.failed   run-0001          [c-w-run-0001-a1-r7]
run.retried  run-0001          （无 corr_id）
```

前两行共享一个 `corr_id`，第三行空着。我一开始以为是漏写了。

**T｜我要解决的问题**

怎么把「一次事故引发的多行」串起来，同时不把「后来人的动作」混进同一条因果链？

**A｜我做了什么**

1. **`corr_id = "c-" + worker.id`**（`store.py:530`）：一次 worker 掉线引发的全部事件共享这个值，跨 entity 也能一把捞出来。
2. **retry 刻意不带**：它是人（或自动策略）**新发起**的动作，不属于那次掉线的因果链。混进来会让「这次事故影响了几行」这个查询失真。
3. **反过来验证**：按 `corr_id` 查正好 2 行（`worker.lost` + `run.failed`），按 Run 查是 5 行——两个数不一样，说明两类查询确实在回答不同问题。

**R｜结果**

事故复盘可以做到「一条 `corr_id` 还原一个现场」；重试作为独立动作用 `run.id + ts` 在时间线上定位，不冒充事故的一部分。

**可能被追问**

- 那「用户点了 retry」这件事怎么串？——真实系统应给一次 API 请求发一个 `corr_id`（source=user），把这次请求引发的全部写入串起来；本项目只做了「事故链」这一半，是缺口，不是设计。
- `corr_id` 会不会重复？——按 worker.id 生成，一次掉线一个值；同一 worker 反复 lost 需要带序号，生产要加去重规则。

**核句**

> `corr_id` 串的是同一条因果链，不是同一个对象；retry 是新的因，不该挂在上一次的果上。

## 第四组：业务映射

### Q9｜把「配置变成一次可追溯的训练执行」翻译成一条链路

**S｜我遇到的情况**

九图第 1 节那句话很顺，但落到代码里，我不知道「可追溯」该怎么证明。直到我在 `cli_submit.log` 里看到一行具体的参数来源：

```
lr                 0.0001                 ← run_override
```

以及落盘的 run JSON 里那根指针 `config_version_id: "cv-0001-662f6354"`。

**T｜我要解决的问题**

从用户填表到磁盘上多出一行，中间每一步写哪张表、留什么痕？

**A｜我做了什么**

1. **把链路写全**：
   `Task(defaults + template)` → `merge_config` 四层（template → task → run_override → platform_injected）→ `ConfigVersion(merged + sources + content_hash)` → `Run(config_version_id, attempt)` → `Worker(rank, role)` + `Artifact(attempt, checksum)` → `Event(每次写入一行)`。
2. **定义「可追溯」为三跳**：`Artifact → Run → ConfigVersion → (merged, sources)`。
   实例：`art-0001 → run-0001 → cv-0001-662f6354 → lr=0.0001（来源 run_override）`。
3. **逐跳验证**：Artifact 自带 `attempt`（`models.py:391`），所以重试后不会和上一轮的 ckpt 混淆；Run 的 JSON 里没有参数值，只有 cv id；cv 的 id 前八位就是内容哈希。

**R｜结果**

给定任意一个 ckpt，四步之内能回答「它是什么参数、每个参数谁定的、是不是和上次同一份配置」。九图第 2 节留下的四个疑点也顺手有了答案：实例不变 attempt 变、自动恢复用原 cv、ckpt 能对回配置、热更新另说（Day 8）。

**可能被追问**

- 平台注入的字段（比如镜像 digest）为什么要记来源？——因为复盘时要区分「用户填错了」和「平台注入了旧镜像」，这两种事故的责任人不同。
- 模板升级了，历史 Run 会不会变？——不会，`TEMPLATES` 的改动只影响新的 `save_config`，历史 cv 已经把值烘进去了。

**核句**

> 可追溯不是「记了很多日志」，而是「任何一个产物，四步之内能走回它的参数和参数的来源」。

### Q10｜连点 10 次提交，会变成 10 个 Run 吗？

**S｜我遇到的情况**

CLI 演示时我在同一个终端里连按了两次回车，那一瞬间冒出来的问题是：这会不会建出两个 Run？于是我去翻代码找「到底哪一层挡重复」——结果发现挡重复的不是一处，是三层，而且有一层没接全。

**T｜我要解决的问题**

重复提交 / 重复落库 / 并发写，分别归谁管？

**A｜我做了什么**

1. **内容哈希去重**（`store.py:257`）：同 `task_id` + 同 `content_hash` 直接复用旧 cv。挡的是「同一份参数产生多个配置版本」。
2. **幂等键**（`store.py:307`）：`create_run(idempotency_key=...)` 命中就返回同一个 Run。挡的是「网络重试 / 用户连点」产生多个 Run。
3. **乐观锁 `rev`**（`store.py:205`、`345`）：`update_task` / `advance_run` 传 `expected_rev`，不匹配抛 `Conflict`。挡的是「两个控制器同时改同一条记录，后写的覆盖先写的」。
4. **把缺口写下来**：`retry_run` / `resume_run` / `finish_run` 目前**没接** `expected_rev`，所以「重试」这一步挡不住并发。这是 toy 缺口，不是设计意图。

**R｜结果**

`test_content_hash_dedupes_and_ignores_metadata`、`test_create_run_is_idempotent`、`test_optimistic_lock_rejects_stale_write` 三条测试各钉一层；面试时能明说「三层挡三种重复，第四处还没接」——比说「我们做了幂等」可信。

**可能被追问**

- 幂等键谁生成？——这次演示没传（日志里 `idempotency_key: null`；CLI 有 `--idem` 选项），真实系统由前端在打开表单时生成一个 request id。
- 三层能不能合成一层？——不能：哈希是内容维度的去重，幂等键是请求维度的去重，`rev` 是并发维度的冲突检测，三个维度。

**核句**

> 内容哈希挡「同样的配置」，幂等键挡「同一次点击」，`rev` 挡「两个人同时改」——三种重复，三种药。

## 第五组：五条命门（速答卡）

**1. 六张表各一行是什么？**

`tasks` 一份可编辑的意图 / `config_versions` 一次冻结的参数快照 / `runs` 一次尝试序列 / `workers` 一个进程 / `artifacts` 一件产物 / `events` 流水里的一行。

**2. 四条边都是 1:N，「单向引用」省掉什么？**

```
Task ─1:N→ ConfigVersion ─1:N→ Run ─1:N→ Worker
                              └──1:N→ Artifact
```

下游存上游 id，上游没有 `*_ids` 数组。省掉：删除时的连锁修改、父表上那个不断变长的数组、两个控制器同步同一对父子关系的竞态；代价是反查要带过滤。

**3. 6 态 8 边，终态出边是什么？**

`pending`(2) + `scheduling`(3) + `running`(3) = 8 条；`succeeded / failed / cancelled` 出边为**空集**。因为 state 只有一个格子，要留下「失败过」这个事实，就不能让下一次尝试把它擦掉——重试走 `begin_retry()` 改 attempt，不走状态迁移。

**4. 为什么 retry 的 `corr_id` 是空的？**

`corr_id` 串的是「一次事故」的因果链（`"c-" + worker.id`）。retry 是人新发起的动作，不是那次掉线的延续；挂上去会让「这次事故影响了几行」的查询失真。

**5. 三个成功是哪三列？为什么不能压成一个 status？**

`exit_code`（进程事实）、`ckpt_state`（文件事实：none/partial/complete）、`loadable`（框架事实）。`fully_succeeded` 要四件事同时成立。压成一个 status，等于让「退出码 0」替「ckpt 存全了」和「能加载」背书——三个来源不同的事实，压一起必然丢信息。
