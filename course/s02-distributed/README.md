# S2 · 分布式最小闭环（2026-10-27）

- 代码：`dist.py`（4 rank all-reduce）、`ckpt.py`（checkpoint 三段实验）、`debug_dist.py`（单进程可断点版）
- 环境修复：`torchrun_local.py`（macOS 上 `socket.getfqdn()` 导致 rendezvous 卡死的绕行）
- 日志：`logs/dist.log`、`logs/ckpt.log`、`logs/debug_dist.log`
- 完整问答：`question.md`（11 张问题卡）、`answer.md`（Q1–Q11 + 五条命门，STAR 版）
- 环境：macOS / Apple Silicon / 无 GPU / torch 2.8.0 / gloo（无 NCCL）

## 这一课的整体知识（人话版）

**一句话**：单卡装不下、跑不快，就把一次训练拆成多个进程；进程数由卡数定，每个进程干什么由切法定；进程之间能协同，靠的是 rank 编号和通信组。

### 1. 拆出来的每一份 = 一个进程 = 一个 rank

```
world_size = TP × PP × DP        # 本课口径，不含 EP/CP
```

三个维度切的对象不同，所以可以相乘：

| 维度 | 切什么 | 每份拿到什么 | 通信频率 | 能跨机吗 |
| --- | --- | --- | --- | --- |
| DP | 数据（batch） | 完整模型副本 + 自己那份数据 | 每步一次 | 能跨机 |
| PP | 层（depth） | 模型的一段 stage | 每个 micro-batch 收发激活 | 能，但气泡变大 |
| TP | 同层内部（width） | 这一层参数/激活的一个切片 | 每层多次 | 一般不能，必须待在 NVLink 域 |

“一个进程一张卡”是本课的简化。真实系统一个进程可以管多卡，也可能有辅助进程，那时进程数不等于卡数。

### 2. 三张网：同一批 rank 上的三种分组

一个“组”就是一串 rank 编号，不是新对象。分组规则只有一条：**三个坐标固定两个，让第三个遍历**。

```
rank = dp_idx × (PP×TP) + pp_idx × TP + tp_idx
```

16 卡、TP=2、PP=2、DP=4 时：

- TP 组（固定 dp/pp，变 tp）：`{0,1} {2,3} … {14,15}`，8 组 × 2 人，全在本机内；
- PP 组（固定 dp/tp，变 pp）：`{0,2} {1,3} … {13,15}`，8 组 × 2 人，也在本机内；
- DP 组（固定 tp/pp，变 dp）：`{0,4,8,12} … {3,7,11,15}`，4 组 × 4 人，必然跨机。

三组数字乘起来 `2×2×4 = 16`，等于 world_size：每个 rank 在每个维度上各属于一个组，三张网叠在一起才定位出它是谁。

组的成员身份是启动时用 `dist.new_group(ranks=[...])` 显式建出来的，所有 rank 必须以相同顺序调用相同次数，否则死锁。

**布局决定哪张网是“本机”的**：如果换一种展平顺序，DP 组会变连续，但 TP 组就跨机了。所以“切法”只决定分成几份，“放置”才决定通信代价。

### 3. backend 是路，collective 是动作

| 概念 | 是什么 | 例子 |
| --- | --- | --- |
| backend | 通信实现（走什么路） | `gloo`（CPU/以太网）、`nccl`（GPU/NVLink/IB） |
| collective | 通信语义（做什么动作） | all-reduce、all-gather、reduce-scatter、broadcast、send/recv |

常用组合：

- **DP 梯度同步**：all-reduce（SUM 后除以 world_size，即平均）；等价实现是 reduce-scatter + all-gather。
- **ZeRO-3 权重**：前向/反向之前 all-gather 分片，反向之后 reduce-scatter 梯度。
- **TP**：前向 all-reduce（合并部分和），反向再来一次。
- **PP**：`send/recv`，不是集合通信。

### 4. 启动顺序（七步）

| # | 步骤 | 要点 |
| --- | --- | --- |
| 1 | 资源与上限 | 机器数、单机卡数、NVLink 域、配额 |
| 2 | 选切法并校验 | `tp·pp·dp == 卡数`、`tp ≤ 单机卡数`、`tp \| heads`、`pp ≤ layers` |
| 3 | 定放置 | 坐标 → rank → 机器 / local_rank |
| 4 | 拉起进程 | launcher 注入 `RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR` |
| 5 | 建全局组 | `init_process_group`，rendezvous 全员到齐，缺一个就 hang |
| 6 | 建子组 | `new_group` 得到 TP / PP / DP 句柄 |
| 7 | 各自装载并开跑 | 代码/配置/数据/ckpt 到位，出现 first loss 才算真跑起来 |

第 1、4、7 步分别对应 S1 说的“资源就绪 / 程序启动 / 训练真正开始”三个时刻。

### 5. checkpoint：存的是“现场”，不是“模型”

| 字段 | 回答什么 | 丢了会怎样 |
| --- | --- | --- |
| `model` | 现在参数是多少 | 从随机初始化重来 |
| `optimizer` | 带着什么惯性和刹车（m / v / 每个参数的 step） | 第一步步长回到满格 lr，loss 出尖峰 |
| `step` | 在第几步 | lr 调度回到 warmup、日志横轴归零、排序错乱 |
| `config` | 这是什么形状的模型 | 造不出空模型，权重没地方放 |
| `rng`（可选） | 随机序列走到哪 | 续训不可复现 |
| `data_pos`（可选） | 数据读到哪 | 重复读或跳过一段 |

`state_dict` 只有“名字 + 张量”，没有结构；所以加载的顺序永远是：**config 造空壳 → load_state_dict 填数据**。config 只保证“装得进去”，代码 commit 才保证“装对了”（pre-LN / post-LN 参数名完全一样，能加载但语义不同）。

## 做了什么、看到了什么

### 实验 A：4 rank all-reduce

```text
[rank 0/4 local_rank=0] before=[1.0, 1.0, 1.0, 1.0] after=[10.0, 10.0, 10.0, 10.0]
[rank 1/4 local_rank=1] before=[2.0, 2.0, 2.0, 2.0] after=[10.0, 10.0, 10.0, 10.0]
[rank 2/4 local_rank=2] before=[3.0, 3.0, 3.0, 3.0] after=[10.0, 10.0, 10.0, 10.0]
[rank 3/4 local_rank=3] before=[4.0, 4.0, 4.0, 4.0] after=[10.0, 10.0, 10.0, 10.0]
```

- 每个 rank 都拿到**总和**，不是只给 rank 0。
- DDP 的实际做法是 SUM 再除以 world_size：这里就是 10/4 = 2.5，每个 rank 用同一份平均梯度更新，所以模型不会分叉。
- 这一组是 `TP=1, PP=1, DP=4` 的特例：4 个进程、只有全局组。

环境变量实测：`WORLD_SIZE=4`、`LOCAL_WORLD_SIZE=4`、`MASTER_ADDR=localhost`、`LOCAL_RANK` 与 `RANK` 一一对应（单机情形）。

**踩到的坑**：本机 `socket.getfqdn()` 返回 `::1` 的反向名 `1.0.0.0...ip6.arpa`，torchrun 的 rendezvous 反复重试，表现为“启动后一直不动”。这正好印证了第 5 步是集合操作——组没建成，后面的代码一行都不会执行。绕行方案是 `torchrun_local.py` 里把 `getfqdn` 固定成 `localhost`。

### 实验 B：存全套 vs 只存 model

同一份 60 步训练，第 30 步“故障”，续训 30 步：

| step | 对照 loss | A 组（存全套） | A 组 Δθ | C 组（只存 model） | C 组 Δθ |
| --- | --- | --- | --- | --- | --- |
| 31 | 0.388313 | 0.388313 | 0.293 | 0.388313 | **2.516** |
| 32 | 0.174320 | 0.174320 | 0.291 | **6.844686** | 1.795 |
| 33 | 0.358832 | 0.358832 | 0.307 | 1.253297 | 1.510 |
| 34 | 0.257866 | 0.257866 | 0.299 | 2.022305 | 1.197 |

- A 组与不中断的对照组**最大 loss 偏差 0.00e+00**，逐位一致。
- C 组第一步的更新量 ‖Δθ‖ 是 A 组的 **8.6 倍**（2.516 vs 0.293），下一步 loss 从 0.17 冲到 6.84。
- 原因：Adam 的步长 ≈ `lr · m / sqrt(v)`。m、v 清零后，偏差修正让第一步退化成 `lr · sign(g)`，刹车消失，所以一步迈过头。

结论：**“能加载”不等于“能接着训”**；判断标准是曲线接不接得上，不是文件能不能打开。

### 实验 C：config 决定形状

```
原 config hidden=64，改成 hidden=128 后加载：
  size mismatch for net.0.weight: copying a param with shape torch.Size([64, 16])
  from checkpoint, the shape in current model is torch.Size([128, 16]).
```

这就是“读 config 才能造空模型”的实证：`load_state_dict` 只按名字找张量、按形状复制，目标模型必须先存在。

ckpt 结构实测：6 个张量、5313 个参数、文件 70.5 KB；`optimizer.state` 的 key 是参数序号，每个里面是 `step`（标量）、`exp_avg`（与参数同形状）、`exp_avg_sq`（同形状）。

### 实验 D：单进程可断点版（`debug_dist.py`）

`dist.py` 必须由 torchrun 拉起 4 个子进程，PyCharm 的断点打在父进程上，子进程里的断点停不下来。所以补了一个单进程版本，三段从假到真：

- **段 1**：一个进程分饰四角，用 for 循环把 all-reduce 演一遍。断点打在 `total = total + before[rank]`，单步 3 次看清 `[1,1,1,1] → [3,3,3,3] → [6,6,6,6] → [10,10,10,10]`，以及 DDP 真正用的平均梯度 `10 / 4 = 2.5`。
- **段 2**：真调 `dist.init_process_group`，但 `world_size=1`。all_reduce 前后张量不变（`[3,3,3,3]` → `[3,3,3,3]`）。这个「空转」是反证——段 1 里「加上别人」那一步，在真实系统里确实由通信层完成，不是张量运算自带的性质。
- **段 3**：只打印真 4 进程的运行命令，以及 PyCharm 附加子进程的配置方法，不执行。

日志：`logs/debug_dist.log`。跑法 `../../.venv/bin/python debug_dist.py`，PyCharm 里右键 Debug 也能直接停。

边界：段 1 是模拟，不走 gloo，验证不了通信本身；段 2 走真 gloo，但 `world_size=1` 验证不了「多个进程互相等」，也就是验证不了 gang 语义。

## 闭卷问答

题目卡见 `question.md`（11 张，分五组：理论四问 / 代码追问三题 / 五条命门 / 动手实验两问 / 业务映射两问），标准答案见 `answer.md`（Q1–Q11 + 五条命门，第一人称 STAR 版）。

原先内联在本文件里的「闭卷三问」已扩进 `question.md` 的 Q1 / Q2 / Q4，答案里带了实测数字和追问，不再在此重复。

## 平台映射（九图）

- **第 6 节 启动执行**：launcher 负责注入 rank 环境变量、拉起 N 个进程；平台不替训练程序做 collective，也不替 Infra 决定 TP/PP/DP。平台能做的是校验 `tp·pp·dp == 卡数`、`tp ≤ 单机卡数`，并在提交时拒绝坏配置。
- **第 9 节 产物链路**：ckpt 只有带 `.done` 标记才算完整；目录里的 manifest 要记 step、config、并行布局和代码版本，否则换卡数续训时无法重新分片。三个成功（训练退出 0 / ckpt 完整 / 模型可导出）在本课的实验中已经能分开观察：A 组训练成功且 ckpt 完整，C 组两者都成立但**续训质量不合格**。
- **观测**：续训后曲线“断崖”多半是横轴从 0 重算，不是训练真的崩了；这是“图正常 ≠ 训练正常”的一个具体例子。

## 边界：本课验证了什么、没验证什么

| 项目 | 本课（toy） | 真实系统 | 保留了什么 | 丢失了什么 |
| --- | --- | --- | --- | --- |
| 通信 | gloo / CPU / 单机 4 进程 | NCCL / GPU / 多机 | rank、组、collective 语义 | 带宽、延迟、拓扑、超时行为 |
| 并行 | 只演示 DP 语义与 rank 布局 | TP/PP/EP 真实切分 | 坐标与分组规则 | 真实切分后的数值正确性、气泡 |
| ckpt | 单文件 `torch.save` | 分片 + 原子写 + DCP resharding | 该存哪几样、为什么 | 100 GB 级写入、跨并行度恢复 |
| 故障 | 人为在第 30 步切断 | 真实 crash / hang / 节点掉线 | “续训质量”的判据 | 检测延迟、MTTD/MTTR |

没做的：真实多机、NCCL、真实 TP/PP 切分、DCP、对象存储上的原子写。这些在后续 session 里补，不在本课假装做过。
