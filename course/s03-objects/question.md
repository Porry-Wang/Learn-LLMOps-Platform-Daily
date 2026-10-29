# S3 · 题目卡（闭卷作答，答完一批发一批）

> 作答前提：读完 `objects.py`、跑过 `logs/cli_submit.log` 和 `logs/cli_retry_events.log` 那四条命令、
> 默写过 `platform/design/object-model.md`。答不出来时不许翻 design 文档，先说出「我不知道的是哪一层」。

## 第一组：对象与基数

Q1  六张表各存什么？为什么 Run 只存 `config_version_id`，不把参数值复制一份进去？

Q2  对象图里四条边都是 1:N，为什么没有一条是 1:1？为什么 Task 里没有 `run_ids` 这一列？
    （提示：10 个 worker 是哪里来的；重试之后 worker 行数变没变）

Q3  `ConfigVersion` 已经 `frozen=True` 了，为什么还挡不住 `cv.merged["lr"] = 1`？
    `merged` / `sources` / `content_hash` 三个字段各解决什么问题？

## 第二组：状态与重试

Q4  Run 有哪 6 个状态、哪 8 条合法边？为什么终态的出边必须是空的？
    `succeeded` 能不能 retry？为什么？

Q5  `retry` 和 `resume` 的区别是什么？为什么 retry 不换 Run、而 resume 要新建 Run？
    retry 之后哪些字段变了、哪个字段一个字没变？

Q6  trainer / sidecar / coordinator 掉线，分别怎么处置？为什么 trainer 掉了是「整个 Run failed」，
    而不是「重启 rank 7 继续跑」？

## 第三组：事件与可追溯

Q7  一次 submit + 一次 retry 一共产生 28 行事件，其中只有 5 行的 `entity_id` 是 `run-0001`。
    这说明 Event 表必须怎么设计？三种查法分别回答什么问题？

Q8  `corr_id` 是什么？为什么 `worker.lost` 和 `run.failed` 共享一个 `corr_id`，
    而 `run.retried` 没有？

## 第四组：业务映射

Q9  九图第 1 节那句话是「配置如何变成一次可追溯的训练执行」。
    用六张表把它翻译成一条链路，并说清「可追溯」到底是哪三跳。

Q10 提交按钮被人连点 10 次，会变成 10 个 Run 吗？三个防重复机制分别挡住了哪一种重复？
    哪一个机制目前还没接全？

## 第五组：五条命门（闭卷速答，每条压到 2 行以内）

1. 六张表各一行是什么？
2. 四条边都是 1:N，画出来，并说明「单向引用」省掉了什么麻烦。
3. 6 态 8 边，终态出边是什么？为什么不能有出边？
4. 为什么 retry 的 `corr_id` 是空的？
5. 三个成功是哪三列？为什么不能压成一个 `status`？
