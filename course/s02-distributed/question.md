# S2 · 题目卡（闭卷作答，答完一批发一批）

## 第一组：理论四问（S2.1）

Q1  gloo 和 nccl 的区别？为什么这台 Mac 只能用 gloo？换 backend 要改几行代码？

Q2  rank / world_size / local_rank 各回答什么问题？为什么工程上要读环境变量，
    不能自己算 `rank = machine_id × 8 + local_rank`？

Q3  all-reduce 和 reduce 的区别是什么？DP 为什么必须用 all-reduce，
    而不是「reduce 到 rank 0 再 broadcast」？

Q4  checkpoint 至少要存哪四样？每一样丢了分别会怎样？



---



## 第二组：代码追问三题（读完/跑完代码后答 S2.2）

Q5  `dist.py` 第 24 行把 `REDUCE_OP` 从 SUM 改成 MAX，4 个 rank 的 `after`
    分别是什么？为什么说 MAX「不再需要加法」？这对 DP 意味着什么？

Q6  `dist.py` 结尾的 `dist.barrier()` 去掉，小规模下输出看起来一样。
    那它到底在防什么？为什么这类 bug 最难查？

Q7  `ckpt.py` 的 C 组只存 model，第 31 步 ‖Δθ‖ 从 0.293 变成 2.516（8.6 倍）。
    把 Adam 的更新公式写出来，解释这个倍数是怎么来的。
    （追问：为什么是 `lr × √2536` 而不是 `lr × √5313`？）



---



## 第三组：五条命门（闭卷速答）

1. `world_size = TP × PP × DP`：三个维度各切什么对象？各自的通信频率和跨机能力？
2. `rank = dp_idx × (PP×TP) + pp_idx × TP + tp_idx`：建组规则用一句话说出来。
   16 卡 TP2/PP2/DP4 时，TP 组、PP 组、DP 组分别是哪些 rank？哪个必然跨机？
3. backend 和 collective 的区别，各举两个例子。DP 梯度 / ZeRO-3 权重 / TP / PP
   分别用哪个 collective？
4. 启动七步里，哪三步对应 S1 说的「资源就绪 / 程序启动 / 训练真正开始」？
   第 2 步的四条校验分别是什么？
5. `state_dict` 里**没有**什么？所以加载顺序永远是什么？



---



## 第四组：动手实验的两个口头解释（S2.3）

Q8  （跑完 `debug_dist.py` 段 2 再答）`world_size=1` 时 all_reduce 前后张量
    一模一样（`[3,3,3,3]` → `[3,3,3,3]`）。这个「什么都没发生」的实验
    证明了什么？它自己的失效边界在哪？

Q9  （`ckpt.py` 实验 3）hidden 从 64 改成 128 再加载同一个 state_dict，
    报 `size mismatch`。但还有一种情况**不会报错却更危险**——
    为什么说「config 只保证装得进去，代码 commit 才保证装对了」？举一个例子。



---



## 第五组：业务映射（S2.4，两三句话）

Q10 平台第 6 节「启动执行」和第 9 节「产物链路」分别对应今天的哪个实验？
    平台在这两件事上**故意不做什么**？这条边界划错会怎样？

Q11 S3 里 Run 的结束状态拆成三个独立字段（`exit_code` / `ckpt_state` / `loadable`）。
    用 C 组的结果说明：为什么三者互相不能推导，必须分开记？
    三个都成立就够了吗？
