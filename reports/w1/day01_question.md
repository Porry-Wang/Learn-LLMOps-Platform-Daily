

# **第一组：理论四问（D1.1）**

Q1  命令式和声明式的区别？K8s 为什么选声明式？

Q2  act() 有 30% 失败率，为什么系统最终还能收敛到 3 副本？

Q3  spec 和 status 为什么要分开？合并成一个字段会出什么问题？

Q4  边缘触发丢一个事件后为什么无法自愈？真实系统怎么补救？



---



# **第二组：代码追问三题（读完/跑完代码后答 D1.2）**

Q5  diff() 里数 Running 时，如果把状态过滤去掉、直接数 dict 里所有 key，
     什么场景下会算错？

Q6  edge 实验里 pod 明明死了，为什么后面 27 轮都是「动作 0」？
     把因果链一步步讲出来（提示：changed 怎么算、last_seen 什么时候刷新）。

Q7  如果把 act() 改成「失败就自动重试直到成功」，外面那个循环还有什么
     存在意义？（考的是循环和重试的本质区别）



# **第三组：词汇检查**（D1.3）

Q8  集合推导式 {name for name, state in observed.items() if state=="Running"}
     的三部件各干什么：① name for name, state ② .items() ③ if 条件



# 第四组：动手实验的两个口头解释（D1.4-1.5）

Q9  （D1.4，先做实验再答）把 pass 改成 break、跑 --disturbance 5 后，为什么日志里根本看不到 💥 那行？「退出后的什么都不做」和「巡视后的什么都不做」差别到底在哪？

Q10 （D1.5）跑三组 edge 参数记录结果：

​     ① --edge-triggered --drop-event 3 --fail-rate 0.5

​     ② --edge-triggered --drop-event 10 --fail-rate 0.3

​     ③ --edge-triggered --drop-event 30 --fail-rate 0.3（几乎不丢，对照组）

​     然后答：「丢事件就下班」和「达标就 break」为什么是同一个错误的两种写法？



# **第五组：业务映射（D1.6，两三句话）**

Q11  Reconcile Loop 这套机制在PorryLLMOps平台里对应什么功能？给算法/Infra 同学解决什么痛点？如果这层抽错了会怎样？

（今天对话里的消防站、监控页一直在刷新，都是素材，用你自己的话组织）



