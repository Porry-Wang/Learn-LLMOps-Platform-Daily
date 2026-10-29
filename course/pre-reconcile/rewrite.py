#!/usr/bin/env python3
"""
前置 · 亲手写一个 Reconcile Loop（教学版完整实现）

世界观演示：
  - 声明式：desired 是一份「合同」，程序只关心目标
  - level-triggered：每轮重新观察现实、重新算差额，不记上次的动作
  - 幂等：act 重复执行同一个目标没有副作用（差额为 0 时什么都不做）
  - 无限逼近：单轮可能失败（随机 30%），但循环会把系统拉回期望状态

运行：
  python3 reconcile_loop.py
  python3 reconcile_loop.py --fail-rate 0.5 --disturbance 5
  python3 reconcile_loop.py --edge-triggered --drop-event 3 --fail-rate 0.5
"""
import argparse
import random
import time
from uuid import uuid4

DESIRED_REPLICAS = 3
MAX_ROUNDS = 30


# ---------- 数据面：一份「现实」 ----------
# actual 只在这一个 dict 里，observe 读它、act 写它、扰动也写它。
# Python 传 dict 是传引用，所以所有函数面对的是同一个世界。

def observe(actual: dict) -> dict:
    """返回当前现实的一份快照（拷贝）。

    为什么拷贝：快照代表「这一轮观察时刻的世界」。
    如果直接返回 actual 本身，后面 act 修改 actual 时，
    你手里的「快照」会跟着变——观察和动作之间世界就失真了。
    """
    return dict(actual)


# ---------- 控制面：算差额 ----------

def diff(desired_replicas: int, observed: dict) -> list:
    """level-triggered 的精髓：只看「现在差多少」，不看「上次做了什么」。

    返回动作列表：[("create", "pod-xxx"), ("delete", "pod-yyy"), ...]
    差额为 0 → 返回空列表 → 什么都不做（幂等）。
    """
    running = {name for name, state in observed.items() if state == "Running"}
    actions = []
    if len(running) < desired_replicas:
        # 缺几个补几个；pod 名字在「计划」阶段就生成好，
        # 执行失败时它根本不会进入现实
        for _ in range(desired_replicas - len(running)):
            actions.append(("create", f"pod-{uuid4().hex[:6]}"))
    elif len(running) > desired_replicas:
        # 多了砍掉（排序保证删除顺序确定，方便复现）
        for name in sorted(running)[desired_replicas:]:
            actions.append(("delete", name))
    return actions


# ---------- 执行：动作会真实改写「现实」 ----------

def act(actions: list, actual: dict, fail_rate: float) -> list:
    """逐条执行动作，每条有 fail_rate 概率失败。

    成功：create → 真的加进 actual；delete → 真的从 actual 删掉
    失败：actual 不变，等于这条动作从没发生过
    返回：[(action_type, target, ok), ...] 供上层打印统计
    """
    results = []
    for action_type, target in actions:
        ok = random.random() >= fail_rate
        if ok:
            if action_type == "create":
                actual[target] = "Running"
            elif action_type == "delete":
                actual.pop(target, None)
        results.append((action_type, target, ok))
    return results


# ---------- 主循环 ----------

def run_level_triggered(fail_rate: float, disturbance: int, verbose: bool = True):
    """控制论演示主体：观察 → 对比 → 行动，无限循环。"""
    desired = DESIRED_REPLICAS
    actual = {"pod-1": "Running"}  # 初始世界只有 1 个 pod，合同要求 3 个

    for round_num in range(1, MAX_ROUNDS + 1):
        # ⑤ 扰动放在 observe 之前：模拟「pod 在两次观察之间死了」，
        #   真实集群的故障从来不会等 controller 来看
        if disturbance and round_num % disturbance == 0:
            running = [k for k, v in actual.items() if v == "Running"]
            if running:
                victim = random.choice(running)
                del actual[victim]
                if verbose:
                    print(f"  Round {round_num:2d} | 💥 扰动：{victim} 挂了")

        observed = observe(actual)
        actions = diff(desired, observed)
        results = act(actions, actual, fail_rate)

        if verbose:
            created = sum(1 for t, _, ok in results if t == "create" and ok)
            failed = sum(1 for _, _, ok in results if not ok)
            n = len([v for v in actual.values() if v == "Running"])
            print(
                f"  Round {round_num:2d} | 期望 {desired} | 实际 {n} | "
                f"动作 {len(actions)} (成功 {created}, 失败 {failed})"
            )

        if not actions and round_num > 5:
            # 已经收敛：连续多轮差额为 0，什么都不做——这就是幂等
            pass

        time.sleep(0.05)  # 教学时是 sleep(1)，这里压缩时间

    final = len([v for v in actual.values() if v == "Running"])
    print(f"\n  ✅ 最终状态：{final}/{desired} 副本在线（fail_rate={fail_rate}, "
          f"disturbance={disturbance}）")
    return final


def run_edge_triggered(fail_rate: float, drop_event: int, verbose: bool = True):
    """反面教材：只在「收到事件」时行动，平时根本不看现实。

    在第 drop_event 轮故意丢掉事件 → 那一轮之后即使现实已经偏离，
    controller 永远不再观察、不再对比 → 系统永久停在错误状态。
    补救办法（真实系统）：定期全量 resync。
    """
    desired = DESIRED_REPLICAS
    actual = {"pod-1": "Running"}
    # edge-triggered 的 controller 不轮询现实，只维护「自己上次看到的世界」
    last_seen = dict(actual)

    for round_num in range(1, MAX_ROUNDS + 1):
        # 真实的事件源：只有现实相对「上次看到的」变了，才有事件送达
        changed = {k for k in set(actual) | set(last_seen)
                   if actual.get(k) != last_seen.get(k)}
        has_event = bool(changed)

        if round_num == drop_event:
            # 现实被改坏了（pod 死了），但这个事件在网络上丢了。
            # ★ 关键在于：controller 的 last_seen 并不会因此保持旧值——
            #   它靠事件流维护缓存，丢事件 = 缓存和现实悄悄错位 ★
            running = [k for k, v in actual.items() if v == "Running"]
            if running:
                victim = random.choice(running)
                del actual[victim]
                last_seen.pop(victim, None)  # 缓存跟着「假装无事发生」
            if verbose:
                print(f"  Round {round_num:2d} | ⚠️  现实已偏离，但事件被丢弃！")
            has_event = False

        if has_event:
            observed = observe(actual)
            actions = diff(desired, observed)
            act(actions, actual, fail_rate)
            last_seen = dict(actual)  # 收到事件 → 刷新缓存
        else:
            actions = []  # 没事件 → 不观察、不对比、不行动

        if verbose:
            n = len([v for v in actual.values() if v == "Running"])
            print(f"  Round {round_num:2d} | 期望 {desired} | 实际 {n} | "
                  f"动作 {len(actions)}{'  (静默)' if not has_event else ''}")

        time.sleep(0.05)

    final = len([v for v in actual.values() if v == "Running"])
    mark = "❌ 永久偏离" if final != desired else "侥幸收敛"
    print(f"\n  {mark}：edge-triggered 最终 {final}/{desired} 副本在线"
          f"（丢事件轮={drop_event}, fail_rate={fail_rate}）")
    return final


def main():
    parser = argparse.ArgumentParser(description="前置 · Reconcile Loop 教学实验")
    parser.add_argument("--fail-rate", type=float, default=0.3,
                        help="每个动作的失败概率，默认 0.3")
    parser.add_argument("--disturbance", type=int, default=0,
                        help="每 N 轮随机杀掉一个 pod；0=关闭。例：--disturbance 5")
    parser.add_argument("--edge-triggered", action="store_true",
                        help="切换为边缘触发模式（反面教材）")
    parser.add_argument("--drop-event", type=int, default=3,
                        help="边缘触发模式下，第几轮丢事件，默认 3")
    args = parser.parse_args()

    if args.edge_triggered:
        run_edge_triggered(args.fail_rate, args.drop_event)
    else:
        run_level_triggered(args.fail_rate, args.disturbance)


if __name__ == "__main__":
    main()
