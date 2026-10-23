"""S1 验收：机器数 × 卡数 → 可行放置。

三个案例对应 D2 的三个配置。每个测试保护的不变量：

    1. 7B/8    —— 单机 8 卡时，TP 上限就是 8；top-1 必须是小 TP + 大 DP。
    2. 70B/128 —— top-1 用满单机 NVLink 域；PP=1 时 TP<8 只能靠 full 重算勉强放下。
    3. 405B/16384 —— 卡多不等于免费；PP 气泡会改变排序，OOM 只是门槛。

断言同时检查：所有返回方案都不 OOM、拓扑自洽、TP 不出单机。
"""

from typing import List

import pytest

from porryllmops.capacity import Placement, feasible_placements


def assert_invariants(
    candidates: List[Placement],
    world_size: int,
    cards_per_machine: int,
    hbm_gb: float,
) -> None:
    assert candidates, "至少应有一个可行方案"
    for p in candidates:
        assert p.world_size == world_size
        assert p.tp * p.pp * p.dp == world_size
        assert p.tp <= cards_per_machine
        assert p.micro_batches >= 2 * p.pp
        assert p.total_gb <= hbm_gb
    # 排序键：扣掉气泡后的等效时长单调不减
    effective = [p.effective_hours for p in candidates]
    assert effective == sorted(effective)


def test_7b_8cards_single_machine():
    """7B on 1×8：放得下优先，8 卡全部做 DP，不切模型。"""
    candidates = feasible_placements(
        machines=1,
        cards_per_machine=8,
        params_b=7,
        seq_len=4096,
        layers=32,
        hidden=4096,
        heads=32,
    )
    assert_invariants(candidates, world_size=8, cards_per_machine=8, hbm_gb=80.0)

    top = candidates[0]
    assert (top.tp, top.pp, top.dp, top.zero, top.recompute) == (1, 1, 8, 1, "selective")
    assert top.total_gb == pytest.approx(56.75, rel=1e-3)
    assert top.bubble == 0.0
    assert top.effective_hours == pytest.approx(3686.4, rel=1e-3)


def test_70b_128cards_needs_tp():
    """70B on 16×8：top-1 用满 NVLink 域；TP<8 只能靠 PP 或 full 重算换显存。"""
    candidates = feasible_placements(
        machines=16,
        cards_per_machine=8,
        params_b=70,
        seq_len=4096,
        layers=80,
        hidden=8192,
        heads=64,
    )
    assert_invariants(candidates, world_size=128, cards_per_machine=8, hbm_gb=80.0)

    top = candidates[0]
    assert (top.tp, top.pp, top.dp, top.zero, top.recompute) == (8, 1, 16, 1, "selective")
    assert top.total_gb == pytest.approx(52.97, rel=1e-3)
    assert top.bubble == 0.0
    assert top.effective_hours == pytest.approx(2304.0, rel=1e-3)

    # 显存门槛不是唯一解：TP4 + full recompute 也能放下，但要多烧 2ND、慢 33%。
    # 这正是「OOM 是门槛、吞吐才是排序键」的例子。
    smaller_tp = [p for p in candidates if p.pp == 1 and p.tp < 8]
    assert smaller_tp
    assert all(p.recompute == "full" for p in smaller_tp)
    assert all(p.effective_hours > top.effective_hours for p in smaller_tp)


def test_405b_16384cards_bubble_changes_ranking():
    """405B on 2048×8：m=32 时 top-1 是 PP4，不是最省显存的 PP16。"""
    candidates = feasible_placements(
        machines=2048,
        cards_per_machine=8,
        params_b=405,
        seq_len=4096,
        layers=126,
        hidden=16384,
        heads=64,
        micro_batches=32,
    )
    assert_invariants(candidates, world_size=16384, cards_per_machine=8, hbm_gb=80.0)

    top = candidates[0]
    assert (top.tp, top.pp, top.dp, top.zero, top.recompute) == (8, 4, 512, 1, "selective")
    assert top.total_gb == pytest.approx(59.91, rel=1e-3)
    assert top.bubble == pytest.approx(3 / 35, rel=1e-6)
    assert top.effective_hours == pytest.approx(113.9, rel=1e-3)

    # 存在更省显存但更慢的方案：显存不是排序键。
    more_memory_efficient = [p for p in candidates if p.total_gb < top.total_gb]
    assert more_memory_efficient
    assert all(p.effective_hours >= top.effective_hours for p in more_memory_efficient)
