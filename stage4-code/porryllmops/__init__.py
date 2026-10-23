"""PorryLLMOps 训练控制面 MVP。

S1 只包含容量与可行放置：把 D2 的三本账从脚本搬进项目包。
"""

from .capacity import Placement, feasible_placements

__all__ = ["Placement", "feasible_placements"]
