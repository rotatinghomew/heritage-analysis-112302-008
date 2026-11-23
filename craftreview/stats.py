"""区间估计。

细类样本往往很小，点估计会被一两个正确样本抬高。手册要求“精度下界达到
阈值”，因此这里用单侧 Wilson 下界；样本为零时返回 0，而不是 1。
"""

from __future__ import annotations

import math
from statistics import NormalDist


def wilson_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
    one_sided: bool = True,
) -> tuple[float, float]:
    """返回 Wilson 区间。

    ``one_sided=True`` 时返回 ``(下界, 点估计)``，上界即点估计；业务只使用
    下界。单侧置信水平通过把 ``confidence`` 映射为双侧 ``1-(1-confidence)*2``
    实现。
    """
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("successes 必须在 [0, total] 内")
    if not 0.5 <= confidence < 1.0:
        raise ValueError("confidence 必须在 [0.5, 1) 内")
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    alpha = 1.0 - confidence
    if one_sided:
        z = NormalDist().inv_cdf(1.0 - alpha)
    else:
        z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2 * total)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total))
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    if one_sided:
        return (lower, p)
    return (lower, upper)


def wilson_lower_bound(
    successes: int,
    total: int,
    confidence: float = 0.95,
    one_sided: bool = True,
) -> float:
    return wilson_interval(successes, total, confidence, one_sided)[0]


def precision_lower_bound(
    correct: int,
    total: int,
    confidence: float = 0.95,
    one_sided: bool = True,
) -> float:
    """细类精度下界；无样本时 0.0。"""
    return wilson_lower_bound(correct, total, confidence, one_sided)
