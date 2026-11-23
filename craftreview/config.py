"""业务策略与阈值。

标注手册中的规则是下限：输入策略只能更严格，不能放宽。这样即使支持表
误写了一个较低阈值，也不会让不合格的细类上线。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Iterable, Mapping

from .errors import PolicyError


MANUAL_FLOOR_ACCURACY = 0.80
MANUAL_FLOOR_FAMILIES = 2
MANUAL_ISOLATION_UNIT = "family"

_ACCURACY_KEYS = {"fine_accuracy_threshold", "accuracy_threshold", "precision_lower_bound"}
_FAMILY_KEYS = {"min_families", "min_independent_families", "min_evidence_families"}
_ISOLATION_KEYS = {"isolation", "isolation_unit", "split_isolation"}
_CONFIDENCE_KEYS = {"confidence", "confidence_level"}
_REVIEW_MINUTES_KEYS = {"review_minutes", "minutes_per_review"}
_ABSTENTION_KEYS = {"abstention_cost", "coarse_fallback_cost"}
_COST_KEYS = {
    "cost_high": "high",
    "cost_medium": "medium",
    "cost_low": "low",
}
_NON_NUMERIC_ITEMS = {"scope", "version", "note", "description"}


def _decimal(value: str, field_name: str) -> Decimal:
    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise PolicyError(f"{field_name} 不是有效数字：{value!r}") from exc


def _bool(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "是"}:
        return True
    if text in {"0", "false", "no", "n", "否"}:
        return False
    raise PolicyError(f"{field_name} 不是有效布尔值：{value!r}")


@dataclass(frozen=True)
class Policy:
    """可执行的手册规则。

    ``accuracy_confidence`` 与 ``one_sided`` 决定精度下界。默认使用单侧 95%
    Wilson 下界，因为业务问题只关心“真实精度是否至少达到阈值”。
    """

    accuracy_threshold: float = MANUAL_FLOOR_ACCURACY
    min_families: int = MANUAL_FLOOR_FAMILIES
    isolation_unit: str = MANUAL_ISOLATION_UNIT
    confidence: float = 0.95
    one_sided: bool = True
    review_minutes: int = 20
    abstention_cost: float = 1.0
    costs: Mapping[str, float] = field(
        default_factory=lambda: {"high": 6.0, "medium": 3.0, "low": 1.0}
    )
    active_version: str | None = None
    review_scope: str | None = None
    warnings: tuple[str, ...] = ()

    def validate(self) -> None:
        if not 0.0 < self.accuracy_threshold <= 1.0:
            raise PolicyError("accuracy_threshold 必须在 (0, 1] 内")
        if self.accuracy_threshold < MANUAL_FLOOR_ACCURACY:
            raise PolicyError(
                f"accuracy_threshold={self.accuracy_threshold} 低于手册下限 "
                f"{MANUAL_FLOOR_ACCURACY}"
            )
        if self.min_families < MANUAL_FLOOR_FAMILIES:
            raise PolicyError(
                f"min_families={self.min_families} 低于手册下限 {MANUAL_FLOOR_FAMILIES}"
            )
        if self.isolation_unit != MANUAL_ISOLATION_UNIT:
            raise PolicyError(
                f"isolation_unit={self.isolation_unit!r} 不允许；手册要求按 "
                f"{MANUAL_ISOLATION_UNIT} 隔离"
            )
        if not 0.5 <= self.confidence < 1.0:
            raise PolicyError("confidence 必须在 [0.5, 1) 内")
        if self.review_minutes <= 0:
            raise PolicyError("review_minutes 必须为正整数")
        if self.abstention_cost < 0:
            raise PolicyError("abstention_cost 不能为负")
        for name, value in self.costs.items():
            if value < 0:
                raise PolicyError(f"costs[{name!r}] 不能为负")
        if self.abstention_cost > self.costs["high"]:
            raise PolicyError(
                "abstention_cost 不能高于高成本错误成本，否则会鼓励冒险猜细类"
            )

    def cost_for_severity(self, severity: str | None) -> float:
        if severity is None:
            return self.costs["medium"]
        return self.costs.get(severity, self.costs["medium"])

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Mapping[str, str]],
        warnings: list[str] | None = None,
    ) -> "Policy":
        warnings = warnings if warnings is not None else []
        data: dict[str, object] = {}
        extra: dict[str, str] = {}
        costs = {"high": 6.0, "medium": 3.0, "low": 1.0}
        for row in rows:
            item = (row.get("item") or "").strip().lower()
            rule = (row.get("rule") or "").strip()
            if not item:
                warnings.append("策略行缺少 item，已忽略")
                continue
            if item in _ACCURACY_KEYS:
                data["accuracy_threshold"] = float(_decimal(rule, item))
            elif item in _FAMILY_KEYS:
                value = int(_decimal(rule, item))
                data["min_families"] = value
            elif item in _ISOLATION_KEYS:
                data["isolation_unit"] = rule.lower()
            elif item in _CONFIDENCE_KEYS:
                value = float(_decimal(rule, item))
                if value > 1.0:  # 允许写成 95 表示 95%
                    value /= 100.0
                data["confidence"] = value
            elif item in _REVIEW_MINUTES_KEYS:
                data["review_minutes"] = int(_decimal(rule, item))
            elif item in _ABSTENTION_KEYS:
                data["abstention_cost"] = float(_decimal(rule, item))
            elif item in _COST_KEYS:
                costs[_COST_KEYS[item]] = float(_decimal(rule, item))
            elif item == "one_sided":
                data["one_sided"] = _bool(rule, item)
            elif item == "version":
                data["active_version"] = rule or None
            elif item == "scope":
                data["review_scope"] = rule or None
            elif item in _NON_NUMERIC_ITEMS:
                pass
            else:
                extra[item] = rule
                warnings.append(f"未识别的策略项 {item!r}，未用于计算")
        data["costs"] = costs
        data["warnings"] = tuple(warnings) + tuple(
            f"策略扩展项 {k}={v}" for k, v in sorted(extra.items())
        )
        policy = cls(**data)
        policy.validate()
        return policy
