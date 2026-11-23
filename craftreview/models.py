"""基于结构化观测的分类模型、家族隔离评估与成本敏感选型。

设计要点：

* 只在同一 ``(工序, 器具)`` 上下文内区分子类。不同器具下的同名分支不共享
  决策边界，避免把上下文差异当成子类证据。
* 证据不足或两类边界不清时输出父类；父类也不清时拒判（``None``）。
* 评估强制按片源家族隔离。检测到跨组泄漏时不给总体准确率，直接标记无效。
* 选型按小类高代价错误与目录覆盖加权，不以总体准确率作为首要指标。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from math import exp, isfinite, sqrt
from typing import Iterable, Mapping, Protocol, Sequence

from .data import Clip, Observation
from .errors import InputError
from .families import FamilyResolution
from .stats import precision_lower_bound


@dataclass(frozen=True)
class Sample:
    clip_id: str
    vector: Mapping[str, float]
    task: str
    context: str | None
    tradition: str
    family: str
    split: str
    fine: str | None
    coarse: str | None
    severity: str = "medium"
    severity_defaulted: bool = False


@dataclass(frozen=True)
class Prediction:
    fine: str | None
    coarse: str | None
    confidence: float
    margin: float
    reason: str

    @property
    def granularity(self) -> str:
        if self.fine is not None:
            return "fine"
        if self.coarse is not None:
            return "coarse"
        return "reject"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    margin_threshold: float = 0.12
    min_class_samples: int = 2
    smoothing: float = 1.0

    def build(self) -> "Classifier":
        if self.kind == "nearest_centroid":
            return NearestCentroidModel(self)
        if self.kind == "gaussian_nb":
            return GaussianNBModel(self)
        raise InputError(f"未知模型类型：{self.kind!r}")


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "centroid-strict": ModelSpec("centroid-strict", "nearest_centroid", margin_threshold=0.20),
    "centroid-balanced": ModelSpec("centroid-balanced", "nearest_centroid", margin_threshold=0.12),
    "nb-strict": ModelSpec("nb-strict", "gaussian_nb", margin_threshold=0.18),
    "nb-balanced": ModelSpec("nb-balanced", "gaussian_nb", margin_threshold=0.10),
}


class Classifier(Protocol):
    spec: ModelSpec

    def fit(self, samples: Sequence[Sample]) -> None: ...
    def predict(self, sample: Sample) -> Prediction: ...


def _feature_names(samples: Iterable[Sample]) -> list[str]:
    names: set[str] = set()
    for sample in samples:
        names.update(sample.vector)
    return sorted(names)


def _context_key(task: str, context: str | None) -> tuple[str, str | None]:
    return (task, context)


class _GroupModel:
    """单个上下文的分类器；上下文外不猜子类。"""

    def __init__(self, spec: ModelSpec, samples: Sequence[Sample]) -> None:
        self.spec = spec
        self.features = _feature_names(samples)
        self.coarse_of: dict[str, str | None] = {}
        self.counts: Counter[str] = Counter()
        rows: list[tuple[dict[str, float], str]] = []
        for sample in samples:
            if not sample.fine:
                continue
            self.counts[sample.fine] += 1
            self.coarse_of.setdefault(sample.fine, sample.coarse)
            rows.append((self._raw_vector(sample.vector), sample.fine))
        self._mean: dict[str, float] = {}
        self._scale: dict[str, float] = {}
        for name in self.features:
            values = [vector[name] for vector, _ in rows if name in vector]
            mean = sum(values) / len(values) if values else 0.0
            var = sum((v - mean) ** 2 for v in values) / len(values) if values else 1.0
            self._mean[name] = mean
            self._scale[name] = sqrt(var) or 1.0
        self.rows = [(self._normalize(vector), label) for vector, label in rows]
        self.class_stats = self._fit()

    def _raw_vector(self, vector: Mapping[str, float]) -> dict[str, float]:
        return {name: float(vector[name]) for name in self.features if name in vector}

    def _normalize(self, raw: Mapping[str, float]) -> dict[str, float]:
        return {
            name: (raw.get(name, self._mean.get(name, 0.0)) - self._mean.get(name, 0.0))
            / self._scale.get(name, 1.0)
            for name in self.features
        }

    def _fit(self) -> dict[str, object]:
        raise NotImplementedError

    def _scores(self, vector: Mapping[str, float]) -> dict[str, float]:
        raise NotImplementedError

    def predict(self, sample: Sample) -> Prediction:
        raw = self._raw_vector(sample.vector)
        scores = self._scores(self._normalize(raw))
        if not scores:
            return Prediction(None, sample.coarse, 0.0, 0.0, "no_fine_evidence")
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        best_label, best_score = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best_score - second
        if self.counts[best_label] < self.spec.min_class_samples:
            return Prediction(
                None,
                self.coarse_of.get(best_label) or sample.coarse,
                best_score,
                margin,
                f"class_support<{self.spec.min_class_samples}",
            )
        if margin < self.spec.margin_threshold:
            return Prediction(
                None,
                self.coarse_of.get(best_label) or sample.coarse,
                best_score,
                margin,
                "boundary_ambiguous",
            )
        return Prediction(
            best_label,
            self.coarse_of.get(best_label) or sample.coarse,
            best_score,
            margin,
            "fine_confident",
        )


class _NearestCentroidGroup(_GroupModel):
    def _fit(self) -> dict[str, object]:
        buckets: dict[str, list[dict[str, float]]] = defaultdict(list)
        for vector, label in self.rows:
            buckets[label].append(vector)
        centroids = {
            label: {name: sum(v[name] for v in vectors) / len(vectors) for name in self.features}
            for label, vectors in buckets.items()
        }
        return {"centroids": centroids}

    def _scores(self, vector: Mapping[str, float]) -> dict[str, float]:
        centroids: dict[str, dict[str, float]] = self.class_stats["centroids"]  # type: ignore[assignment]
        if not centroids:
            return {}
        distances = {
            label: sqrt(sum((vector[name] - center[name]) ** 2 for name in self.features))
            for label, center in centroids.items()
        }
        # 用负距离的 softmax 得到可比较的置信度
        shift = min(distances.values())
        weights = {label: exp(-(distance - shift)) for label, distance in distances.items()}
        total = sum(weights.values())
        return {label: weight / total for label, weight in weights.items()}


class _GaussianNBGroup(_GroupModel):
    def _fit(self) -> dict[str, object]:
        buckets: dict[str, list[dict[str, float]]] = defaultdict(list)
        for vector, label in self.rows:
            buckets[label].append(vector)
        stats: dict[str, dict[str, object]] = {}
        for label, vectors in buckets.items():
            means = {
                name: sum(v[name] for v in vectors) / len(vectors) for name in self.features
            }
            variances = {
                name: sum((v[name] - means[name]) ** 2 for v in vectors) / len(vectors)
                + self.spec.smoothing
                for name in self.features
            }
            stats[label] = {"mean": means, "var": variances, "prior": len(vectors) / len(self.rows)}
        return {"classes": stats}

    def _scores(self, vector: Mapping[str, float]) -> dict[str, float]:
        classes: dict[str, dict[str, object]] = self.class_stats["classes"]  # type: ignore[assignment]
        log_scores: dict[str, float] = {}
        for label, stats in classes.items():
            means = stats["mean"]  # type: ignore[assignment]
            variances = stats["var"]  # type: ignore[assignment]
            prior = float(stats["prior"])  # type: ignore[arg-type]
            score = 0.0
            for name in self.features:
                diff = vector[name] - means[name]
                score += -0.5 * (diff * diff / variances[name] + _log(variances[name]))
            log_scores[label] = score + _log(prior)
        if not log_scores:
            return {}
        shift = max(log_scores.values())
        weights = {label: exp(value - shift) for label, value in log_scores.items()}
        total = sum(weights.values())
        return {label: weight / total for label, weight in weights.items()}


def _log(value: float) -> float:
    from math import log

    return log(value) if value > 0 else -1e9


class NearestCentroidModel:
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.groups: dict[tuple[str, str | None], _GroupModel] = {}
        self.fallback_coarse: dict[str, str | None] = {}

    def fit(self, samples: Sequence[Sample]) -> None:
        buckets: dict[tuple[str, str | None], list[Sample]] = defaultdict(list)
        # 只用 train 拟合；valid 留作校准，test/holdout 只用于报告。
        for sample in samples:
            if sample.split != "train":
                continue
            buckets[_context_key(sample.task, sample.context)].append(sample)
            if sample.fine:
                self.fallback_coarse.setdefault(sample.fine, sample.coarse)
        self.groups = {
            key: _NearestCentroidGroup(self.spec, group) for key, group in buckets.items()
        }

    def predict(self, sample: Sample) -> Prediction:
        model = self.groups.get(_context_key(sample.task, sample.context))
        if model is None:
            return Prediction(
                None,
                sample.coarse or self.fallback_coarse.get(sample.fine or "", None),
                0.0,
                0.0,
                "unseen_context",
            )
        return model.predict(sample)


class GaussianNBModel:
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.groups: dict[tuple[str, str | None], _GroupModel] = {}
        self.fallback_coarse: dict[str, str | None] = {}

    def fit(self, samples: Sequence[Sample]) -> None:
        buckets: dict[tuple[str, str | None], list[Sample]] = defaultdict(list)
        # 只用 train 拟合；valid 留作校准，test/holdout 只用于报告。
        for sample in samples:
            if sample.split != "train":
                continue
            buckets[_context_key(sample.task, sample.context)].append(sample)
            if sample.fine:
                self.fallback_coarse.setdefault(sample.fine, sample.coarse)
        self.groups = {
            key: _GaussianNBGroup(self.spec, group) for key, group in buckets.items()
        }

    def predict(self, sample: Sample) -> Prediction:
        model = self.groups.get(_context_key(sample.task, sample.context))
        if model is None:
            return Prediction(
                None,
                sample.coarse or self.fallback_coarse.get(sample.fine or "", None),
                0.0,
                0.0,
                "unseen_context",
            )
        return model.predict(sample)


def build_samples(
    clips: Sequence[Clip],
    observations: Iterable[Observation],
    severity_by_clip: Mapping[str, str] | None = None,
) -> list[Sample]:
    severity_by_clip = severity_by_clip or {}
    obs_by_clip = {obs.clip_id: obs for obs in observations}
    samples: list[Sample] = []
    for clip in clips:
        obs = obs_by_clip.get(clip.clip_id, Observation(clip_id=clip.clip_id))
        vector = obs.vector()
        if not vector:
            # 没有结构化观测的片段不进入模型；仍可在评估中计为无法覆盖
            continue
        samples.append(
            Sample(
                clip_id=clip.clip_id,
                vector=vector,
                task=clip.task,
                context=clip.context,
                tradition=clip.tradition,
                family=clip.family,
                split=clip.split,
                fine=clip.fine,
                coarse=clip.coarse,
                severity=severity_by_clip.get(clip.clip_id, "medium"),
                severity_defaulted=clip.clip_id not in severity_by_clip,
            )
        )
    return samples


@dataclass
class ClassOutcome:
    task: str
    tradition: str
    fine: str
    context: str | None = None
    support: int = 0
    fine_correct: int = 0
    fine_wrong: int = 0
    parent_correct: int = 0
    parent_wrong: int = 0
    rejected: int = 0
    cost: float = 0.0
    high_cost_errors: int = 0

    @property
    def precision(self) -> float | None:
        denom = self.fine_correct + self.fine_wrong
        return self.fine_correct / denom if denom else None

    @property
    def precision_lower(self) -> float:
        return precision_lower_bound(self.fine_correct, self.fine_correct + self.fine_wrong)


@dataclass
class Evaluation:
    model: str
    status: str
    isolation_violations: list[dict[str, object]] = field(default_factory=list)
    outcomes: dict[tuple[str, str, str | None, str], ClassOutcome] = field(default_factory=dict)
    predictions: dict[str, Prediction] = field(default_factory=dict)
    total_cost: float = 0.0
    abstentions: int = 0
    rejects: int = 0
    covered: int = 0
    total: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        return self.covered / self.total if self.total else 0.0

    @property
    def valid(self) -> bool:
        return self.status == "valid"

    def fine_usable(
        self, task: str, tradition: str, context: str | None, fine: str, threshold: float
    ) -> bool:
        outcome = self.outcomes.get((task, tradition, context, fine))
        if outcome is None:
            return False
        return outcome.precision_lower >= threshold

    def by_granularity(self) -> dict[str, int]:
        counts = Counter(pred.granularity for pred in self.predictions.values())
        return dict(counts)

    def by_scope(self) -> list[dict[str, object]]:
        """按地域、工序、器具逐项给出覆盖、拒判与误分。

        总体准确率会掩盖小类；这里每个作用域单独给出细类输出数、父类回退
        数、拒判数和细类错误数，便于编辑按能力安排复核。
        """
        buckets: dict[tuple[str, str, str | None], Counter] = defaultdict(Counter)
        for (task, tradition, context, _fine), outcome in self.outcomes.items():
            counter = buckets[(tradition, task, context)]
            counter["support"] += outcome.support
            counter["fine_output"] += outcome.fine_correct + outcome.fine_wrong
            counter["fine_wrong"] += outcome.fine_wrong
            counter["parent_output"] += outcome.parent_correct + outcome.parent_wrong
            counter["parent_wrong"] += outcome.parent_wrong
            counter["rejected"] += outcome.rejected
            counter["high_cost_errors"] += outcome.high_cost_errors
            counter["cost"] += outcome.cost
        rows: list[dict[str, object]] = []
        for (tradition, task, context), counter in sorted(
            buckets.items(), key=lambda item: (item[0][0], item[0][1], item[0][2] or "")
        ):
            support = counter["support"] or 1
            rows.append(
                {
                    "tradition": tradition,
                    "task": task,
                    "context": context,
                    "support": counter["support"],
                    "fine_output": counter["fine_output"],
                    "fine_coverage": counter["fine_output"] / support,
                    "fine_wrong": counter["fine_wrong"],
                    "parent_output": counter["parent_output"],
                    "parent_coverage": counter["parent_output"] / support,
                    "parent_wrong": counter["parent_wrong"],
                    "rejected": counter["rejected"],
                    "reject_rate": counter["rejected"] / support,
                    "high_cost_errors": counter["high_cost_errors"],
                    "cost": counter["cost"],
                }
            )
        return rows

    def by_task(self) -> dict[str, dict[str, float]]:
        buckets: dict[str, Counter] = defaultdict(Counter)
        for (task, _tradition, _context, _fine), outcome in self.outcomes.items():
            buckets[task]["support"] += outcome.support
            buckets[task]["fine_correct"] += outcome.fine_correct
            buckets[task]["fine_wrong"] += outcome.fine_wrong
            buckets[task]["parent_correct"] += outcome.parent_correct
            buckets[task]["parent_wrong"] += outcome.parent_wrong
            buckets[task]["rejected"] += outcome.rejected
        result: dict[str, dict[str, float]] = {}
        for task, counter in buckets.items():
            support = counter["support"] or 1
            result[task] = {
                "support": counter["support"],
                "fine_precision": counter["fine_correct"] / max(1, counter["fine_correct"] + counter["fine_wrong"]),
                "fine_coverage": (counter["fine_correct"] + counter["fine_wrong"]) / support,
                "parent_coverage": counter["parent_correct"] / support,
                "reject_rate": counter["rejected"] / support,
            }
        return result


def _isolation_violations(
    samples: Sequence[Sample], resolution: FamilyResolution
) -> list[dict[str, object]]:
    evaluated = [s for s in samples if s.split in {"train", "valid", "test", "holdout"}]
    by_group: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for sample in evaluated:
        by_group[resolution.group_key(sample.family)][sample.split].append(sample.clip_id)
    violations: list[dict[str, object]] = []
    for key, splits in sorted(by_group.items()):
        if len(set(splits)) > 1:
            violations.append({"group": key, "splits": {k: sorted(v) for k, v in splits.items()}})
    return violations


def severities_from_courses(chapters: Iterable) -> dict[str, str]:
    """从课程章节得到片段的高成本程度；同一片段取最高。"""
    rank = {"low": 0, "medium": 1, "high": 2}
    result: dict[str, str] = {}
    for chapter in chapters:
        clip_id = getattr(chapter, "clip_id", None)
        severity = getattr(chapter, "severity", "medium")
        if not clip_id:
            continue
        if clip_id not in result or rank[severity] > rank[result[clip_id]]:
            result[clip_id] = severity
    return result


def _evaluate_model(
    classifier: Classifier,
    samples: Sequence[Sample],
    resolution: FamilyResolution,
    policy,
    allowed_splits: set[str],
) -> Evaluation:
    violations = _isolation_violations(samples, resolution)
    evaluation = Evaluation(
        model=classifier.spec.name,
        status="invalid_isolation" if violations else "valid",
        isolation_violations=violations,
    )
    eval_samples = [s for s in samples if s.split in allowed_splits]
    evaluation.total = len(eval_samples)
    if violations:
        evaluation.warnings.append(
            "检测到同一片源家族跨训练/检验组，评估无效；不报告总体准确率"
        )
        return evaluation
    for sample in eval_samples:
        prediction = classifier.predict(sample)
        evaluation.predictions[sample.clip_id] = prediction
        key = (sample.task, sample.tradition, sample.context, sample.fine or "unlabelled")
        outcome = evaluation.outcomes.setdefault(
            key,
            ClassOutcome(
                task=sample.task,
                tradition=sample.tradition,
                fine=sample.fine or "unlabelled",
                context=sample.context,
            ),
        )
        outcome.support += 1
        if prediction.granularity == "fine":
            evaluation.covered += 1
            if prediction.fine == sample.fine:
                outcome.fine_correct += 1
            else:
                outcome.fine_wrong += 1
                cost = policy.cost_for_severity(sample.severity)
                outcome.cost += cost
                evaluation.total_cost += cost
                if sample.severity == "high":
                    outcome.high_cost_errors += 1
        elif prediction.granularity == "coarse":
            evaluation.abstentions += 1
            evaluation.total_cost += policy.abstention_cost
            outcome.cost += policy.abstention_cost
            if prediction.coarse and sample.coarse and prediction.coarse == sample.coarse:
                outcome.parent_correct += 1
            else:
                outcome.parent_wrong += 1
        else:
            evaluation.rejects += 1
            evaluation.total_cost += policy.abstention_cost * 2
            outcome.rejected += 1
    defaulted = [s.clip_id for s in eval_samples if s.severity_defaulted]
    if defaulted:
        evaluation.warnings.append(
            f"{len(defaulted)} 个检验片段未关联课程用途，按中等成本计；"
            "这些片段若实际用于高成本课程，小类错误成本会被低估"
        )
    return evaluation


def evaluate_models(
    samples: Sequence[Sample],
    resolution: FamilyResolution,
    policy,
    model_names: Iterable[str] | None = None,
    allowed_splits: Iterable[str] = ("test", "holdout"),
) -> dict[str, Evaluation]:
    """训练并评估一个或多个模型，按家族隔离检查。"""
    evaluations: dict[str, Evaluation] = {}
    names = list(model_names or MODEL_REGISTRY)
    for name in names:
        spec = MODEL_REGISTRY.get(name)
        if spec is None:
            raise InputError(f"未注册的模型：{name!r}")
        classifier = spec.build()
        classifier.fit(samples)
        evaluations[name] = _evaluate_model(
            classifier, samples, resolution, policy, set(allowed_splits)
        )
    return evaluations


@dataclass
class Selection:
    chosen: str | None
    ranking: list[tuple[str, float, str]]
    reason: str
    high_cost_guard: bool = False


def select_model(
    evaluations: Mapping[str, Evaluation],
    policy,
    min_coverage: float = 0.0,
    require_valid: bool = True,
) -> Selection:
    """成本敏感选型。

    先剔除隔离无效的模型，再按字典序比较：

    1. 高代价用途上的细类错误次数（越少越好）；
    2. 课程用途加权的期望成本；
    3. 输出覆盖率（越高越好）。

    高代价错误单独作为第一关键字，不会被总体覆盖或总体准确率抵消；它已经
    包含在期望成本中，所以不再重复计费。
    """
    ranking: list[tuple[str, float, str]] = []
    high_cost_guard = False
    for name, evaluation in evaluations.items():
        if evaluation.status != "valid":
            ranking.append((name, float("inf"), "片源隔离无效，不可选"))
            continue
        high_errors = sum(outcome.high_cost_errors for outcome in evaluation.outcomes.values())
        coverage_penalty = max(0.0, min_coverage - evaluation.coverage) * 100.0
        # 主分数：高代价错误次数占主导，成本作为第二量纲。1e6 的系数只用于
        # 保证任何一次高代价错误都排在零高代价错误之后。
        score = high_errors * 1_000_000 + evaluation.total_cost + coverage_penalty
        note = (
            f"成本={evaluation.total_cost:.1f}；高代价错误={high_errors} 次；"
            f"覆盖={evaluation.coverage:.0%}；拒判={evaluation.rejects}"
        )
        ranking.append((name, score, note))
        if high_errors:
            high_cost_guard = True
    ranking.sort(key=lambda item: (item[1], -evaluations[item[0]].coverage, item[0]))
    eligible = [item for item in ranking if isfinite(item[1])]
    if require_valid and not eligible:
        return Selection(None, ranking, "没有通过家族隔离的候选模型", high_cost_guard)
    if not eligible:
        return Selection(None, ranking, "没有可用候选模型", high_cost_guard)
    chosen, _score, note = eligible[0]
    if high_cost_guard:
        reason = (
            f"选择 {chosen}：{note}。所有候选都含高代价小类错误，已优先比较其"
            "次数；请结合逐小类下界决定是否只发布父类。"
        )
    else:
        reason = f"选择 {chosen}：{note}。"
    return Selection(chosen, ranking, reason, high_cost_guard)
