"""细类上线资格。

规则链（全部满足才允许细类上线）：

1. 片源家族隔离有效；
2. 细类在目录中已批准，且该上下文允许细类；
3. 至少两个不同证据家族的独立复核样本；
4. 复核样本本身是细类级别的双评（只有父类一致的双评不计入）；
5. 细类精度下界达到手册阈值；
6. 未被“同一表演”核对撤销资格。

任何一项不满足都退回父类，而不是用总体准确率补足。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Iterable, Mapping, Sequence

from .data import CatalogEntry, Clip, Finding, ReviewRecord
from .families import FamilyResolution
from .models import Evaluation


@dataclass
class FineClassEvidence:
    task: str
    tradition: str
    fine: str
    coarse: str | None
    context: str | None = None
    eligible_families: tuple[str, ...] = ()
    corpus_families: tuple[str, ...] = ()
    corroborated_clips: tuple[str, ...] = ()
    fine_reviews: int = 0
    coarse_only_reviews: int = 0
    evaluated_predictions: int = 0
    correct_predictions: int = 0
    precision_lower: float = 0.0
    catalog_approved: bool = False
    context_allowed: bool = True
    invalidated_by: tuple[str, ...] = ()
    blockers: list[str] = field(default_factory=list)

    @property
    def family_count(self) -> int:
        return len(self.eligible_families)

    @property
    def corpus_family_count(self) -> int:
        """已入库的独立家族数（不限复核状态）。"""
        return len(self.corpus_families)

    @property
    def can_ever_reach_two_families(self) -> bool:
        return self.corpus_family_count >= 2

    @property
    def status(self) -> str:
        if (
            self.family_count >= 2
            and self.precision_lower > 0
            and "fine_disagreement" not in self.blockers
            and not self.blockers
        ):
            return "online"
        if "invalid_isolation" in self.blockers:
            return "blocked_isolation"
        if "invalidated_by_finding" in self.blockers:
            return "invalidated"
        if "not_in_catalog" in self.blockers or "context_not_allowed" in self.blockers:
            return "catalog_blocked"
        if "fine_disagreement" in self.blockers:
            return "unresolved"
        if self.family_count < 2:
            return "insufficient_families"
        return "insufficient_precision"

    @property
    def recommended_granularity(self) -> str:
        return "fine" if self.status == "online" else "coarse"

    def explain(self) -> str:
        scope = f"{self.task}/{self.context or '通用'}/{self.fine}"
        if self.status == "online":
            return (
                f"{scope}: 可上线；{self.family_count} 个独立家族，"
                f"精度下界 {self.precision_lower:.2f}"
            )
        reasons = "；".join(self.blockers) if self.blockers else self.status
        suffix = ""
        if (
            self.status != "online"
            and not self.can_ever_reach_two_families
            and self.fine
        ):
            suffix = "；现有片源只有 1 个家族，补复核时间无用，需补不同家族的片源"
        return (
            f"{scope}: 继续使用上位概念（{self.coarse or '父类'}）；"
            f"原因：{reasons}{suffix}"
        )


def evidence_key(
    task: str, tradition: str, context: str | None, fine: str
) -> tuple[str, str, str | None, str]:
    """细类证据的原子键：同名分支在不同器具下不共享证据。"""
    return (task, tradition, context, fine)


def _context_allowed(entry: CatalogEntry | None, clip_task: str, clip_context: str | None) -> bool:
    if entry is None:
        return False
    if entry.max_granularity != "fine":
        return False
    if not entry.scope_notes:
        return True
    # scope_notes 支持 "task:join|carve; context:oak|elm" 形式；未写维度即不限制
    allowed = True
    for chunk in entry.scope_notes.replace("；", ";").split(";"):
        if ":" not in chunk:
            continue
        dim, values = chunk.split(":", 1)
        dim = dim.strip().lower()
        options = {v.strip().lower() for v in values.replace("|", ",").split(",") if v.strip()}
        if not options:
            continue
        if dim in {"task", "工序"}:
            allowed = allowed and clip_task.lower() in options
        elif dim in {"context", "器具"}:
            allowed = allowed and (clip_context or "").lower() in options
    return allowed


def _is_fine_review(review: ReviewRecord) -> bool:
    return review.verdict == "fine" and bool(review.fine)


def _review_pairs_by_clip(reviews: Iterable[ReviewRecord]) -> dict[str, list[ReviewRecord]]:
    grouped: dict[str, list[ReviewRecord]] = defaultdict(list)
    for review in reviews:
        grouped[review.clip_id].append(review)
    return grouped


@dataclass(frozen=True)
class GoldResolution:
    """片段级复核结论。"""

    clip_id: str
    fine: str | None
    coarse: str | None
    basis: str
    reviewers: tuple[str, ...]
    families: tuple[str, ...]

    @property
    def resolved_fine(self) -> bool:
        return self.basis == "double_fine_agreement"


def resolve_gold(
    reviews_by_clip: Mapping[str, Sequence[ReviewRecord]],
    clip_family: Mapping[str, str],
    resolution: FamilyResolution,
    clip_tradition: Mapping[str, str] | None = None,
) -> dict[str, GoldResolution]:
    """把复核记录归并为片段级结论。

    * 两位（及以上）复核人细类一致 -> 可用于细类验证的片段级样本；
    * 只有父类一致 -> 仅作父类结论，不能用于细类；
    * 细类不一致 -> 未解决，不得硬判子类；
    * 复核人所属支系与片段不一致的复核不参与归并。
    """
    clip_tradition = clip_tradition or {}
    gold: dict[str, GoldResolution] = {}
    for clip_id, reviews in reviews_by_clip.items():
        tradition = clip_tradition.get(clip_id)
        valid_reviews = [
            review
            for review in reviews
            if not tradition or not review.tradition or review.tradition == tradition
        ]
        fine_reviews = [r for r in valid_reviews if _is_fine_review(r)]
        reviewers = tuple(sorted({r.reviewer for r in valid_reviews}))
        families = tuple(
            sorted({resolution.group_key(clip_family.get(clip_id, ""))} - {""})
        )
        if len(fine_reviews) >= 2:
            fines = {r.fine for r in fine_reviews}
            if len(fines) == 1:
                gold[clip_id] = GoldResolution(
                    clip_id=clip_id,
                    fine=next(iter(fines)),
                    coarse=next((r.coarse for r in fine_reviews if r.coarse), None),
                    basis="double_fine_agreement",
                    reviewers=reviewers,
                    families=families,
                )
                continue
            gold[clip_id] = GoldResolution(
                clip_id=clip_id,
                fine=None,
                coarse=next((r.coarse for r in valid_reviews if r.coarse), None),
                basis="fine_disagreement",
                reviewers=reviewers,
                families=families,
            )
            continue
        coarse_votes = {r.coarse for r in valid_reviews if r.coarse}
        if len(valid_reviews) >= 2 and len(coarse_votes) == 1:
            gold[clip_id] = GoldResolution(
                clip_id=clip_id,
                fine=None,
                coarse=next(iter(coarse_votes)),
                basis="coarse_only_agreement",
                reviewers=reviewers,
                families=families,
            )
        elif len(valid_reviews) >= 2:
            gold[clip_id] = GoldResolution(
                clip_id=clip_id,
                fine=None,
                coarse=None,
                basis="unresolved",
                reviewers=reviewers,
                families=families,
            )
        else:
            gold[clip_id] = GoldResolution(
                clip_id=clip_id,
                fine=None,
                coarse=(valid_reviews[0].coarse if valid_reviews else None),
                basis="single_review",
                reviewers=reviewers,
                families=families,
            )
    return gold


def _admissible_reviews(
    reviews: Sequence[ReviewRecord],
    reviewers: Mapping[str, object],
    warnings: list[str],
) -> list[ReviewRecord]:
    """按复核人名册校正复核记录。

    只有传承支系是硬门槛：支系不符的复核不能佐证本支系的细类，降级为父类
    结论并提示。技能栏未单列该细类时只提示，不擅自否定复核人的判断——手册
    只规定“只有父类一致的双评不得用于细类验证”，没有要求细类必须由同名
    技能的人签署。
    """
    result: list[ReviewRecord] = []
    noted: set[tuple[str, str, str | None]] = set()
    for review in reviews:
        roster = reviewers.get(review.reviewer)
        if roster is None:
            if review.verdict == "fine" and review.fine:
                warnings.append(
                    f"复核人 {review.reviewer} 不在名册中，其细类记录无法核验资质"
                )
            result.append(review)
            continue
        skill = getattr(roster, "skill", None)
        tradition = getattr(roster, "tradition", None)
        if review.tradition and tradition and review.tradition != tradition:
            warnings.append(
                f"复核 {review.review_id} 的支系 {review.tradition} 与名册 "
                f"{tradition} 不一致，降级为父类结论"
            )
            result.append(replace(review, verdict="coarse", fine=None))
            continue
        if (
            review.verdict == "fine"
            and review.fine
            and skill
            and skill not in {review.fine, review.coarse}
            and (review.reviewer, skill, review.fine) not in noted
        ):
            noted.add((review.reviewer, skill, review.fine))
            warnings.append(
                f"复核 {review.review_id} 由 {review.reviewer}（技能 {skill}）给出细类 "
                f"{review.fine}；技能栏未单列该细类，仍按有效细类复核计，建议补登记"
            )
        result.append(review)
    return result


def evaluate_evidence(
    clips: Sequence[Clip],
    reviews: Sequence[ReviewRecord],
    catalog: Sequence[CatalogEntry],
    evaluation: Evaluation | None,
    resolution: FamilyResolution,
    policy,
    findings: Sequence[Finding] = (),
    catalog_version: str | None = None,
    reviewers: Mapping[str, object] | None = None,
) -> tuple[
    dict[tuple[str, str, str | None, str], FineClassEvidence],
    dict[str, GoldResolution],
    list[str],
]:
    """汇总每个细类的证据与上线资格。"""
    warnings: list[str] = []
    if reviewers:
        reviews = _admissible_reviews(reviews, reviewers, warnings)
    clip_by_id = {clip.clip_id: clip for clip in clips}
    clip_family = {clip.clip_id: clip.family for clip in clips}
    clip_tradition = {clip.clip_id: clip.tradition for clip in clips}
    reviews_by_clip = _review_pairs_by_clip(reviews)
    gold = resolve_gold(reviews_by_clip, clip_family, resolution, clip_tradition)
    findings_by_clip: dict[str, list[Finding]] = defaultdict(list)
    for finding in findings:
        if not finding.is_same_performance:
            continue
        for clip_id in finding.clips:
            findings_by_clip[clip_id].append(finding)

    evidence: dict[tuple[str, str, str | None, str], FineClassEvidence] = {}
    # 先登记所有片段中出现的细类
    for clip in clips:
        if not clip.fine:
            continue
        key = evidence_key(clip.task, clip.tradition, clip.context, clip.fine)
        evidence.setdefault(
            key,
            FineClassEvidence(
                task=clip.task,
                tradition=clip.tradition,
                fine=clip.fine,
                coarse=clip.coarse,
                context=clip.context,
            ),
        )

    # 片段级复核：细类结论需要两位细类复核且家族独立
    fine_by_family: dict[tuple[str, str, str | None, str], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for clip_id, result in gold.items():
        clip = clip_by_id.get(clip_id)
        if clip is None:
            warnings.append(f"复核记录引用未知片段 {clip_id}，已忽略")
            continue
        if result.basis != "double_fine_agreement" or not result.fine:
            continue
        key = evidence_key(clip.task, clip.tradition, clip.context, result.fine)
        item = evidence.setdefault(
            key,
            FineClassEvidence(
                task=clip.task,
                tradition=clip.tradition,
                fine=result.fine,
                coarse=result.coarse or clip.coarse,
                context=clip.context,
            ),
        )
        item.fine_reviews += 1
        for family in result.families:
            fine_by_family[key][family].add(clip_id)
        item.corroborated_clips = tuple(sorted(set(item.corroborated_clips) | {clip_id}))

    # 只有父类一致的双评 / 细类分歧：记录下来，明确不能用于细类
    for clip_id, result in gold.items():
        clip = clip_by_id.get(clip_id)
        if clip is None or not clip.fine:
            continue
        key = evidence_key(clip.task, clip.tradition, clip.context, clip.fine)
        item = evidence.setdefault(
            key,
            FineClassEvidence(
                task=clip.task,
                tradition=clip.tradition,
                fine=clip.fine,
                coarse=clip.coarse,
                context=clip.context,
            ),
        )
        if result.basis == "coarse_only_agreement":
            item.coarse_only_reviews += 1
        elif result.basis == "fine_disagreement":
            item.blockers.append("fine_disagreement")

    corpus_by_key: dict[tuple[str, str, str | None, str], set[str]] = defaultdict(set)
    for clip in clips:
        if clip.fine and clip.split != "train":
            corpus_by_key[evidence_key(clip.task, clip.tradition, clip.context, clip.fine)].add(
                resolution.group_key(clip.family)
            )

    for key, item in evidence.items():
        item.eligible_families = tuple(sorted(fine_by_family.get(key, {})))
        item.corpus_families = tuple(sorted(corpus_by_key.get(key, ())))
        entry = _entry_for(catalog, item.fine, catalog_version)
        item.catalog_approved = bool(entry and entry.approved)
        item.context_allowed = _context_allowed(entry, item.task, item.context)
        if not item.catalog_approved:
            item.blockers.append("not_in_catalog")
        if not item.context_allowed:
            item.blockers.append("context_not_allowed")

        # 同一表演核对：若用于佐证细类的复核不是都在核对之后完成，则证据
        # 依赖了被合并的家族，迁移资格失效。核对之后补齐的复核可重新累积。
        invalidators = sorted(
            {
                finding.finding_id
                for finding in findings
                if finding.is_same_performance
                and any(clip_id in finding.clips for clip_id in item.corroborated_clips)
                and not _all_reviews_after(
                    reviews_by_clip, item.corroborated_clips, finding.decided_at
                )
            }
        )
        item.invalidated_by = tuple(invalidators)
        if invalidators:
            item.blockers.append("invalidated_by_finding")

        if evaluation is not None:
            if evaluation.status != "valid":
                item.blockers.append("invalid_isolation")
            outcome = evaluation.outcomes.get(key)
            if outcome is not None:
                item.evaluated_predictions = outcome.fine_correct + outcome.fine_wrong
                item.correct_predictions = outcome.fine_correct
                item.precision_lower = outcome.precision_lower
            else:
                item.evaluated_predictions = 0
                item.correct_predictions = 0
                item.precision_lower = 0.0
            if (
                item.precision_lower < policy.accuracy_threshold
                and "invalid_isolation" not in item.blockers
            ):
                item.blockers.append("precision_below_threshold")
        else:
            item.blockers.append("no_evaluation")
            warnings.append("未提供模型评估，细类无法计算精度下界")

        if item.family_count < policy.min_families:
            item.blockers.append("insufficient_families")
        if len(item.corpus_families) < policy.min_families:
            item.blockers.append("corpus_single_family")

    # 去重并稳定排序
    for item in evidence.values():
        item.blockers[:] = sorted(set(item.blockers))
    return evidence, gold, warnings


def _entry_for(
    catalog: Sequence[CatalogEntry], code: str, version: str | None = None
) -> CatalogEntry | None:
    """按目标版本取目录项；未指定版本时取最新版本。"""
    candidates = [entry for entry in catalog if entry.code == code]
    if not candidates:
        return None
    if version is not None:
        exact = [entry for entry in candidates if entry.version == version]
        if exact:
            return exact[0]
        # 目标版本没有该代码，视为未批准，而不是回退到旧版
        return None
    return max(candidates, key=lambda e: e.version)


def _all_reviews_after(
    reviews_by_clip: Mapping[str, Sequence[ReviewRecord]],
    clip_ids: Iterable[str],
    decided_at: str,
) -> bool:
    """所有参与细类佐证的复核都在核对之后。"""
    saw_review = False
    for clip_id in clip_ids:
        for review in reviews_by_clip.get(clip_id, ()):
            if review.verdict == "fine" and review.fine:
                saw_review = True
                if review.reviewed_at <= decided_at:
                    return False
    return saw_review


def summarize_evidence(
    evidence: Mapping[tuple[str, str, str | None, str], FineClassEvidence]
) -> dict[str, list[FineClassEvidence]]:
    buckets: dict[str, list[FineClassEvidence]] = {
        "online": [],
        "continue_parent": [],
    }
    for item in sorted(evidence.values(), key=lambda e: (e.task, e.tradition, e.context or "", e.fine)):
        if item.status == "online":
            buckets["online"].append(item)
        else:
            buckets["continue_parent"].append(item)
    return buckets
