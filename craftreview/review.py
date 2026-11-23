"""有限复核时间的排期。

每期复核时间有限，且复核人只具备某些传承支系的资格。排期不是按片段数
平均分，而是优先解决“差一个独立家族”“精度下界不足”“高成本小类”三类
缺口，并显式报告排不下的需求和下一期应按什么能力加人。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence

from .data import Clip, CourseChapter, Finding, ReviewRecord, Reviewer
from .eligibility import FineClassEvidence
from .families import FamilyResolution, resolve_families


@dataclass(frozen=True)
class ReviewTask:
    clip_id: str
    task: str
    tradition: str
    coarse: str | None
    fine: str | None
    needed_skill: str
    priority: float
    minutes: int
    reason: str
    blockers: tuple[str, ...] = ()
    risk_note: str | None = None
    family: str | None = None


@dataclass
class Assignment:
    task: ReviewTask
    reviewer: str
    minutes: int


@dataclass
class CapabilityNeed:
    tradition: str
    skill: str
    minutes: int
    tasks: int


@dataclass
class ReviewPlan:
    period: str
    assignments: list[Assignment] = field(default_factory=list)
    unassigned: list[ReviewTask] = field(default_factory=list)
    not_needed: list[ReviewTask] = field(default_factory=list)
    capability_needs: list[CapabilityNeed] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    used_minutes: dict[str, int] = field(default_factory=dict)

    @property
    def total_minutes(self) -> int:
        return sum(a.minutes for a in self.assignments)

    def by_reviewer(self) -> dict[str, list[Assignment]]:
        buckets: dict[str, list[Assignment]] = defaultdict(list)
        for assignment in self.assignments:
            buckets[assignment.reviewer].append(assignment)
        return dict(buckets)


def _class_key(clip: Clip) -> tuple[str, str, str | None, str] | None:
    if not clip.fine:
        return None
    return (clip.task, clip.tradition, clip.context, clip.fine)


def _has_double_fine(clip_id: str, reviews_by_clip: Mapping[str, Sequence[ReviewRecord]]) -> bool:
    records = reviews_by_clip.get(clip_id, ())
    fine_reviews = [r for r in records if r.verdict == "fine" and r.fine]
    return len({(r.reviewer, r.fine) for r in fine_reviews}) >= 2


def _severity_weight(severity: str, policy) -> float:
    return policy.cost_for_severity(severity) / max(policy.costs.values())


def _review_candidates(
    clips: Sequence[Clip],
    evidence: Mapping[tuple[str, str, str | None, str], FineClassEvidence],
    reviews: Sequence[ReviewRecord],
    chapters: Sequence[CourseChapter],
    policy,
    resolution: FamilyResolution,
) -> dict[str, tuple[float, str, tuple[str, ...]]]:
    reviews_by_clip: dict[str, list[ReviewRecord]] = defaultdict(list)
    for review in reviews:
        reviews_by_clip[review.clip_id].append(review)
    severity_by_clip: dict[str, str] = {}
    rank = {"low": 0, "medium": 1, "high": 2}
    for chapter in chapters:
        if chapter.clip_id and (
            chapter.clip_id not in severity_by_clip
            or rank[chapter.severity] > rank[severity_by_clip[chapter.clip_id]]
        ):
            severity_by_clip[chapter.clip_id] = chapter.severity

    candidates: dict[str, tuple[float, str, tuple[str, ...]]] = {}
    used_clips = {chapter.clip_id for chapter in chapters if chapter.clip_id}
    if not used_clips:
        # 没有课程表时无法判断哪些片源已定版，退化为“所有非训练片段”
        used_clips = {clip.clip_id for clip in clips if clip.split != "train"}

    # 关键判断：一次复核只有在“该片段所属证据组尚未用于该细类”时才增加
    # 独立家族。每个（细类，证据组）只挑一个代表片段，其余同组片段不排。
    # 优先选已进入课程、已有复核记录、高成本的片段作代表。
    def representative_rank(clip: Clip) -> tuple:
        records = reviews_by_clip.get(clip.clip_id, ())
        has_fine = any(r.verdict == "fine" and r.fine for r in records)
        # 已有细类复核的片段只差一位，优先作代表；其次是课程内片段。
        return (
            0 if has_fine else 1,
            0 if clip.clip_id in used_clips else 1,
            0 if severity_by_clip.get(clip.clip_id) == "high" else 1,
            clip.clip_id,
        )

    representatives: dict[
        tuple[tuple[str, str, str | None, str], str], Clip
    ] = {}
    for clip in clips:
        key = _class_key(clip)
        if key is None or clip.split == "train":
            continue
        item = evidence.get(key)
        if item is None or item.status == "online" or not item.can_ever_reach_two_families:
            continue
        group = resolution.group_key(clip.family)
        representative_key = (key, group)
        current = representatives.get(representative_key)
        if current is None or representative_rank(clip) < representative_rank(current):
            representatives[representative_key] = clip

    representative_ids = {clip.clip_id for clip in representatives.values()}
    for clip in clips:
        key = _class_key(clip)
        if key is None or clip.split == "train":
            continue
        item = evidence.get(key)
        if item is not None and not item.can_ever_reach_two_families:
            # 语料里只有单一非训练家族，排复核也无法凑够两个独立家族，
            # 不应占用本期复核时间，应转为征集片源。
            continue
        if item is not None and item.status == "online":
            continue
        group = resolution.group_key(clip.family)
        adds_family = bool(
            item is not None
            and group not in set(item.eligible_families)
            and representative_ids and clip.clip_id in representative_ids
        )
        is_course_clip = clip.clip_id in used_clips
        if not is_course_clip and not adds_family:
            continue
        if adds_family:
            reason = (
                f"该家族尚无细类证据；补齐后才有独立家族"
                f"（当前 {item.family_count if item else 0} 个）"
            )
        elif item is not None and item.status == "unresolved":
            reason = "双评细类不一致，需第三位细类复核裁定；不能硬判子类"
        elif _has_double_fine(clip.clip_id, reviews_by_clip):
            # 已有双细评但未上线：需要不同家族，而不是重复同家族
            reason = "已有细类双评，但独立家族不足或精度未达阈值"
        elif any(r.reviewer for r in reviews_by_clip.get(clip.clip_id, ())):
            reason = "只有父类一致或单次复核，需要第二位细类复核"
        else:
            reason = "尚无细类双评，无法用于细类验证"
        blockers = tuple(item.blockers) if item else ("no_evidence",)
        severity = severity_by_clip.get(clip.clip_id, "medium")
        score = 1.0
        if item is not None:
            if "insufficient_families" in blockers:
                score += 3.0
            if "precision_below_threshold" in blockers:
                score += 2.5
            if "invalidated_by_finding" in blockers:
                score += 4.0
            if "invalid_isolation" in blockers:
                score += 2.0
        if adds_family:
            score += 5.0
        if is_course_clip:
            score += 1.0
        score += 2.0 * _severity_weight(severity, policy)
        if severity == "high":
            reason += "；高成本小类，误判代价高"
        candidates[clip.clip_id] = (score, reason, blockers)
    return candidates


def _same_performance_risk(
    clips: Sequence[Clip],
    findings: Sequence[Finding],
    resolution: FamilyResolution,
) -> dict[str, str]:
    """标出与另一家族候选片段可能同源的复核，提示不会增加独立家族。"""
    notes: dict[str, str] = {}
    clip_by_id = {clip.clip_id: clip for clip in clips}
    for finding in findings:
        if not finding.is_same_performance:
            continue
        for clip_id in finding.clips:
            other = [c for c in finding.clips if c != clip_id and c in clip_by_id]
            if other:
                notes[clip_id] = (
                    f"核对 {finding.finding_id} 已判定与 {', '.join(other)} 同源；"
                    "后续复核不再增加独立家族"
                )
    return notes


def build_review_plan(
    clips: Sequence[Clip],
    reviewers: Sequence[Reviewer],
    evidence: Mapping[tuple[str, str, str | None, str], FineClassEvidence],
    reviews: Sequence[ReviewRecord],
    chapters: Sequence[CourseChapter],
    policy,
    *,
    period: str,
    resolution: FamilyResolution | None = None,
    findings: Sequence[Finding] = (),
    max_minutes_per_reviewer: int | None = None,
) -> ReviewPlan:
    plan = ReviewPlan(period=period)
    if resolution is None:
        resolution = resolve_families(clips, findings)
    candidates = _review_candidates(clips, evidence, reviews, chapters, policy, resolution)
    risk_notes = _same_performance_risk(clips, findings, resolution)
    clip_by_id = {clip.clip_id: clip for clip in clips}
    tasks: list[ReviewTask] = []
    for clip_id, (score, reason, blockers) in candidates.items():
        clip = clip_by_id[clip_id]
        skill = clip.fine or clip.coarse or clip.task
        tasks.append(
            ReviewTask(
                clip_id=clip_id,
                task=clip.task,
                tradition=clip.tradition,
                coarse=clip.coarse,
                fine=clip.fine,
                needed_skill=skill,
                priority=score,
                minutes=policy.review_minutes,
                reason=reason,
                blockers=blockers,
                risk_note=risk_notes.get(clip_id),
                family=clip.family,
            )
        )
    tasks.sort(key=lambda task: (-task.priority, task.clip_id))

    # 复核人本期可用时间
    budgets: dict[str, int] = defaultdict(int)
    for reviewer in reviewers:
        if reviewer.period not in {period, "unspecified"}:
            plan.notes.append(
                f"复核人 {reviewer.reviewer_id} 的记录属于 {reviewer.period}，不计入 {period} 预算"
            )
            continue
        budgets[reviewer.reviewer_id] += reviewer.minutes
    if max_minutes_per_reviewer is not None:
        for reviewer_id in list(budgets):
            budgets[reviewer_id] = min(budgets[reviewer_id], max_minutes_per_reviewer)

    review_by_id = {reviewer.reviewer_id: reviewer for reviewer in reviewers}
    assigned_reviewers: dict[str, set[str]] = defaultdict(set)
    # 同一（细类，证据组）只需要凑一个独立家族；若该组已有代表片段拿到
    # 足够复核，同组其他片段不再重复占用时间。
    group_done: set[tuple[tuple[str, str, str | None, str], str]] = set()
    # 已有双评的片段再排一次复核不会增加独立家族；同源核对已判定与另一
    # 片段同一表演的，同样不增加独立家族。这些任务只提示，不占预算。
    skip_for_independence: set[str] = {
        clip_id for clip_id in candidates if risk_notes.get(clip_id)
    }

    for task in tasks:
        if task.clip_id in skip_for_independence:
            plan.not_needed.append(task)
            if task.risk_note:
                plan.notes.append(f"{task.clip_id}：{task.risk_note}；不安排重复复核")
            continue
        clip = clip_by_id[task.clip_id]
        # 已参与该片段复核的人不能再次充当“第二人”
        already = {
            review.reviewer for review in reviews if review.clip_id == task.clip_id
        }
        existing_fine_reviewers = {
            review.reviewer
            for review in reviews
            if review.clip_id == task.clip_id and review.verdict == "fine" and review.fine
        }
        # 目标是两名不同的细类复核人；已有一人时只需要再补一人，而不是
        # 无条件排满两人。细类分歧已满足人数，但需要第三位裁定。
        # 补齐“两个独立家族”通常还要求同一片段有两位不同复核人；若该片段
        # 已有 1 位，就只差 1 位。分组完成判断在下面单独处理。
        needed = max(0, 2 - len(existing_fine_reviewers))
        if task.blockers and "fine_disagreement" in task.blockers:
            needed = 1
        if not clip_by_id[task.clip_id].fine:
            needed = 1
        if needed == 0:
            plan.not_needed.append(task)
            continue
        # 用片段所属证据组判断该家族是否已在别处补齐
        evidence_group = resolution.group_key(clip.family)
        class_key = (task.task, task.tradition, clip.context, task.fine or task.coarse or "")
        if task.fine and (class_key, evidence_group) in group_done:
            plan.not_needed.append(
                replace(
                    task,
                    reason="同一细类在该家族已有代表片段排入复核，重复复核不会增加独立家族",
                )
            )
            continue
        eligible = [
            review_by_id[reviewer_id]
            for reviewer_id in budgets
            if budgets[reviewer_id] >= task.minutes
            and reviewer_id not in already
            and review_by_id[reviewer_id].eligible_for(clip)
        ]
        assigned = False
        # 优先给余额最多的合格复核人
        for reviewer in sorted(eligible, key=lambda r: (-budgets[r.reviewer_id], r.reviewer_id)):
            if len(assigned_reviewers[task.clip_id]) >= needed:
                break
            budgets[reviewer.reviewer_id] -= task.minutes
            plan.assignments.append(Assignment(task, reviewer.reviewer_id, task.minutes))
            assigned_reviewers[task.clip_id].add(reviewer.reviewer_id)
            assigned = True
            if len(assigned_reviewers[task.clip_id]) >= needed:
                break
        if not assigned:
            plan.unassigned.append(task)
        elif task.fine and len(assigned_reviewers[task.clip_id]) >= needed:
            # 只有真正补齐了所需人数，才算该家族完成；排不下时不能阻止
            # 同组其他片段在后续尝试。
            group_done.add((class_key, evidence_group))

    plan.used_minutes = {
        reviewer_id: sum(a.minutes for a in plan.assignments if a.reviewer == reviewer_id)
        for reviewer_id in sorted(budgets)
    }
    for task in plan.unassigned:
        if task.risk_note:
            plan.notes.append(f"{task.clip_id}：{task.risk_note}")

    # 下一期能力需求：按支系×技能聚合未排上的分钟与任务数
    need_counter: Counter[tuple[str, str]] = Counter()
    task_counter: Counter[tuple[str, str]] = Counter()
    for task in plan.unassigned:
        need_counter[(task.tradition, task.needed_skill)] += task.minutes
        task_counter[(task.tradition, task.needed_skill)] += 1
    plan.capability_needs = [
        CapabilityNeed(tradition=tradition, skill=skill, minutes=minutes, tasks=task_counter[(tradition, skill)])
        for (tradition, skill), minutes in sorted(
            need_counter.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    if plan.capability_needs:
        top = plan.capability_needs[0]
        plan.notes.append(
            f"下一期最值得补充的能力：{top.tradition} 支系的 {top.skill} 复核"
            f"（缺口 {top.minutes} 分钟，{top.tasks} 个片段）"
        )
    blocked_sources = sorted(
        {
            f"{item.task}/{item.context or '通用'}/{item.fine}"
            for item in evidence.values()
            if not item.can_ever_reach_two_families and item.fine
        }
    )
    if blocked_sources:
        plan.notes.append(
            "以下细类只有单一非训练家族，补复核时间无效，需征集其他家族片源："
            + "，".join(blocked_sources)
        )
    if not plan.assignments and tasks:
        plan.notes.append("本期没有可用复核人，所有任务顺延，不应据此改版")
    return plan
