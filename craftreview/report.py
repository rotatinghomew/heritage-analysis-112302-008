"""把评估结果组织成编辑人员可读的报告。"""

from __future__ import annotations

from typing import Mapping, Sequence

from .eligibility import FineClassEvidence, GoldResolution
from .models import Evaluation, Selection
from .migration import MigrationPlan
from .review import ReviewPlan


def render_evaluation(evaluation: Evaluation, policy) -> list[str]:
    lines: list[str] = []
    lines.append(f"模型 {evaluation.model}：{evaluation.status}")
    if not evaluation.valid:
        for violation in evaluation.isolation_violations:
            lines.append(f"  - 片源隔离违例：{violation['group']} -> {violation['splits']}")
        lines.append("  - 本次评估不给总体准确率；请先按家族重新分组")
        return lines
    counts = evaluation.by_granularity()
    lines.append(
        f"  输出覆盖：细类 {counts.get('fine', 0)}，父类 {counts.get('coarse', 0)}，"
        f"拒判 {counts.get('reject', 0)}；总体覆盖 {evaluation.coverage:.0%}"
    )
    lines.append(f"  成本（按课程用途加权）：{evaluation.total_cost:.1f}")
    for row in evaluation.by_scope():
        lines.append(
            f"  地域 {row['tradition']} / 工序 {row['task']} / 器具 {row['context'] or '通用'}："
            f"支持 {row['support']}，细类覆盖 {row['fine_coverage']:.0%}"
            f"（错 {row['fine_wrong']}，高代价 {row['high_cost_errors']}），"
            f"父类覆盖 {row['parent_coverage']:.0%}，拒判率 {row['reject_rate']:.0%}"
        )
    for (task, tradition, context, fine), outcome in sorted(evaluation.outcomes.items(), key=lambda i: (i[0][0], i[0][1], i[0][2] or "", i[0][3])):
        if outcome.fine_wrong or outcome.parent_wrong or outcome.rejected:
            lines.append(
                f"  误分/拒判 {task}/{tradition}/{context or '通用'}/{fine}："
                f"细类错 {outcome.fine_wrong}，父类错 {outcome.parent_wrong}，拒判 {outcome.rejected}"
            )
    for warning in evaluation.warnings:
        lines.append(f"  注意：{warning}")
    return lines


def render_selection(selection: Selection) -> list[str]:
    lines = ["模型比较（成本敏感，先剔除隔离无效）："]
    for name, score, note in selection.ranking:
        score_text = "不可选" if score == float("inf") else f"{score:.1f}"
        lines.append(f"  - {name}: 分数 {score_text}；{note}")
    lines.append(f"结论：{selection.reason}")
    return lines


def render_evidence(evidence: Mapping[tuple[str, str, str | None, str], FineClassEvidence]) -> list[str]:
    lines = ["细类上线资格："]
    for item in sorted(evidence.values(), key=lambda e: (e.task, e.tradition, e.context or "", e.fine)):
        lines.append(f"  - {item.explain()}")
        if item.coarse_only_reviews:
            lines.append(
                f"    只有父类一致的复核 {item.coarse_only_reviews} 次，未用于细类验证"
            )
        if item.invalidated_by:
            lines.append(f"    被核对撤销：{', '.join(item.invalidated_by)}")
    online = [item for item in evidence.values() if item.status == "online"]
    parent = [item for item in evidence.values() if item.status != "online"]
    lines.append(
        f"  可上线细类：{', '.join(e.fine for e in online) if online else '无'}"
    )
    lines.append(
        f"  继续沿用上位概念：{', '.join(e.fine for e in parent) if parent else '无'}"
    )
    return lines


def render_gold(gold: Mapping[str, GoldResolution]) -> list[str]:
    lines = ["片段级复核结论："]
    for clip_id, result in sorted(gold.items()):
        lines.append(
            f"  - {clip_id}: {result.basis}；细类={result.fine or '—'}，"
            f"父类={result.coarse or '—'}，复核人={', '.join(result.reviewers) or '—'}"
        )
    return lines


def render_migration(plan: MigrationPlan) -> list[str]:
    lines = [f"目录迁移 {plan.from_version} -> {plan.to_version}："]
    for change in plan.changes:
        lines.append(f"  - 目录变化 {change.code}: {change.change}")
    for action in plan.actions:
        lines.append(
            f"  - {action.course_id}/{action.chapter} 片段 {action.clip_id or '—'}："
            f"{action.action} -> {action.new_code or '—'}；{action.reason}"
        )
        if action.blockers:
            lines.append(f"    阻塞：{', '.join(action.blockers)}")
    lines.append(
        f"  需换说明 {len(plan.pending_updates())} 章，"
        f"可继续引用 {len(plan.keep_parent())} 章，"
        f"阻塞 {len(plan.blocked())} 章"
    )
    for warning in plan.warnings:
        lines.append(f"  注意：{warning}")
    return lines


def render_review_plan(plan: ReviewPlan) -> list[str]:
    lines = [f"{plan.period} 复核排期：已排 {plan.total_minutes} 分钟"]
    for reviewer_id, assignments in sorted(plan.by_reviewer().items()):
        lines.append(f"  - {reviewer_id}：{len(assignments)} 项")
        for assignment in assignments:
            task = assignment.task
            lines.append(
                f"    {task.clip_id}（{task.tradition}/{task.task}/{task.fine or task.coarse or '—'}）"
                f"{assignment.minutes} 分钟；{task.reason}"
            )
    if plan.not_needed:
        lines.append("  无需再排（复核不会增加独立家族）：")
        for task in plan.not_needed:
            lines.append(f"    {task.clip_id}：{task.risk_note or task.reason}")
    if plan.unassigned:
        lines.append("  未排上：")
        for task in plan.unassigned:
            lines.append(
                f"    {task.clip_id}（{task.tradition}/{task.task}）{task.minutes} 分钟；"
                f"需要 {task.tradition} 支系的 {task.needed_skill} 复核人"
            )
    for need in plan.capability_needs:
        lines.append(
            f"  能力缺口：{need.tradition}/{need.skill} 需 {need.minutes} 分钟，{need.tasks} 个片段"
        )
    for note in plan.notes:
        lines.append(f"  注意：{note}")
    return lines


def render_ungradable(gaps: Sequence) -> list[str]:
    lines = ["仍无法升级的课件："]
    if not gaps:
        lines.append("  无")
    for gap in gaps:
        lines.append(
            f"  - {gap.clip_id}（{gap.tradition}/{gap.task}）：{gap.reason}；"
            f"需要 {gap.needed_skill} 资质"
        )
    return lines


def render_summary(
    evaluation: Evaluation | None,
    selection: Selection | None,
    evidence: Mapping[tuple[str, str, str | None, str], FineClassEvidence],
    plan: MigrationPlan | None,
    review_plan: ReviewPlan | None,
) -> list[str]:
    lines = ["摘要："]
    if selection:
        lines.append(f"  - {selection.reason}")
    online = [item for item in evidence.values() if item.status == "online"]
    parent = [item for item in evidence.values() if item.status != "online"]
    lines.append(f"  - 可上线细类 {len(online)} 个；继续沿用上位概念 {len(parent)} 个")
    if evaluation and not evaluation.valid:
        lines.append("  - 模型评估因片源隔离问题无效，不能作为改版依据")
    if plan:
        lines.append(
            f"  - 迁移：需换说明 {len(plan.pending_updates())} 章，"
            f"沿用 {len(plan.keep_parent())} 章，阻塞 {len(plan.blocked())} 章"
        )
    if review_plan:
        lines.append(
            f"  - 复核：已排 {len(review_plan.assignments)} 项，"
            f"未排 {len(review_plan.unassigned)} 项"
        )
    return lines
