"""命令行入口：``python -m craftreview analyze --data fixtures``。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass

from .catalog import catalog_scope_changes
from .errors import CraftReviewError
from .pipeline import Dataset, analyze
from .report import (
    render_evidence,
    render_evaluation,
    render_gold,
    render_migration,
    render_review_plan,
    render_selection,
    render_summary,
    render_ungradable,
)


def _json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def _dedupe_violations(violations) -> list[dict]:
    """每个模型都会报告同一批违例；按违例内容去重，避免重复刷屏。"""
    seen: dict[str, dict] = {}
    for violation in violations:
        key = f"{violation['group']}|{sorted(violation['splits'])}"
        seen.setdefault(key, violation)
    return list(seen.values())


def _analysis_payload(analysis) -> dict:
    payload: dict = {
        "summary": {
            "online": sorted(e.fine for e in analysis.evidence.values() if e.status == "online"),
            "continue_parent": sorted(
                e.fine for e in analysis.evidence.values() if e.status != "online"
            ),
            "chosen_model": analysis.selection.chosen if analysis.selection else None,
            "warnings": analysis.warnings,
        },
        "evidence": [
            {
                "task": item.task,
                "tradition": item.tradition,
                "fine": item.fine,
                "coarse": item.coarse,
                "context": item.context,
                "status": item.status,
                "families": list(item.eligible_families),
                "family_count": item.family_count,
                "precision_lower": round(item.precision_lower, 4),
                "coarse_only_reviews": item.coarse_only_reviews,
                "invalidated_by": list(item.invalidated_by),
                "blockers": item.blockers,
                "recommended_granularity": item.recommended_granularity,
            }
            for item in sorted(analysis.evidence.values(), key=lambda e: (e.task, e.tradition, e.context or "", e.fine))
        ],
        "family_isolation": {
            "violations": _dedupe_violations(
                violation
                for evaluation in analysis.evaluations.values()
                for violation in evaluation.isolation_violations
            )
        },
    }
    if analysis.evaluations:
        payload["models"] = {
            name: {
                "status": evaluation.status,
                "coverage": round(evaluation.coverage, 4),
                "total_cost": round(evaluation.total_cost, 4),
                "rejects": evaluation.rejects,
                "abstentions": evaluation.abstentions,
                "by_granularity": evaluation.by_granularity(),
                "by_scope": evaluation.by_scope(),
                "outcomes": [
                    {
                        "task": outcome.task,
                        "tradition": outcome.tradition,
                        "fine": outcome.fine,
                        "context": outcome.context,
                        "support": outcome.support,
                        "fine_correct": outcome.fine_correct,
                        "fine_wrong": outcome.fine_wrong,
                        "parent_correct": outcome.parent_correct,
                        "parent_wrong": outcome.parent_wrong,
                        "rejected": outcome.rejected,
                        "precision": outcome.precision,
                        "precision_lower": round(outcome.precision_lower, 4),
                    }
                    for outcome in evaluation.outcomes.values()
                ],
                "warnings": evaluation.warnings,
            }
            for name, evaluation in analysis.evaluations.items()
        }
        payload["selection"] = {
            "chosen": analysis.selection.chosen,
            "reason": analysis.selection.reason,
            "high_cost_guard": analysis.selection.high_cost_guard,
            "ranking": [
                {"model": name, "score": None if score == float("inf") else round(score, 4), "note": note}
                for name, score, note in analysis.selection.ranking
            ],
        }
    if analysis.migration:
        payload["migration"] = {
            "from": analysis.migration.from_version,
            "to": analysis.migration.to_version,
            "changes": [
                {"code": c.code, "change": c.change}
                for c in analysis.migration.changes
            ],
            "actions": [
                {
                    "course": a.course_id,
                    "chapter": a.chapter,
                    "clip": a.clip_id,
                    "old_code": a.old_code,
                    "new_code": a.new_code,
                    "action": a.action,
                    "reason": a.reason,
                    "severity": a.severity,
                    "eligible": a.eligible,
                    "blockers": list(a.blockers),
                }
                for a in analysis.migration.actions
            ],
        }
    payload["ungradable"] = [
        {
            "clip": gap.clip_id,
            "task": gap.task,
            "tradition": gap.tradition,
            "reason": gap.reason,
            "needed_skill": gap.needed_skill,
        }
        for gap in analysis.gaps
    ]
    if analysis.review_plan:
        payload["review_plan"] = {
            "period": analysis.review_plan.period,
            "total_minutes": analysis.review_plan.total_minutes,
            "assignments": [
                {
                    "clip": a.task.clip_id,
                    "reviewer": a.reviewer,
                    "minutes": a.minutes,
                    "reason": a.task.reason,
                    "risk_note": a.task.risk_note,
                }
                for a in analysis.review_plan.assignments
            ],
            "not_needed": [
                {
                    "clip": task.clip_id,
                    "reason": task.risk_note or task.reason,
                }
                for task in analysis.review_plan.not_needed
            ],
            "unassigned": [
                {
                    "clip": task.clip_id,
                    "tradition": task.tradition,
                    "skill": task.needed_skill,
                    "minutes": task.minutes,
                    "reason": task.reason,
                }
                for task in analysis.review_plan.unassigned
            ],
            "capability_needs": [asdict(need) for need in analysis.review_plan.capability_needs],
            "notes": analysis.review_plan.notes,
        }
    return payload


def _print_analysis(analysis) -> None:
    for line in render_summary(
        analysis.chosen_evaluation, analysis.selection, analysis.evidence, analysis.migration, analysis.review_plan
    ):
        print(line)
    print()
    if analysis.evaluations:
        for evaluation in analysis.evaluations.values():
            for line in render_evaluation(evaluation, analysis.dataset.policy):
                print(line)
        if analysis.selection:
            print()
            for line in render_selection(analysis.selection):
                print(line)
        print()
    for line in render_evidence(analysis.evidence):
        print(line)
    print()
    for line in render_gold(analysis.gold):
        print(line)
    if analysis.migration:
        print()
        for line in render_migration(analysis.migration):
            print(line)
    print()
    for line in render_ungradable(analysis.gaps):
        print(line)
    if analysis.review_plan:
        print()
        for line in render_review_plan(analysis.review_plan):
            print(line)
    if analysis.warnings:
        print()
        print("输入与规则提示：")
        for warning in dict.fromkeys(analysis.warnings):
            print(f"  - {warning}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="craftreview", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    analyze_cmd = sub.add_parser("analyze", help="运行完整分析")
    analyze_cmd.add_argument("--data", default="fixtures", help="输入目录")
    analyze_cmd.add_argument("--models", nargs="*", help="限定候选模型")
    analyze_cmd.add_argument("--from-version", dest="from_version")
    analyze_cmd.add_argument("--to-version", dest="to_version")
    analyze_cmd.add_argument("--period", default=None, help="复核期；默认取复核人名册中的最新期")
    analyze_cmd.add_argument("--no-model", action="store_true", help="只做证据与迁移，不训练模型")
    analyze_cmd.add_argument("--json", action="store_true", help="输出 JSON")

    catalog_cmd = sub.add_parser("catalog-diff", help="查询目录版本之间标签与解释适用范围的变化")
    catalog_cmd.add_argument("--data", default="fixtures")
    catalog_cmd.add_argument("--from-version", dest="from_version", required=True)
    catalog_cmd.add_argument("--to-version", dest="to_version", required=True)
    catalog_cmd.add_argument("--json", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "catalog-diff":
            dataset = Dataset.load(args.data)
            if not dataset.catalog:
                print("输入目录没有 catalog.csv", file=sys.stderr)
                return 2
            changes = catalog_scope_changes(dataset.catalog, args.from_version, args.to_version)
            if args.json:
                print(json.dumps([asdict(change) for change in changes], ensure_ascii=False, indent=2))
            else:
                for change in changes:
                    print(
                        f"{change.code}: 级别 {change.old_level}->{change.new_level}；"
                        f"批准 {change.old_approved}->{change.new_approved}；"
                        f"范围 {change.old_scope or '—'} -> {change.new_scope or '—'}"
                    )
                if not changes:
                    print("两个版本之间没有目录差异")
            return 0

        dataset = Dataset.load(args.data)
        analysis = analyze(
            dataset,
            models=args.models,
            from_version=getattr(args, "from_version", None),
            to_version=getattr(args, "to_version", None),
            period=getattr(args, "period", None),
            evaluate=not getattr(args, "no_model", False),
        )
        if getattr(args, "json", False):
            print(json.dumps(_analysis_payload(analysis), ensure_ascii=False, indent=2, default=_json_default))
        else:
            _print_analysis(analysis)
        return 0
    except CraftReviewError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
