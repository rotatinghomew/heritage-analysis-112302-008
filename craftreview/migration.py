"""目录版本差异与说明迁移。

迁移只使用已有证据：细类资格成立才换到新细类；否则保留父类说明。课程
章节若引用的正是仍然正确的广义说明，不算待改；只有适用范围或用词变化才
生成迁移动作。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .data import CatalogEntry, CourseChapter
from .eligibility import FineClassEvidence


@dataclass(frozen=True)
class CatalogChange:
    code: str
    change: str  # added / removed / renamed / scope_changed / approval_changed
    old: CatalogEntry | None
    new: CatalogEntry | None


@dataclass
class MigrationAction:
    course_id: str
    chapter: str
    clip_id: str | None
    old_code: str | None
    new_code: str | None
    action: str  # update_explanation / keep_parent / keep_as_is / blocked
    reason: str
    severity: str = "medium"
    eligible: bool = True
    blockers: tuple[str, ...] = ()


@dataclass
class MigrationPlan:
    from_version: str
    to_version: str
    changes: list[CatalogChange] = field(default_factory=list)
    actions: list[MigrationAction] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def pending_updates(self) -> list[MigrationAction]:
        return [a for a in self.actions if a.action == "update_explanation"]

    def keep_parent(self) -> list[MigrationAction]:
        return [a for a in self.actions if a.action in {"keep_parent", "keep_as_is"}]

    def blocked(self) -> list[MigrationAction]:
        return [a for a in self.actions if a.action == "blocked"]

    def invalidated(self) -> list[MigrationAction]:
        return [a for a in self.actions if not a.eligible]


def _index_version(catalog: Iterable[CatalogEntry], version: str) -> dict[str, CatalogEntry]:
    result: dict[str, CatalogEntry] = {}
    for entry in catalog:
        if entry.version == version:
            result[entry.code] = entry
    return result


def diff_versions(
    catalog: Sequence[CatalogEntry], from_version: str, to_version: str
) -> tuple[list[CatalogChange], list[str]]:
    """比较两个目录版本的标签、父子关系、适用范围与批准状态。"""
    warnings: list[str] = []
    old = _index_version(catalog, from_version)
    new = _index_version(catalog, to_version)
    if not old:
        warnings.append(f"来源版本 {from_version} 在目录中不存在")
    if not new:
        warnings.append(f"目标版本 {to_version} 在目录中不存在")
    changes: list[CatalogChange] = []
    for code in sorted(set(old) | set(new)):
        before, after = old.get(code), new.get(code)
        if before is None and after is not None:
            changes.append(CatalogChange(code, "added", None, after))
        elif before is not None and after is None:
            changes.append(CatalogChange(code, "removed", before, None))
        elif before is not None and after is not None:
            if (before.label, before.parent) != (after.label, after.parent):
                changes.append(CatalogChange(code, "renamed", before, after))
            elif (before.scope_notes or "") != (after.scope_notes or ""):
                changes.append(CatalogChange(code, "scope_changed", before, after))
            elif before.approved != after.approved or before.max_granularity != after.max_granularity:
                changes.append(CatalogChange(code, "approval_changed", before, after))
    return changes, warnings


def _scope_holds(entry: CatalogEntry | None, task: str | None, context: str | None) -> bool:
    if entry is None or not entry.scope_notes:
        return True
    for chunk in entry.scope_notes.replace("；", ";").split(";"):
        if ":" not in chunk:
            continue
        dim, values = chunk.split(":", 1)
        dim = dim.strip().lower()
        options = {v.strip().lower() for v in values.replace("|", ",").split(",") if v.strip()}
        if not options:
            continue
        if dim in {"task", "工序"} and (task or "").lower() not in options:
            return False
        if dim in {"context", "器具"} and (context or "").lower() not in options:
            return False
    return True


def _can_stay_coarse(entry: CatalogEntry) -> bool:
    """新版目录是否仍允许只使用父类说明。"""
    if entry.level == "coarse":
        return entry.approved
    return entry.approved and entry.max_granularity in {"coarse", "fine"}


def _online_for_chapter(
    chapter: CourseChapter,
    clip,
    evidence: Mapping[tuple[str, str, str | None, str], FineClassEvidence],
    by_parent: Mapping[str, list[FineClassEvidence]],
) -> FineClassEvidence | None:
    """找出该章节可以升级到的已上线细类；没有则返回 None。"""
    if clip is not None and clip.fine:
        item = evidence.get((clip.task, clip.tradition, clip.context, clip.fine))
        if item is not None and item.status == "online":
            return item
        return None
    if chapter.catalog_code in by_parent:
        online = [item for item in by_parent[chapter.catalog_code] if item.status == "online"]
        return online[0] if len(online) == 1 else None
    if clip is not None:
        online = [
            item
            for item in evidence.values()
            if item.task == clip.task
            and item.tradition == clip.tradition
            and item.status == "online"
        ]
        return online[0] if len(online) == 1 else None
    return None


def build_migration_plan(
    chapters: Sequence[CourseChapter],
    catalog: Sequence[CatalogEntry],
    evidence: Mapping[tuple[str, str, str | None, str], FineClassEvidence],
    clips=Sequence,
    *,
    from_version: str,
    to_version: str,
) -> MigrationPlan:
    """生成课程章节的迁移动作。

    已定版片源只有在细类资格有效且目录批准时才改说明；否则继续引用父类。
    """
    plan = MigrationPlan(from_version=from_version, to_version=to_version)
    changes, warnings = diff_versions(catalog, from_version, to_version)
    plan.changes = changes
    plan.warnings.extend(warnings)

    clip_index = {clip.clip_id: clip for clip in clips} if clips else {}
    new_catalog = _index_version(catalog, to_version)
    old_catalog = _index_version(catalog, from_version)
    by_parent: dict[str, list[FineClassEvidence]] = defaultdict(list)
    for item in evidence.values():
        if item.coarse:
            by_parent[item.coarse].append(item)

    for chapter in chapters:
        if chapter.status not in {"published", "active", "final"}:
            plan.actions.append(
                MigrationAction(
                    chapter.course_id,
                    chapter.chapter,
                    chapter.clip_id,
                    chapter.catalog_code,
                    chapter.catalog_code,
                    "blocked",
                    f"章节状态为 {chapter.status}，不在已定版迁移范围",
                    chapter.severity,
                    eligible=False,
                    blockers=("chapter_not_final",),
                )
            )
            continue
        clip = clip_index.get(chapter.clip_id) if chapter.clip_id else None
        task = clip.task if clip else None
        context = clip.context if clip else None
        tradition = clip.tradition if clip else None

        # 当前引用本身在新版里是否仍有效
        if chapter.catalog_code:
            new_entry = new_catalog.get(chapter.catalog_code)
            if new_entry is None:
                plan.actions.append(
                    MigrationAction(
                        chapter.course_id,
                        chapter.chapter,
                        chapter.clip_id,
                        chapter.catalog_code,
                        None,
                        "update_explanation",
                        "该代码在新版目录中已移除，需改写说明",
                        chapter.severity,
                    )
                )
                continue
            if (
                _scope_holds(new_entry, task, context)
                and chapter.explanation
                and _can_stay_coarse(new_entry)
            ):
                # 已适用且仍合规的父类说明：默认继续引用。但若该片段的细类
                # 已经具备上线资格，则属于“可以升级”，仍需列出待换说明。
                upgrade = _online_for_chapter(chapter, clip, evidence, by_parent)
                if upgrade is None:
                    stay_blockers: tuple[str, ...] = ()
                    reason = "新版仍允许该父类说明，可继续引用"
                    if clip is not None and clip.fine:
                        item = evidence.get(
                            (clip.task, clip.tradition, clip.context, clip.fine)
                        )
                        if item is not None and item.status in {"unresolved", "invalidated"}:
                            stay_blockers = tuple(item.blockers)
                            reason = (
                                "新版仍允许该父类说明；但细类结论"
                                + ("双评不一致" if item.status == "unresolved" else "已被同源核对撤销")
                                + "，只能继续引用父类并安排复审"
                            )
                    plan.actions.append(
                        MigrationAction(
                            chapter.course_id,
                            chapter.chapter,
                            chapter.clip_id,
                            chapter.catalog_code,
                            chapter.catalog_code,
                            "keep_parent",
                            reason,
                            chapter.severity,
                            blockers=stay_blockers,
                        )
                    )
                    continue
            elif new_entry.level == "coarse" and not _scope_holds(
                new_entry, task, context
            ):
                plan.actions.append(
                    MigrationAction(
                        chapter.course_id,
                        chapter.chapter,
                        chapter.clip_id,
                        chapter.catalog_code,
                        chapter.catalog_code,
                        "update_explanation",
                        "父类代码仍存在，但新版适用范围不再覆盖该片段的工序/器具，需改写说明",
                        chapter.severity,
                        blockers=("scope_mismatch",),
                    )
                )
                continue

        # 找到能升级的细类
        candidates: list[FineClassEvidence] = []
        if clip and clip.fine:
            item = evidence.get((clip.task, clip.tradition, clip.context, clip.fine))
            if item:
                candidates = [item]
        elif chapter.catalog_code in by_parent:
            candidates = [
                item for item in by_parent[chapter.catalog_code] if item.status == "online"
            ]
        elif task and tradition:
            candidates = [
                item
                for item in evidence.values()
                if item.task == task and item.tradition == tradition and item.status == "online"
            ]

        if not candidates:
            blockers = tuple(
                sorted(
                    {
                        blocker
                        for item in evidence.values()
                        if item.task == task and item.tradition == tradition
                        for blocker in item.blockers
                    }
                )
            )
            target = chapter.catalog_code
            action = "keep_parent" if target else "blocked"
            reason = "没有合格细类证据，继续使用上位概念"
            if not target:
                reason = "章节未绑定父类，且无合格细类，无法自动迁移"
            plan.actions.append(
                MigrationAction(
                    chapter.course_id,
                    chapter.chapter,
                    chapter.clip_id,
                    chapter.catalog_code,
                    target,
                    action,
                    reason,
                    chapter.severity,
                    eligible=bool(target),
                    blockers=blockers or ("no_evidence",),
                )
            )
            continue

        online = [item for item in candidates if item.status == "online"]
        invalidated = [item for item in candidates if item.status == "invalidated"]
        if invalidated and not online:
            plan.actions.append(
                MigrationAction(
                    chapter.course_id,
                    chapter.chapter,
                    chapter.clip_id,
                    chapter.catalog_code,
                    chapter.catalog_code,
                    "blocked",
                    "细类资格因同一表演核对而失效，需重新复核后再迁移",
                    chapter.severity,
                    eligible=False,
                    blockers=("invalidated_by_finding",),
                )
            )
            continue
        if not online:
            plan.actions.append(
                MigrationAction(
                    chapter.course_id,
                    chapter.chapter,
                    chapter.clip_id,
                    chapter.catalog_code,
                    chapter.catalog_code,
                    "keep_parent",
                    "细类证据未达上线条件，继续使用父类",
                    chapter.severity,
                    blockers=candidates[0].blockers,
                )
            )
            continue

        item = online[0]
        new_entry = new_catalog.get(item.fine)
        if new_entry is None or not new_entry.approved or not _scope_holds(new_entry, task, context):
            plan.actions.append(
                MigrationAction(
                    chapter.course_id,
                    chapter.chapter,
                    chapter.clip_id,
                    chapter.catalog_code,
                    chapter.catalog_code,
                    "keep_parent",
                    "细类证据成立，但新版目录未批准或适用范围不符，暂不改说明",
                    chapter.severity,
                    blockers=("catalog_not_approved",),
                )
            )
            continue

        old_entry = old_catalog.get(chapter.catalog_code) if chapter.catalog_code else None
        scope_changed = old_entry is not None and (old_entry.scope_notes or "") != (new_entry.scope_notes or "")
        plan.actions.append(
            MigrationAction(
                chapter.course_id,
                chapter.chapter,
                chapter.clip_id,
                chapter.catalog_code,
                item.fine,
                "update_explanation",
                (
                    f"细类资格成立（{item.family_count} 个独立家族，精度下界 "
                    f"{item.precision_lower:.2f}），更换为新版细类说明"
                    + ("；适用范围有变化" if scope_changed else "")
                ),
                chapter.severity,
            )
        )
    return plan


@dataclass(frozen=True)
class CoursewareGap:
    clip_id: str
    task: str
    tradition: str
    coarse: str | None
    reason: str
    blockers: tuple[str, ...]
    needed_skill: str
    suggested_tradition: str


def find_ungradable_courseware(
    chapters: Sequence[CourseChapter],
    clips: Sequence,
    evidence: Mapping[tuple[str, str, str | None, str], FineClassEvidence],
) -> list[CoursewareGap]:
    """列出仍无法升级为细类的课件，作为下一期复核需求。"""
    clip_index = {clip.clip_id: clip for clip in clips}
    gaps: dict[str, CoursewareGap] = {}
    for chapter in chapters:
        if not chapter.clip_id or chapter.clip_id not in clip_index:
            continue
        clip = clip_index[chapter.clip_id]
        key = (clip.task, clip.tradition, clip.context, clip.fine) if clip.fine else None
        item = evidence.get(key) if key else None
        if item is not None and item.status == "online":
            continue
        blockers = tuple(item.blockers) if item else ("no_fine_label",)
        gaps[clip.clip_id] = CoursewareGap(
            clip_id=clip.clip_id,
            task=clip.task,
            tradition=clip.tradition,
            coarse=clip.coarse,
            reason=item.explain() if item else "片段缺少细类标注，只能继续使用父类",
            blockers=blockers,
            needed_skill=clip.fine or clip.coarse or clip.task,
            suggested_tradition=clip.tradition,
        )
    return sorted(gaps.values(), key=lambda gap: (gap.tradition, gap.task, gap.clip_id))
