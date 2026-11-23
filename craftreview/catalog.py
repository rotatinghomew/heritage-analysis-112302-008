"""目录与版本查询。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from .data import CatalogEntry
from .eligibility import FineClassEvidence
from .migration import diff_versions


@dataclass(frozen=True)
class ScopeChange:
    code: str
    old_scope: str | None
    new_scope: str | None
    old_approved: bool
    new_approved: bool
    old_level: str | None
    new_level: str | None


def catalog_scope_changes(
    catalog: Sequence[CatalogEntry], from_version: str, to_version: str
) -> list[ScopeChange]:
    changes, _ = diff_versions(catalog, from_version, to_version)
    result: list[ScopeChange] = []
    for change in changes:
        result.append(
            ScopeChange(
                code=change.code,
                old_scope=change.old.scope_notes if change.old else None,
                new_scope=change.new.scope_notes if change.new else None,
                old_approved=bool(change.old and change.old.approved),
                new_approved=bool(change.new and change.new.approved),
                old_level=change.old.level if change.old else None,
                new_level=change.new.level if change.new else None,
            )
        )
    return result


def evidence_by_scope(
    evidence: Iterable[FineClassEvidence],
) -> dict[tuple[str, str, str | None], list[FineClassEvidence]]:
    buckets: dict[tuple[str, str, str | None], list[FineClassEvidence]] = defaultdict(list)
    for item in evidence:
        buckets[(item.tradition, item.task, item.context)].append(item)
    for items in buckets.values():
        items.sort(key=lambda e: (e.fine, e.status))
    return dict(buckets)
