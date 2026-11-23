"""片源家族与遗传关联。

同一表演的剪辑不能被拆到训练与检验两侧。双评核对若判定两个片段同源，
这里把它们的家族合并成一个证据组；合并后独立家族数下降，依赖“两个家族”
的资格会随之失效。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from .data import Clip, Finding


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, name: str) -> None:
        self._parent.setdefault(name, name)

    def find(self, name: str) -> str:
        self.add(name)
        root = name
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[name] != root:
            self._parent[name], name = root, self._parent[name]
        return root

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        # 稳定根，便于输出可复现
        if rb < ra:
            ra, rb = rb, ra
        self._parent[rb] = ra
        return True

    def groups(self) -> dict[str, tuple[str, ...]]:
        buckets: dict[str, list[str]] = {}
        for name in self._parent:
            buckets.setdefault(self.find(name), []).append(name)
        return {root: tuple(sorted(members)) for root, members in buckets.items()}


@dataclass(frozen=True)
class FamilyResolution:
    """家族到证据组的映射及合并来源。"""

    group_of: Mapping[str, tuple[str, ...]]
    merged_by: Mapping[tuple[str, ...], tuple[str, ...]] = field(default_factory=dict)

    def group(self, family: str) -> tuple[str, ...]:
        return self.group_of.get(family, (family,))

    def group_key(self, family: str) -> str:
        return "+".join(self.group(family))

    def independent_count(self, families: Iterable[str]) -> int:
        return len({self.group_key(family) for family in families})

    def split_violations(self, clips: Iterable[Clip]) -> list[dict[str, object]]:
        """返回同一证据组跨多个 split 的违例。"""
        by_group: dict[str, dict[str, list[str]]] = {}
        for clip in clips:
            key = self.group_key(clip.family)
            by_group.setdefault(key, {}).setdefault(clip.split, []).append(clip.clip_id)
        violations: list[dict[str, object]] = []
        for key, splits in sorted(by_group.items()):
            if len(splits) > 1:
                violations.append({"group": key, "splits": splits})
        return violations


def resolve_families(clips: Iterable[Clip], findings: Iterable[Finding]) -> FamilyResolution:
    clips = list(clips)
    uf = _UnionFind()
    family_of_clip: dict[str, str] = {}
    for clip in clips:
        uf.add(clip.family)
        family_of_clip[clip.clip_id] = clip.family
    merged_by: dict[tuple[str, ...], tuple[str, ...]] = {}
    # Finding 中的 clip 来自输入表，family 可能大小写不一致；统一按大小写
    # 不敏感匹配，避免同一家族被当成两个。
    family_index = {family.lower(): family for family in family_of_clip.values()}
    for finding in findings:
        if not finding.is_same_performance:
            continue
        families = tuple(
            sorted({family_index[family_of_clip[c].lower()] for c in finding.clips if c in family_of_clip})
        )
        if len(families) < 2:
            continue
        base = families[0]
        for other in families[1:]:
            uf.union(base, other)
        merged_by[finding.clips] = families
    roots = uf.groups()
    group_of = {member: roots[root] for root, members in roots.items() for member in members}
    return FamilyResolution(group_of=group_of, merged_by=merged_by)
