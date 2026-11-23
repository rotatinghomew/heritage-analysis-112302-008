"""实体与 CSV 读取。

各表只按标识关联；同一标识在多张表中出现不构成第二份独立观察。所有
读取函数都保留原始值并报告警告，避免把未知、零和缺报混为一谈。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from .errors import InputError

VALID_SPLITS = {"train", "valid", "test", "holdout"}
VALID_SEVERITIES = {"high", "medium", "low"}


def _text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def _read_rows(path: Path, required: Sequence[str]) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise InputError(f"找不到输入文件：{path}")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise InputError(f"{path} 缺少表头")
        header = [name.strip() for name in reader.fieldnames]
        if len(header) != len(set(header)):
            raise InputError(f"{path} 表头有重复字段")
        missing = [name for name in required if name not in header]
        if missing:
            raise InputError(f"{path} 缺少字段：{', '.join(missing)}")
        rows: list[dict[str, str]] = []
        warnings: list[str] = []
        for line_no, raw in enumerate(reader, start=2):
            row = {(k or "").strip(): (v or "") for k, v in raw.items()}
            if not any(value.strip() for value in row.values()):
                warnings.append(f"{path.name}:{line_no} 空行已跳过")
                continue
            rows.append(row)
    return rows, warnings


@dataclass(frozen=True)
class Clip:
    clip_id: str
    family: str
    tradition: str
    task: str
    context: str | None
    split: str
    coarse: str | None
    fine: str | None
    reviewer: str | None
    version: str
    line_no: int = 0

    @property
    def key(self) -> tuple[str, str, str | None]:
        """细类只在其器具与工序上下文中可区分。"""
        return (self.task, self.context, self.fine)


@dataclass(frozen=True)
class Reviewer:
    reviewer_id: str
    tradition: str
    skill: str
    minutes: int
    period: str

    def eligible_for(self, clip: Clip) -> bool:
        """复核人须同时覆盖该片段的传承支系与工序/类别。"""
        if clip.tradition != self.tradition:
            return False
        if self.skill not in {clip.task, clip.coarse, clip.fine}:
            return False
        return True

    def fine_eligible_for(self, clip: Clip) -> bool:
        """细类佐证需要该细类本身的复核资质。

        只具备工序或父类资质的人可以作父类结论，但不能把自己的判断当作
        细类双评的一半。
        """
        return (
            clip.tradition == self.tradition
            and bool(clip.fine)
            and self.skill == clip.fine
        )


@dataclass(frozen=True)
class Observation:
    """结构化观测，仅使用非视觉字段。"""

    clip_id: str
    task: str | None = None
    context: str | None = None
    family: str | None = None
    tradition: str | None = None
    duration_ms: float | None = None
    landmark_count: int | None = None
    signals: Mapping[str, float] = field(default_factory=dict)

    def vector(self) -> dict[str, float]:
        values = dict(self.signals)
        if self.duration_ms is not None:
            values["duration_ms"] = self.duration_ms
        if self.landmark_count is not None:
            values["landmark_count"] = float(self.landmark_count)
        return values


@dataclass(frozen=True)
class ReviewRecord:
    review_id: str
    clip_id: str
    reviewer: str
    tradition: str
    coarse: str | None
    fine: str | None
    verdict: str
    evidence_key: str | None
    reviewed_at: str
    note: str | None = None

    @property
    def agrees_on_coarse_only(self) -> bool:
        return self.verdict == "coarse" or (bool(self.coarse) and not self.fine)


@dataclass(frozen=True)
class Finding:
    """双评核对发现：若干片段来自同一表演。"""

    finding_id: str
    clips: tuple[str, ...]
    finding: str
    reviewer: str
    decided_at: str
    invalidates: tuple[str, ...] = ()

    @property
    def is_same_performance(self) -> bool:
        normalized = self.finding.strip().lower()
        return normalized in {"same_performance", "same", "同一表演", "同源"}


@dataclass(frozen=True)
class CatalogEntry:
    code: str
    version: str
    level: str
    parent: str | None
    label: str
    scope_notes: str | None
    approved: bool
    max_granularity: str = "fine"


@dataclass(frozen=True)
class CourseChapter:
    course_id: str
    chapter: str
    clip_id: str | None
    catalog_code: str | None
    explanation: str | None
    severity: str
    status: str
    valid_from: str
    valid_to: str | None = None


def load_clips(path: str | Path) -> tuple[list[Clip], list[str]]:
    path = Path(path)
    rows, warnings = _read_rows(
        path,
        ["clip", "family", "tradition", "task", "context", "split", "coarse", "fine", "reviewer", "version"],
    )
    clips: list[Clip] = []
    seen: set[str] = set()
    for line_no, row in enumerate(rows, start=2):
        clip_id = _text(row.get("clip"))
        if not clip_id:
            warnings.append(f"{path.name}:{line_no} 缺少 clip，已跳过")
            continue
        if clip_id in seen:
            raise InputError(f"{path.name}:{line_no} 片段标识重复：{clip_id}")
        seen.add(clip_id)
        split = (_text(row.get("split")) or "").lower()
        if split not in VALID_SPLITS:
            raise InputError(f"{path.name}:{line_no} split 非法：{split!r}")
        family = _text(row.get("family"))
        tradition = _text(row.get("tradition"))
        task = _text(row.get("task"))
        version = _text(row.get("version"))
        if not family:
            raise InputError(f"{path.name}:{line_no} 缺少 family，无法隔离")
        if not tradition or not task:
            raise InputError(f"{path.name}:{line_no} 缺少 tradition 或 task")
        if not version:
            warnings.append(f"{path.name}:{line_no} 缺少 version，按 unknown 处理")
            version = "unknown"
        clips.append(
            Clip(
                clip_id=clip_id,
                family=family.lower(),
                tradition=tradition.lower(),
                task=task.lower(),
                context=_text(row.get("context")),
                split=split,
                coarse=(_text(row.get("coarse")) or "").lower() or None,
                fine=(_text(row.get("fine")) or "").lower() or None,
                reviewer=_text(row.get("reviewer")),
                version=version,
                line_no=line_no,
            )
        )
    if not clips:
        raise InputError(f"{path.name} 没有可用片段")
    return clips, warnings


def load_reviewers(path: str | Path) -> tuple[list[Reviewer], list[str]]:
    path = Path(path)
    rows, warnings = _read_rows(path, ["reviewer", "tradition", "skill", "minutes", "period"])
    reviewers: list[Reviewer] = []
    for line_no, row in enumerate(rows, start=2):
        reviewer_id = _text(row.get("reviewer"))
        tradition = _text(row.get("tradition"))
        skill = _text(row.get("skill"))
        if not reviewer_id or not tradition or not skill:
            warnings.append(f"{path.name}:{line_no} 复核人缺少标识、支系或技能，已跳过")
            continue
        try:
            minutes = int(_text(row.get("minutes")) or "")
        except ValueError as exc:
            raise InputError(f"{path.name}:{line_no} minutes 不是整数") from exc
        if minutes <= 0:
            raise InputError(f"{path.name}:{line_no} minutes 必须为正")
        reviewers.append(
            Reviewer(
                reviewer_id=reviewer_id,
                tradition=tradition.lower(),
                skill=skill.lower(),
                minutes=minutes,
                period=_text(row.get("period")) or "unspecified",
            )
        )
    return reviewers, warnings


def load_observations(path: str | Path) -> tuple[list[Observation], list[str]]:
    path = Path(path)
    if not path.is_file():
        raise InputError(f"找不到观测文件：{path}")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise InputError(f"{path} 缺少表头")
        header = [name.strip() for name in reader.fieldnames if name]
        known = {"clip", "task", "context", "family", "tradition", "duration_ms", "landmark_count"}
        signal_names = [name for name in header if name not in known]
        rows = list(reader)
    observations: list[Observation] = []
    warnings: list[str] = []
    for line_no, raw in enumerate(rows, start=2):
        row = {(k or "").strip(): (v or "").strip() for k, v in raw.items()}
        clip_id = row.get("clip")
        if not clip_id:
            warnings.append(f"{path.name}:{line_no} 缺少 clip，已跳过")
            continue
        signals: dict[str, float] = {}
        for name in signal_names:
            text = row.get(name, "")
            if text == "":
                continue
            try:
                signals[name] = float(text)
            except ValueError:
                warnings.append(
                    f"{path.name}:{line_no} 信号 {name}={text!r} 非数值，视为缺报"
                )
        try:
            duration = float(row["duration_ms"]) if row.get("duration_ms") else None
        except ValueError:
            warnings.append(f"{path.name}:{line_no} duration_ms 非数值，视为缺报")
            duration = None
        try:
            landmarks = int(row["landmark_count"]) if row.get("landmark_count") else None
        except ValueError:
            warnings.append(f"{path.name}:{line_no} landmark_count 非数值，视为缺报")
            landmarks = None
        observations.append(
            Observation(
                clip_id=clip_id,
                task=_text(row.get("task")),
                context=_text(row.get("context")),
                family=_text(row.get("family")),
                tradition=_text(row.get("tradition")),
                duration_ms=duration,
                landmark_count=landmarks,
                signals=signals,
            )
        )
    return observations, warnings


def load_reviews(path: str | Path) -> tuple[list[ReviewRecord], list[str]]:
    path = Path(path)
    rows, warnings = _read_rows(
        path, ["review", "clip", "reviewer", "tradition", "coarse", "fine", "verdict", "evidence_key", "reviewed_at"]
    )
    reviews: list[ReviewRecord] = []
    for line_no, row in enumerate(rows, start=2):
        review_id = _text(row.get("review"))
        clip_id = _text(row.get("clip"))
        reviewer = _text(row.get("reviewer"))
        if not review_id or not clip_id or not reviewer:
            warnings.append(f"{path.name}:{line_no} 复核记录缺少标识，已跳过")
            continue
        verdict = (_text(row.get("verdict")) or "fine").lower()
        if verdict not in {"fine", "coarse", "reject", "abstain"}:
            warnings.append(f"{path.name}:{line_no} verdict={verdict!r} 未识别，按 abstain")
            verdict = "abstain"
        reviews.append(
            ReviewRecord(
                review_id=review_id,
                clip_id=clip_id,
                reviewer=reviewer,
                tradition=(_text(row.get("tradition")) or "").lower(),
                coarse=(_text(row.get("coarse")) or "").lower() or None,
                fine=(_text(row.get("fine")) or "").lower() or None,
                verdict=verdict,
                evidence_key=_text(row.get("evidence_key")),
                reviewed_at=_text(row.get("reviewed_at")) or "unspecified",
                note=_text(row.get("note")),
            )
        )
    return reviews, warnings


def load_findings(path: str | Path) -> tuple[list[Finding], list[str]]:
    path = Path(path)
    if not path.exists():
        return [], []
    rows, warnings = _read_rows(
        path, ["finding", "clips", "verdict", "reviewer", "decided_at"]
    )
    findings: list[Finding] = []
    for line_no, row in enumerate(rows, start=2):
        finding_id = _text(row.get("finding"))
        clips_raw = row.get("clips") or ""
        clips = tuple(part.strip() for part in clips_raw.replace(";", "|").split("|") if part.strip())
        if not finding_id or len(clips) < 2:
            warnings.append(f"{path.name}:{line_no} 核对记录缺少标识或片段不足两个，已忽略")
            continue
        invalidates = tuple(
            part.strip()
            for part in (row.get("invalidates") or "").replace(";", "|").split("|")
            if part.strip()
        )
        findings.append(
            Finding(
                finding_id=finding_id,
                clips=clips,
                finding=(_text(row.get("verdict")) or "").lower(),
                reviewer=_text(row.get("reviewer")) or "unspecified",
                decided_at=_text(row.get("decided_at")) or "unspecified",
                invalidates=invalidates,
            )
        )
    return findings, warnings


def load_policies(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    path = Path(path)
    rows, warnings = _read_rows(path, ["policy", "item", "rule"])
    return rows, warnings


def load_catalog(path: str | Path) -> tuple[list[CatalogEntry], list[str]]:
    path = Path(path)
    if not path.exists():
        return [], []
    rows, warnings = _read_rows(
        path, ["code", "version", "level", "parent", "label", "scope_notes", "approved"]
    )
    entries: list[CatalogEntry] = []
    for line_no, row in enumerate(rows, start=2):
        code = _text(row.get("code"))
        version = _text(row.get("version"))
        if not code or not version:
            warnings.append(f"{path.name}:{line_no} 目录项缺少 code 或 version，已跳过")
            continue
        level = (_text(row.get("level")) or "fine").lower()
        if level not in {"coarse", "fine"}:
            raise InputError(f"{path.name}:{line_no} level 非法：{level}")
        approved = (_text(row.get("approved")) or "no").lower() in {"1", "true", "yes", "y", "是", "approved"}
        max_granularity = (_text(row.get("max_granularity")) or "fine").lower()
        if max_granularity not in {"coarse", "fine"}:
            raise InputError(f"{path.name}:{line_no} max_granularity 非法：{max_granularity}")
        entries.append(
            CatalogEntry(
                code=code,
                version=version,
                level=level,
                parent=_text(row.get("parent")),
                label=_text(row.get("label")) or code,
                scope_notes=_text(row.get("scope_notes")),
                approved=approved,
                max_granularity=max_granularity,
            )
        )
    return entries, warnings


def load_courses(path: str | Path) -> tuple[list[CourseChapter], list[str]]:
    path = Path(path)
    if not path.exists():
        return [], []
    rows, warnings = _read_rows(
        path, ["course", "chapter", "clip", "catalog_code", "explanation", "severity", "status", "valid_from"]
    )
    chapters: list[CourseChapter] = []
    for line_no, row in enumerate(rows, start=2):
        course_id = _text(row.get("course"))
        chapter = _text(row.get("chapter"))
        if not course_id or not chapter:
            warnings.append(f"{path.name}:{line_no} 课程章节缺少标识，已跳过")
            continue
        severity = (_text(row.get("severity")) or "medium").lower()
        if severity not in VALID_SEVERITIES:
            raise InputError(f"{path.name}:{line_no} severity 非法：{severity}")
        chapters.append(
            CourseChapter(
                course_id=course_id,
                chapter=chapter,
                clip_id=_text(row.get("clip")),
                catalog_code=_text(row.get("catalog_code")),
                explanation=_text(row.get("explanation")),
                severity=severity,
                status=(_text(row.get("status")) or "published").lower(),
                valid_from=_text(row.get("valid_from")) or "unspecified",
                valid_to=_text(row.get("valid_to")),
            )
        )
    return chapters, warnings
