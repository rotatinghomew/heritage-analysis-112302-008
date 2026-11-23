"""从数据目录装配完整分析。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .config import Policy
from .data import (
    CatalogEntry,
    Clip,
    CourseChapter,
    Finding,
    Observation,
    ReviewRecord,
    Reviewer,
    load_catalog,
    load_clips,
    load_courses,
    load_findings,
    load_observations,
    load_policies,
    load_reviewers,
    load_reviews,
)
from .eligibility import FineClassEvidence, GoldResolution, evaluate_evidence
from .families import FamilyResolution, resolve_families
from .migration import MigrationPlan, build_migration_plan, find_ungradable_courseware
from .models import (
    Evaluation,
    Selection,
    build_samples,
    evaluate_models,
    select_model,
    severities_from_courses,
)
from .review import ReviewPlan, build_review_plan


@dataclass
class Dataset:
    directory: Path
    clips: list[Clip]
    reviewers: list[Reviewer] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    reviews: list[ReviewRecord] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    catalog: list[CatalogEntry] = field(default_factory=list)
    courses: list[CourseChapter] = field(default_factory=list)
    policy: Policy = field(default_factory=Policy)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, directory: str | Path) -> "Dataset":
        directory = Path(directory)
        warnings: list[str] = []
        clips, w = load_clips(directory / "primary.csv")
        warnings += w
        reviewers, w = _optional(load_reviewers, directory / "support-1.csv", warnings, "复核人")
        policy_rows, w = _optional(load_policies, directory / "support-2.csv", warnings, "策略")
        warnings += w
        policy = Policy.from_rows(policy_rows, warnings) if policy_rows else Policy()
        observations, w = _optional(load_observations, directory / "observations.csv", warnings, "结构化观测")
        reviews, w = _optional(load_reviews, directory / "reviews.csv", warnings, "复核记录")
        findings, w = _optional(load_findings, directory / "findings.csv", warnings, "核对记录")
        catalog, w = _optional(load_catalog, directory / "catalog.csv", warnings, "目录")
        courses, w = _optional(load_courses, directory / "courses.csv", warnings, "课程章节")
        if not observations:
            warnings.append("没有结构化观测，无法训练模型；只能报告证据与迁移资格")
        return cls(
            directory=directory,
            clips=clips,
            reviewers=reviewers,
            observations=observations,
            reviews=reviews,
            findings=findings,
            catalog=catalog,
            courses=courses,
            policy=policy,
            warnings=warnings,
        )

    def resolve_families(self) -> FamilyResolution:
        return resolve_families(self.clips, self.findings)


def _optional(loader, path: Path, warnings: list[str], label: str):
    if not path.exists():
        return [], []
    items, extra = loader(path)
    warnings += extra
    return items, extra


@dataclass
class Analysis:
    dataset: Dataset
    resolution: FamilyResolution
    evaluations: dict[str, Evaluation]
    selection: Selection | None
    evidence: dict[tuple[str, str, str | None, str], FineClassEvidence]
    gold: dict[str, GoldResolution]
    migration: MigrationPlan | None
    review_plan: ReviewPlan | None
    gaps: list
    warnings: list[str] = field(default_factory=list)

    @property
    def chosen_evaluation(self) -> Evaluation | None:
        if self.selection and self.selection.chosen:
            return self.evaluations.get(self.selection.chosen)
        return None


def analyze(
    dataset: Dataset,
    *,
    models: Sequence[str] | None = None,
    from_version: str | None = None,
    to_version: str | None = None,
    period: str | None = None,
    evaluate: bool = True,
) -> Analysis:
    policy = dataset.policy
    resolution = dataset.resolve_families()
    warnings = list(dataset.warnings)
    severity = severities_from_courses(dataset.courses)
    evaluations: dict[str, Evaluation] = {}
    selection: Selection | None = None
    if evaluate and dataset.observations:
        samples = build_samples(dataset.clips, dataset.observations, severity)
        evaluations = evaluate_models(
            samples, resolution, policy, models or None, allowed_splits=("test", "holdout")
        )
        selection = select_model(evaluations, policy)
    elif evaluate:
        warnings.append("缺少结构化观测，跳过模型比较")

    chosen = evaluations.get(selection.chosen) if selection and selection.chosen else None
    if chosen is None and evaluations:
        # 没有可选的模型时，也要把隔离无效的状态传给证据层，避免把
        # “评估无效”误报成“没有评估”。
        chosen = next(iter(evaluations.values()))
    target_version = to_version or policy.active_version
    evidence, gold, evidence_warnings = evaluate_evidence(
        dataset.clips,
        dataset.reviews,
        dataset.catalog,
        chosen,
        resolution,
        policy,
        dataset.findings,
        catalog_version=target_version,
        reviewers={reviewer.reviewer_id: reviewer for reviewer in dataset.reviewers},
    )
    warnings += evidence_warnings

    migration: MigrationPlan | None = None
    gaps: list = []
    if dataset.catalog and dataset.courses:
        target = target_version
        source = from_version
        if source is None:
            versions = sorted({entry.version for entry in dataset.catalog})
            source = versions[0] if versions else None
        if target and source:
            migration = build_migration_plan(
                dataset.courses,
                dataset.catalog,
                evidence,
                dataset.clips,
                from_version=source,
                to_version=target,
            )
            gaps = find_ungradable_courseware(dataset.courses, dataset.clips, evidence)
        else:
            warnings.append("无法确定目录来源/目标版本，跳过迁移")

    review_plan: ReviewPlan | None = None
    if dataset.reviewers:
        resolved_period = period
        if resolved_period is None:
            periods = sorted({reviewer.period for reviewer in dataset.reviewers if reviewer.period})
            resolved_period = periods[-1] if periods else "next"
        review_plan = build_review_plan(
            dataset.clips,
            dataset.reviewers,
            evidence,
            dataset.reviews,
            dataset.courses,
            policy,
            period=resolved_period,
            resolution=resolution,
            findings=dataset.findings,
        )
    return Analysis(
        dataset=dataset,
        resolution=resolution,
        evaluations=evaluations,
        selection=selection,
        evidence=evidence,
        gold=gold,
        migration=migration,
        review_plan=review_plan,
        gaps=gaps,
        warnings=warnings,
    )
