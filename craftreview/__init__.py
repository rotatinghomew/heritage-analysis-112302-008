"""工序影像分类复核工具。

该包把标注手册中的硬性规则实现为可执行检查：

* 按片源家族隔离训练、校准与检验；
* 细类上线需要两个独立家族样本且精度下界达到阈值；
* 只有父类一致的双评不得用于细类验证；
* 复核发现同一表演的片段会合并证据家族并触发迁移资格失效；
* 选型使用课程用途成本，而不是总体准确率。
"""

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
from .eligibility import FineClassEvidence, evaluate_evidence
from .migration import MigrationPlan, build_migration_plan, diff_versions
from .models import MODEL_REGISTRY, evaluate_models, select_model
from .review import ReviewPlan, build_review_plan
from .stats import wilson_lower_bound

__all__ = [
    "Policy",
    "Clip",
    "Reviewer",
    "Observation",
    "ReviewRecord",
    "Finding",
    "CatalogEntry",
    "CourseChapter",
    "FineClassEvidence",
    "evaluate_evidence",
    "MODEL_REGISTRY",
    "evaluate_models",
    "select_model",
    "ReviewPlan",
    "build_review_plan",
    "MigrationPlan",
    "build_migration_plan",
    "diff_versions",
    "wilson_lower_bound",
    "load_clips",
    "load_reviewers",
    "load_policies",
    "load_observations",
    "load_reviews",
    "load_findings",
    "load_catalog",
    "load_courses",
]
