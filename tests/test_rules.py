import unittest
from pathlib import Path

from craftreview.config import Policy
from craftreview.data import (
    CatalogEntry,
    Clip,
    CourseChapter,
    Finding,
    ReviewRecord,
    Reviewer,
)
from craftreview.eligibility import evaluate_evidence, resolve_gold
from craftreview.errors import PolicyError
from craftreview.families import resolve_families
from craftreview.migration import build_migration_plan, diff_versions
from craftreview.models import (
    Evaluation,
    ClassOutcome,
    build_samples,
    evaluate_models,
    select_model,
)
from craftreview.data import Observation
from craftreview.review import build_review_plan
from craftreview.stats import wilson_lower_bound


def clip(clip_id, family, split, fine=None, coarse="joinery", tradition="east", task="join", context="oak", version="v1"):
    return Clip(
        clip_id=clip_id,
        family=family,
        tradition=tradition,
        task=task,
        context=context,
        split=split,
        coarse=coarse,
        fine=fine,
        reviewer=None,
        version=version,
    )


def review(review_id, clip_id, reviewer, fine, verdict="fine", tradition="east", coarse="joinery", reviewed_at="2026-07-01"):
    return ReviewRecord(
        review_id=review_id,
        clip_id=clip_id,
        reviewer=reviewer,
        tradition=tradition,
        coarse=coarse,
        fine=fine,
        verdict=verdict,
        evidence_key=None,
        reviewed_at=reviewed_at,
    )


def catalog_entry(code, version="v2", level="fine", parent="joinery", approved=True, scope="task:join; context:oak", max_gran="fine"):
    return CatalogEntry(
        code=code,
        version=version,
        level=level,
        parent=parent,
        label=code,
        scope_notes=scope,
        approved=approved,
        max_granularity=max_gran,
    )


class PolicyTests(unittest.TestCase):
    def test_manual_floor_is_enforced(self):
        with self.assertRaises(PolicyError):
            Policy.from_rows([{"item": "accuracy_threshold", "rule": "0.60"}])

    def test_family_floor_is_enforced(self):
        with self.assertRaises(PolicyError):
            Policy.from_rows([{"item": "min_families", "rule": "1"}])

    def test_isolation_unit_must_be_family(self):
        with self.assertRaises(PolicyError):
            Policy.from_rows([{"item": "isolation_unit", "rule": "clip"}])

    def test_abstention_cannot_cost_more_than_high_severity(self):
        with self.assertRaises(PolicyError):
            Policy.from_rows(
                [
                    {"item": "abstention_cost", "rule": "50"},
                    {"item": "cost_high", "rule": "6"},
                ]
            )

    def test_confidence_can_be_percent(self):
        policy = Policy.from_rows([{"item": "confidence", "rule": "95"}])
        self.assertAlmostEqual(policy.confidence, 0.95)


class StatsTests(unittest.TestCase):
    def test_zero_samples_has_zero_lower_bound(self):
        self.assertEqual(wilson_lower_bound(0, 0), 0.0)

    def test_lower_bound_is_below_point_estimate(self):
        lower = wilson_lower_bound(6, 6)
        self.assertLess(lower, 1.0)
        self.assertGreater(lower, 0.60)

    def test_small_sample_cannot_clear_threshold(self):
        # 2/2 正确也不应达到 0.80 下界
        self.assertLess(wilson_lower_bound(2, 2), 0.80)


class IsolationTests(unittest.TestCase):
    def test_same_family_in_train_and_test_is_flagged(self):
        clips = [
            clip("C1", "F1", "train", "edge"),
            clip("C2", "F1", "test", "edge"),
        ]
        resolution = resolve_families(clips, [])
        violations = resolution.split_violations(clips)
        self.assertEqual(len(violations), 1)

    def test_same_performance_finding_merges_families(self):
        clips = [
            clip("C1", "F1", "train", "lap"),
            clip("C2", "F2", "test", "lap"),
        ]
        finding = Finding("FD1", ("C1", "C2"), "same_performance", "T1", "2026-06-01")
        resolution = resolve_families(clips, [finding])
        self.assertEqual(resolution.independent_count(["F1", "F2"]), 1)
        self.assertEqual(len(resolution.split_violations(clips)), 1)


class GoldTests(unittest.TestCase):
    def test_two_fine_reviews_resolve_fine(self):
        clips = [clip("C1", "F1", "test", "edge")]
        resolution = resolve_families(clips, [])
        gold = resolve_gold(
            {"C1": [review("R1", "C1", "T1", "edge"), review("R2", "C1", "T2", "edge")]},
            {"C1": "F1"},
            resolution,
            {"C1": "east"},
        )
        self.assertEqual(gold["C1"].basis, "double_fine_agreement")
        self.assertEqual(gold["C1"].fine, "edge")

    def test_coarse_only_agreement_is_not_fine(self):
        clips = [clip("C1", "F1", "test", "edge")]
        resolution = resolve_families(clips, [])
        gold = resolve_gold(
            {
                "C1": [
                    review("R1", "C1", "T1", None, verdict="coarse"),
                    review("R2", "C1", "T2", None, verdict="coarse"),
                ]
            },
            {"C1": "F1"},
            resolution,
            {"C1": "east"},
        )
        self.assertEqual(gold["C1"].basis, "coarse_only_agreement")
        self.assertIsNone(gold["C1"].fine)

    def test_fine_disagreement_is_unresolved(self):
        clips = [clip("C1", "F1", "test", "edge")]
        resolution = resolve_families(clips, [])
        gold = resolve_gold(
            {"C1": [review("R1", "C1", "T1", "edge"), review("R2", "C1", "T2", "lap")]},
            {"C1": "F1"},
            resolution,
            {"C1": "east"},
        )
        self.assertEqual(gold["C1"].basis, "fine_disagreement")
        self.assertIsNone(gold["C1"].fine)

    def test_wrong_tradition_review_does_not_count(self):
        clips = [clip("C1", "F1", "test", "edge")]
        resolution = resolve_families(clips, [])
        gold = resolve_gold(
            {
                "C1": [
                    review("R1", "C1", "T1", "edge"),
                    review("R2", "C1", "T9", "edge", tradition="west"),
                ]
            },
            {"C1": "F1"},
            resolution,
            {"C1": "east"},
        )
        self.assertEqual(gold["C1"].basis, "single_review")


class EvidenceTests(unittest.TestCase):
    def _evidence(self, clips, reviews, findings=(), catalog=None, evaluation=None, version="v2"):
        catalog = catalog if catalog is not None else [catalog_entry("edge")]
        resolution = resolve_families(clips, findings)
        evidence, gold, warnings = evaluate_evidence(
            clips,
            reviews,
            catalog,
            evaluation,
            resolution,
            Policy(),
            findings,
            catalog_version=version,
        )
        return evidence, resolution, warnings

    def test_two_families_and_high_precision_goes_online(self):
        clips = [clip("C1", "F1", "test", "edge"), clip("C2", "F2", "test", "edge")]
        reviews = [
            review("R1", "C1", "T1", "edge"),
            review("R2", "C1", "T2", "edge"),
            review("R3", "C2", "T3", "edge"),
            review("R4", "C2", "T4", "edge"),
        ]
        evaluation = Evaluation(model="m", status="valid")
        evaluation.outcomes[("join", "east", "oak", "edge")] = ClassOutcome(
            task="join", tradition="east", fine="edge", support=40, fine_correct=40
        )
        evidence, _, _ = self._evidence(clips, reviews, evaluation=evaluation)
        item = evidence[("join", "east", "oak", "edge")]
        self.assertEqual(item.family_count, 2)
        self.assertEqual(item.status, "online")

    def test_two_clips_same_family_count_once(self):
        clips = [clip("C1", "F1", "test", "edge"), clip("C2", "F1", "test", "edge")]
        reviews = [
            review("R1", "C1", "T1", "edge"),
            review("R2", "C1", "T2", "edge"),
            review("R3", "C2", "T3", "edge"),
            review("R4", "C2", "T4", "edge"),
        ]
        evaluation = Evaluation(model="m", status="valid")
        evaluation.outcomes[("join", "east", "oak", "edge")] = ClassOutcome(
            task="join", tradition="east", fine="edge", support=40, fine_correct=40
        )
        evidence, _, _ = self._evidence(clips, reviews, evaluation=evaluation)
        item = evidence[("join", "east", "oak", "edge")]
        self.assertEqual(item.family_count, 1)
        self.assertEqual(item.status, "insufficient_families")

    def test_same_performance_finding_invalidates_prior_evidence(self):
        clips = [clip("C1", "F1", "test", "lap"), clip("C2", "F2", "test", "lap")]
        reviews = [
            review("R1", "C1", "T1", "lap", reviewed_at="2026-05-01"),
            review("R2", "C1", "T2", "lap", reviewed_at="2026-05-01"),
            review("R3", "C2", "T3", "lap", reviewed_at="2026-05-02"),
            review("R4", "C2", "T4", "lap", reviewed_at="2026-05-02"),
        ]
        finding = Finding("FD1", ("C1", "C2"), "same_performance", "T5", "2026-06-01")
        evaluation = Evaluation(model="m", status="valid")
        evaluation.outcomes[("join", "east", "oak", "lap")] = ClassOutcome(
            task="join", tradition="east", fine="lap", support=40, fine_correct=40
        )
        evidence, _, _ = self._evidence(
            clips, reviews, findings=[finding], catalog=[catalog_entry("lap")], evaluation=evaluation
        )
        item = evidence[("join", "east", "oak", "lap")]
        self.assertEqual(item.status, "invalidated")
        self.assertIn("invalidated_by_finding", item.blockers)

    def test_low_precision_stays_on_parent(self):
        clips = [clip("C1", "F1", "test", "edge"), clip("C2", "F2", "test", "edge")]
        reviews = [
            review("R1", "C1", "T1", "edge"),
            review("R2", "C1", "T2", "edge"),
            review("R3", "C2", "T3", "edge"),
            review("R4", "C2", "T4", "edge"),
        ]
        evaluation = Evaluation(model="m", status="valid")
        evaluation.outcomes[("join", "east", "oak", "edge")] = ClassOutcome(
            task="join", tradition="east", fine="edge", support=8, fine_correct=3, fine_wrong=5
        )
        evidence, _, _ = self._evidence(clips, reviews, evaluation=evaluation)
        item = evidence[("join", "east", "oak", "edge")]
        self.assertEqual(item.status, "insufficient_precision")

    def test_invalid_isolation_blocks_everything(self):
        clips = [clip("C1", "F1", "train", "edge"), clip("C2", "F2", "test", "edge")]
        reviews = [
            review("R1", "C1", "T1", "edge"),
            review("R2", "C1", "T2", "edge"),
            review("R3", "C2", "T3", "edge"),
            review("R4", "C2", "T4", "edge"),
        ]
        evaluation = Evaluation(model="m", status="invalid_isolation")
        evidence, _, _ = self._evidence(clips, reviews, evaluation=evaluation)
        item = evidence[("join", "east", "oak", "edge")]
        self.assertEqual(item.status, "blocked_isolation")

    def test_context_scope_is_respected(self):
        clips = [clip("C1", "F1", "test", "edge", context="pine")]
        reviews = [review("R1", "C1", "T1", "edge"), review("R2", "C1", "T2", "edge")]
        evaluation = Evaluation(model="m", status="valid")
        evaluation.outcomes[("join", "east", "pine", "edge")] = ClassOutcome(
            task="join", tradition="east", fine="edge", support=40, fine_correct=40
        )
        evidence, _, _ = self._evidence(clips, reviews, evaluation=evaluation)
        self.assertIn("context_not_allowed", evidence[("join", "east", "pine", "edge")].blockers)


class ModelTests(unittest.TestCase):
    def _samples(self):
        clips = []
        observations = []
        for index in range(4):
            clips.append(clip(f"E{index}", f"FE{index}", "train", "edge"))
            clips.append(clip(f"L{index}", f"FL{index}", "train", "lap"))
            clips.append(clip(f"ET{index}", f"FET{index}", "test", "edge"))
            clips.append(clip(f"LT{index}", f"FLT{index}", "test", "lap"))
        for index in range(4):
            observations.append(Observation(f"E{index}", duration_ms=2200 + index, signals={"hold": 0.8}))
            observations.append(Observation(f"L{index}", duration_ms=1500 + index, signals={"hold": 0.2}))
            observations.append(Observation(f"ET{index}", duration_ms=2200 + index, signals={"hold": 0.8}))
            observations.append(Observation(f"LT{index}", duration_ms=1500 + index, signals={"hold": 0.2}))
        return build_samples(clips, observations)

    def test_models_are_compared_on_test_split(self):
        samples = self._samples()
        resolution = resolve_families(
            [clip("E0", "FE0", "train"), clip("ET0", "FET0", "test")], []
        )
        evaluations = evaluate_models(samples, resolution, Policy())
        self.assertTrue(evaluations)
        for evaluation in evaluations.values():
            self.assertEqual(evaluation.total, 8)
            self.assertEqual(evaluation.status, "valid")

    def test_leaky_family_returns_no_accuracy(self):
        clips = [
            clip("C1", "F1", "train", "edge"),
            clip("C2", "F1", "test", "edge"),
            clip("C3", "F2", "train", "lap"),
            clip("C4", "F3", "test", "lap"),
        ]
        observations = [
            Observation("C1", duration_ms=2200, signals={"hold": 0.8}),
            Observation("C2", duration_ms=2200, signals={"hold": 0.8}),
            Observation("C3", duration_ms=1500, signals={"hold": 0.2}),
            Observation("C4", duration_ms=1500, signals={"hold": 0.2}),
        ]
        samples = build_samples(clips, observations)
        evaluations = evaluate_models(samples, resolve_families(clips, []), Policy())
        for evaluation in evaluations.values():
            self.assertEqual(evaluation.status, "invalid_isolation")
            self.assertEqual(evaluation.predictions, {})
        selection = select_model(evaluations, Policy())
        self.assertIsNone(selection.chosen)

    def test_high_cost_error_penalized_over_coverage(self):
        cheap = Evaluation(model="cheap", status="valid", covered=100, total=100)
        cheap.total_cost = 10
        cheap.outcomes[("join", "east", "oak", "edge")] = ClassOutcome(
            task="join", tradition="east", context="oak", fine="edge", support=10, fine_wrong=0, parent_correct=10
        )
        risky = Evaluation(model="risky", status="valid", covered=100, total=100)
        risky.total_cost = 5
        risky.outcomes[("join", "east", "oak", "edge")] = ClassOutcome(
            task="join",
            tradition="east",
            context="oak",
            fine="edge",
            support=10,
            fine_wrong=1,
            cost=6,
            high_cost_errors=1,
        )
        selection = select_model({"cheap": cheap, "risky": risky}, Policy())
        self.assertEqual(selection.chosen, "cheap")
        self.assertTrue(selection.high_cost_guard)


class MigrationTests(unittest.TestCase):
    def test_removed_code_requires_explanation_update(self):
        catalog = [
            catalog_entry("joinery", level="coarse", parent=None, scope="", max_gran="fine"),
            catalog_entry("joinery", version="v2", level="coarse", parent=None, scope="", max_gran="fine"),
            catalog_entry("lap", version="v1", level="fine", scope="", max_gran="fine"),
        ]
        changes, _ = diff_versions(catalog, "v1", "v2")
        self.assertIn("removed", [change.change for change in changes])

    def test_online_fine_class_updates_explanation(self):
        clips = [clip("C1", "F1", "test", "edge")]
        chapters = [
            CourseChapter("J1", "第1章", "C1", "joinery", "榫接通用说明", "high", "published", "2026-01-01")
        ]
        catalog = [
            catalog_entry("joinery", version="v1", level="coarse", parent=None, scope=""),
            catalog_entry("joinery", version="v2", level="coarse", parent=None, scope=""),
            catalog_entry("edge", version="v2"),
        ]
        evidence = {
            ("join", "east", "oak", "edge"): type(
                "E",
                (),
                {
                    "task": "join",
                    "tradition": "east",
                    "fine": "edge",
                    "coarse": "joinery",
                    "status": "online",
                    "family_count": 2,
                    "precision_lower": 0.82,
                    "blockers": [],
                },
            )()
        }
        plan = build_migration_plan(chapters, catalog, evidence, clips, from_version="v1", to_version="v2")
        action = plan.actions[0]
        self.assertEqual(action.action, "update_explanation")
        self.assertEqual(action.new_code, "edge")

    def test_unqualified_fine_class_keeps_parent(self):
        clips = [clip("C1", "F1", "test", "lap")]
        chapters = [
            CourseChapter("J1", "第1章", "C1", "joinery", "榫接通用说明", "high", "published", "2026-01-01")
        ]
        catalog = [
            catalog_entry("joinery", version="v1", level="coarse", parent=None, scope=""),
            catalog_entry("joinery", version="v2", level="coarse", parent=None, scope=""),
        ]
        evidence = {
            ("join", "east", "oak", "lap"): type(
                "E",
                (),
                {
                    "task": "join",
                    "tradition": "east",
                    "fine": "lap",
                    "coarse": "joinery",
                    "status": "invalidated",
                    "family_count": 2,
                    "precision_lower": 0.9,
                    "blockers": ["invalidated_by_finding"],
                },
            )()
        }
        plan = build_migration_plan(chapters, catalog, evidence, clips, from_version="v1", to_version="v2")
        action = plan.actions[0]
        self.assertEqual(action.action, "keep_parent")

    def test_draft_chapter_is_blocked(self):
        clips = [clip("C1", "F1", "test", "edge")]
        chapters = [
            CourseChapter("J1", "第1章", "C1", "joinery", "说明", "high", "draft", "2026-01-01")
        ]
        catalog = [
            catalog_entry("joinery", version="v1", level="coarse", parent=None, scope=""),
            catalog_entry("joinery", version="v2", level="coarse", parent=None, scope=""),
        ]
        plan = build_migration_plan(chapters, catalog, {}, clips, from_version="v1", to_version="v2")
        self.assertEqual(plan.actions[0].action, "blocked")


class ReviewPlanTests(unittest.TestCase):
    def test_tradition_mismatch_means_unassigned(self):
        clips = [clip("C1", "F1", "test", "edge", tradition="east")]
        reviewers = [Reviewer("T9", "west", "edge", 120, "2026-07")]
        chapters = [
            CourseChapter("J1", "第1章", "C1", "joinery", "说明", "high", "published", "2026-01-01")
        ]
        evidence = {}
        plan = build_review_plan(
            clips, reviewers, evidence, [], chapters, Policy(), period="2026-07"
        )
        self.assertEqual(plan.assignments, [])
        self.assertEqual(len(plan.unassigned), 1)

    def test_limited_budget_is_not_overspent(self):
        clips = [clip(f"C{i}", f"F{i}", "test", "edge") for i in range(3)]
        reviewers = [Reviewer("T1", "east", "edge", 40, "2026-07")]
        chapters = [
            CourseChapter(f"J{i}", f"第{i}章", f"C{i}", "joinery", "说明", "high", "published", "2026-01-01")
            for i in range(3)
        ]
        plan = build_review_plan(clips, reviewers, {}, [], chapters, Policy(), period="2026-07")
        self.assertLessEqual(plan.total_minutes, 40)
        self.assertTrue(plan.unassigned)

    def test_same_performance_review_is_not_scheduled(self):
        clips = [clip("C1", "F1", "test", "lap"), clip("C2", "F2", "test", "lap")]
        reviewers = [Reviewer("T1", "east", "lap", 120, "2026-07")]
        chapters = [
            CourseChapter("J1", "第1章", "C1", "joinery", "说明", "high", "published", "2026-01-01")
        ]
        finding = Finding("FD1", ("C1", "C2"), "same_performance", "T5", "2026-06-01")
        plan = build_review_plan(
            clips,
            reviewers,
            {},
            [],
            chapters,
            Policy(),
            period="2026-07",
            resolution=resolve_families(clips, [finding]),
            findings=[finding],
        )
        self.assertEqual(plan.assignments, [])
        self.assertTrue(plan.not_needed)


class FixtureTests(unittest.TestCase):
    def test_checked_in_fixtures_do_not_claim_online_classes(self):
        from craftreview.pipeline import Dataset, analyze

        dataset = Dataset.load(Path(__file__).resolve().parents[1] / "fixtures")
        analysis = analyze(dataset)
        self.assertEqual([e for e in analysis.evidence.values() if e.status == "online"], [])
        for item in analysis.evidence.values():
            self.assertIn("insufficient_families", item.blockers)

    def test_repository_has_no_quantitative_claims(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / "README.md").read_text(encoding="utf-8")
        text += (root / "docs" / "业务规程.md").read_text(encoding="utf-8")
        self.assertIn("合成样例", text)




class SemanticsTests(unittest.TestCase):
    def test_single_corpus_family_is_not_schedulable(self):
        clips = [clip("C1", "F1", "test", "coat", tradition="north", task="paint", context="clay", coarse="finish")]
        reviewers = [Reviewer("T1", "north", "coat", 120, "2026-07")]
        chapters = [
            CourseChapter("P1", "第1章", "C1", "finish", "说明", "high", "published", "2026-01-01")
        ]
        evidence = {
            ("paint", "north", "clay", "coat"): type(
                "E",
                (),
                {
                    "task": "paint",
                    "tradition": "north",
                    "context": "clay",
                    "fine": "coat",
                    "coarse": "finish",
                    "status": "insufficient_precision",
                    "family_count": 0,
                    "corpus_family_count": 1,
                    "can_ever_reach_two_families": False,
                    "precision_lower": 0.0,
                    "blockers": ["precision_below_threshold"],
                },
            )()
        }
        plan = build_review_plan(clips, reviewers, evidence, [], chapters, Policy(), period="2026-07")
        self.assertEqual(plan.assignments, [])
        self.assertTrue(any("征集" in note for note in plan.notes))

    def test_second_reviewer_must_differ_from_existing(self):
        clips = [clip("C1", "F1", "test", "edge")]
        reviewers = [
            Reviewer("T1", "east", "edge", 120, "2026-07"),
            Reviewer("T2", "east", "edge", 120, "2026-07"),
        ]
        chapters = [
            CourseChapter("J1", "第1章", "C1", "joinery", "说明", "high", "published", "2026-01-01")
        ]
        existing = [review("R1", "C1", "T1", "edge")]
        plan = build_review_plan(
            clips, reviewers, {}, existing, chapters, Policy(), period="2026-07"
        )
        self.assertEqual([a.reviewer for a in plan.assignments], ["T2"])

    def test_unresolved_fine_disagreement_blocks_migration(self):
        clips = [clip("C1", "F1", "test", "edge")]
        chapters = [
            CourseChapter("J1", "第1章", "C1", "joinery", "说明", "high", "published", "2026-01-01")
        ]
        catalog = [
            catalog_entry("joinery", version="v1", level="coarse", parent=None, scope=""),
            catalog_entry("joinery", version="v2", level="coarse", parent=None, scope=""),
            catalog_entry("edge", version="v2"),
        ]
        evidence = {
            ("join", "east", "oak", "edge"): type(
                "E",
                (),
                {
                    "task": "join",
                    "tradition": "east",
                    "context": "oak",
                    "fine": "edge",
                    "coarse": "joinery",
                    "status": "unresolved",
                    "family_count": 2,
                    "precision_lower": 0.0,
                    "blockers": ["fine_disagreement"],
                },
            )()
        }
        plan = build_migration_plan(chapters, catalog, evidence, clips, from_version="v1", to_version="v2")
        self.assertEqual(plan.actions[0].action, "keep_parent")
        self.assertIn("fine_disagreement", plan.actions[0].blockers)
        self.assertIn("双评不一致", plan.actions[0].reason)


class ContextTests(unittest.TestCase):
    def test_same_fine_name_in_different_tools_keeps_separate_evidence(self):
        clips = [
            clip("C1", "F1", "test", "edge", context="oak"),
            clip("C2", "F2", "test", "edge", context="pine"),
        ]
        reviews = [
            review("R1", "C1", "T1", "edge"),
            review("R2", "C1", "T2", "edge"),
            review("R3", "C2", "T3", "edge"),
            review("R4", "C2", "T4", "edge"),
        ]
        evaluation = Evaluation(model="m", status="valid")
        evaluation.outcomes[("join", "east", "oak", "edge")] = ClassOutcome(
            task="join", tradition="east", context="oak", fine="edge", support=40, fine_correct=40
        )
        evaluation.outcomes[("join", "east", "pine", "edge")] = ClassOutcome(
            task="join", tradition="east", context="pine", fine="edge", support=40, fine_correct=40
        )
        catalog = [catalog_entry("edge", scope="task:join; context:oak")]
        evidence, _, _ = evaluate_evidence(
            clips,
            reviews,
            catalog,
            evaluation,
            resolve_families(clips, []),
            Policy(),
            catalog_version="v2",
        )
        oak = evidence[("join", "east", "oak", "edge")]
        pine = evidence[("join", "east", "pine", "edge")]
        # 两个器具各自只有一个家族；不能把它们的家族合并成“两个独立家族”。
        self.assertEqual(oak.family_count, 1)
        self.assertEqual(pine.family_count, 1)
        self.assertEqual(oak.status, "insufficient_families")
        self.assertIn("context_not_allowed", pine.blockers)
        self.assertEqual(len(evidence), 2)


class CliTests(unittest.TestCase):
    def test_cli_json_is_machine_readable(self):
        import json
        import subprocess
        import sys

        root = Path(__file__).resolve().parents[1]
        if not (root / "examples" / "demo" / "primary.csv").exists():
            subprocess.run(
                [sys.executable, str(root / "examples" / "build_demo.py")],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
        result = subprocess.run(
            [sys.executable, "-m", "craftreview", "analyze", "--data", str(root / "examples" / "demo"), "--json"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertIn("evidence", payload)
        self.assertIn("migration", payload)
        self.assertIn("review_plan", payload)


if __name__ == "__main__":
    unittest.main()


class RepresentativeSchedulingTests(unittest.TestCase):
    def test_one_representative_per_family(self):
        clips = [
            clip("C1", "F1", "test", "edge"),
            clip("C2", "F1", "test", "edge"),
            clip("C3", "F2", "test", "edge"),
        ]
        reviewers = [Reviewer("T1", "east", "edge", 120, "2026-07")]
        chapters = [
            CourseChapter("J1", "第1章", "C1", "joinery", "说明", "high", "published", "2026-01-01")
        ]
        evidence = {
            ("join", "east", "oak", "edge"): type(
                "E",
                (),
                {
                    "task": "join",
                    "tradition": "east",
                    "context": "oak",
                    "fine": "edge",
                    "coarse": "joinery",
                    "status": "insufficient_families",
                    "family_count": 0,
                    "corpus_family_count": 2,
                    "can_ever_reach_two_families": True,
                    "corpus_families": ("F1", "F2"),
                    "eligible_families": (),
                    "precision_lower": 0.0,
                    "blockers": ["insufficient_families"],
                },
            )()
        }
        plan = build_review_plan(
            clips, reviewers, evidence, [], chapters, Policy(), period="2026-07"
        )
        assigned_families = {clip.family for clip in clips if any(a.task.clip_id == clip.clip_id for a in plan.assignments)}
        # F1 只应有一个代表片段；F2 也应有机会
        f1_count = sum(1 for a in plan.assignments if a.task.clip_id in {"C1", "C2"})
        self.assertLessEqual(f1_count, 1)
        self.assertIn("F1", assigned_families)
