#!/usr/bin/env python3
"""生成可复现的合成演示数据。

用法：``python3 examples/build_demo.py``

生成：

* ``examples/demo``：完整流程（模型评估、细类资格、迁移、复核排期）；
* ``examples/leaky``：同一片源家族跨训练与检验，用于展示评估被拒。
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CLASSES = {
    "edge": {
        "task": "join",
        "context": "oak",
        "tradition": "east",
        "coarse": "joinery",
        "duration": 2200,
        "landmarks": 12,
        "hold": 0.80,
        "sweep": 0.20,
        "press": 0.40,
    },
    "lap": {
        "task": "join",
        "context": "oak",
        "tradition": "east",
        "coarse": "joinery",
        "duration": 1500,
        "landmarks": 7,
        "hold": 0.20,
        "sweep": 0.80,
        "press": 0.55,
    },
    "relief": {
        "task": "carve",
        "context": "elm",
        "tradition": "west",
        "coarse": "carving",
        "duration": 3200,
        "landmarks": 16,
        "hold": 0.65,
        "sweep": 0.35,
        "press": 0.75,
    },
    "incised": {
        "task": "carve",
        "context": "elm",
        "tradition": "west",
        "coarse": "carving",
        "duration": 2400,
        "landmarks": 9,
        "hold": 0.30,
        "sweep": 0.70,
        "press": 0.25,
    },
    "coat": {
        "task": "paint",
        "context": "clay",
        "tradition": "north",
        "coarse": "finish",
        "duration": 1800,
        "landmarks": 10,
        "hold": 0.52,
        "sweep": 0.48,
        "press": 0.58,
        "noise": 0.30,
        "duration_jitter": 700,
    },
    "wash": {
        "task": "paint",
        "context": "clay",
        "tradition": "north",
        "coarse": "finish",
        "duration": 1200,
        "landmarks": 6,
        "hold": 0.38,
        "sweep": 0.62,
        "press": 0.32,
        "noise": 0.30,
        "duration_jitter": 700,
    },
}

# 每个细类：训练家族与样本数、检验家族与样本数
LAYOUT = {
    "edge": ("FE_TA", 6, [("F_edgeA", 6), ("F_edgeB", 6)]),
    "lap": ("FE_TB", 6, [("F_lapA", 6), ("F_lapB", 6)]),
    "relief": ("FW_TA", 6, [("F_relA", 6), ("F_relB", 6)]),
    "incised": ("FW_TB", 6, [("F_incA", 6), ("F_incB", 6)]),
    "coat": ("FN_TA", 4, [("F_coatA", 6)]),
    "wash": ("FN_TB", 4, [("F_washA", 6)]),
}

CLIP_ROWS: list[dict[str, str]] = []
OBS_ROWS: list[dict[str, str]] = []


def _obs(base: dict, index: int, salt: int) -> dict[str, float]:
    # 每个类别可单独设置抖动。罩面/洗色故意做得接近，用来演示模型
    # 在边界上选择“退父类”而不是硬猜细类。
    noise = base.get("noise", 0.01)
    duration_jitter = base.get("duration_jitter", 20)
    wobble = noise * ((index + salt) % 3)
    return {
        "duration_ms": float(base["duration"] + duration_jitter * ((index + salt) % 4)),
        "landmark_count": float(base["landmarks"] + (index + salt) % 2),
        "sig_hold": base["hold"] + wobble,
        "sig_sweep": base["sweep"] - wobble,
        "sig_press": base["press"] + wobble / 2,
    }


def _make_clip(
    clip_id: str,
    family: str,
    fine: str,
    split: str,
    index: int,
    salt: int,
) -> None:
    meta = CLASSES[fine]
    # 罩面与洗色有少量“示意动作近似”的检验片段：观测与另一类几乎重合。
    # 这类片段正是示意动作被误当规范工序的高成本风险来源。
    if split == "test" and fine in {"coat", "wash"} and index in {2, 3}:
        sibling = CLASSES["wash" if fine == "coat" else "coat"]
        blended = dict(meta)
        for key in ("duration", "landmarks", "hold", "sweep", "press"):
            blended[key] = sibling[key]
        blended["noise"] = 0.02
        blended["duration_jitter"] = 5
        meta = blended
    CLIP_ROWS.append(
        {
            "clip": clip_id,
            "family": family,
            "tradition": meta["tradition"],
            "task": meta["task"],
            "context": meta["context"],
            "split": split,
            "coarse": meta["coarse"],
            "fine": fine,
            "reviewer": "",
            "version": "v1",
        }
    )
    obs = _obs(meta, index, salt)
    OBS_ROWS.append(
        {
            "clip": clip_id,
            "task": meta["task"],
            "context": meta["context"],
            "family": family,
            "tradition": meta["tradition"],
            "duration_ms": f"{obs['duration_ms']:.0f}",
            "landmark_count": f"{obs['landmark_count']:.0f}",
            "sig_hold": f"{obs['sig_hold']:.3f}",
            "sig_sweep": f"{obs['sig_sweep']:.3f}",
            "sig_press": f"{obs['sig_press']:.3f}",
        }
    )


def build_clips() -> None:
    for fine, (train_family, train_n, tests) in LAYOUT.items():
        short = fine.upper()[:2]
        for i in range(train_n):
            _make_clip(f"{short}_TRAIN{i + 1}", train_family, fine, "train", i, 1)
        for family, n in tests:
            for i in range(n):
                _make_clip(f"{short}_{family}_{i + 1}", family, fine, "test", i, 2)


REVIEWS = [
    # edge：两个独立家族、两位细类复核人
    ("RV01", "ED_F_edgeA_1", "T1", "east", "joinery", "edge", "fine", "2026-07-02"),
    ("RV02", "ED_F_edgeA_1", "T2", "east", "joinery", "edge", "fine", "2026-07-02"),
    ("RV03", "ED_F_edgeB_1", "T1", "east", "joinery", "edge", "fine", "2026-07-03"),
    ("RV04", "ED_F_edgeB_1", "T2", "east", "joinery", "edge", "fine", "2026-07-03"),
    # relief：两个独立家族
    ("RV05", "RE_F_relA_1", "T3", "west", "carving", "relief", "fine", "2026-07-04"),
    ("RV06", "RE_F_relA_1", "T4", "west", "carving", "relief", "fine", "2026-07-04"),
    ("RV07", "RE_F_relB_1", "T3", "west", "carving", "relief", "fine", "2026-07-05"),
    ("RV08", "RE_F_relB_1", "T4", "west", "carving", "relief", "fine", "2026-07-05"),
    # incised：两位只达成父类，不计入细类验证
    ("RV09", "IN_F_incA_1", "T3", "west", "carving", "", "coarse", "2026-07-06"),
    ("RV10", "IN_F_incA_1", "T4", "west", "carving", "", "coarse", "2026-07-06"),
    # incised：单独一位给出细类，也不构成双评
    ("RV11", "IN_F_incA_2", "T4", "west", "carving", "incised", "fine", "2026-07-06"),
    # lap：核对判定同源之前的细类双评，之后资格应失效
    ("RV12", "LA_F_lapA_1", "T1", "east", "joinery", "lap", "fine", "2026-05-01"),
    ("RV13", "LA_F_lapA_1", "T2", "east", "joinery", "lap", "fine", "2026-05-01"),
    ("RV14", "LA_F_lapB_1", "T1", "east", "joinery", "lap", "fine", "2026-05-02"),
    ("RV15", "LA_F_lapB_1", "T2", "east", "joinery", "lap", "fine", "2026-05-02"),
]

FINDINGS = [
    ("FD01", "LA_F_lapA_1|LA_F_lapB_1", "same_performance", "T5", "2026-06-15"),
]

CATALOG = [
    ("joinery", "v1", "coarse", "", "榫接工序", "", "yes", "fine"),
    ("carving", "v1", "coarse", "", "雕刻工序", "", "yes", "fine"),
    ("finish", "v1", "coarse", "", "表面处理", "", "yes", "fine"),
    ("joinery", "v2", "coarse", "", "榫接工序", "task:join", "yes", "fine"),
    ("carving", "v2", "coarse", "", "雕刻工序", "task:carve", "yes", "fine"),
    ("finish", "v2", "coarse", "", "表面处理", "task:paint", "yes", "fine"),
    ("edge", "v2", "fine", "joinery", "边缘接合", "task:join; context:oak", "yes", "fine"),
    ("lap", "v1", "fine", "joinery", "搭接", "task:join; context:oak", "yes", "fine"),
    ("relief", "v2", "fine", "carving", "浮雕", "task:carve; context:elm", "yes", "fine"),
    ("incised", "v2", "fine", "carving", "阴刻", "task:carve; context:elm", "yes", "fine"),
    ("coat", "v2", "fine", "finish", "罩面", "task:paint; context:clay", "yes", "fine"),
    ("wash", "v2", "fine", "finish", "洗色", "task:paint; context:clay", "no", "fine"),
]

COURSES = [
    ("J101", "第1章 边缘接合", "ED_F_edgeA_1", "joinery", "榫接通用说明", "high", "published", "2026-01-01"),
    ("J102", "第2章 搭接", "LA_F_lapA_1", "joinery", "榫接通用说明", "high", "published", "2026-01-01"),
    ("R201", "第3章 浮雕", "RE_F_relA_1", "carving", "雕刻通用说明", "high", "published", "2026-02-01"),
    ("R202", "第4章 阴刻", "IN_F_incA_1", "carving", "雕刻通用说明", "medium", "published", "2026-02-01"),
    ("P301", "第5章 罩面", "CO_F_coatA_1", "finish", "表面处理通用说明", "high", "published", "2026-03-01"),
    ("P302", "第6章 洗色", "WA_F_washA_1", "finish", "表面处理通用说明", "low", "published", "2026-03-01"),
    ("J103", "第7章 边缘接合复练", "ED_F_edgeB_1", "joinery", "榫接通用说明", "high", "published", "2026-03-15"),
    ("P303", "第8章 罩面草案", "CO_F_coatA_2", "finish", "表面处理通用说明", "high", "draft", "2026-04-01"),
    # 边界检验片段且用于高成本课程：示意动作被误标成规范工序时代价最高
    ("P304", "第9章 罩面与洗色辨析", "CO_F_coatA_3", "finish", "表面处理通用说明", "high", "published", "2026-04-10"),
    ("P305", "第10章 洗色与罩面辨析", "WA_F_washA_3", "finish", "表面处理通用说明", "high", "published", "2026-04-10"),
    # 阴刻需要两位细类复核，本期只有 T6 有余额，用来演示“排不下”的能力缺口
    ("R203", "第11章 阴刻复练", "IN_F_incA_2", "carving", "雕刻通用说明", "medium", "published", "2026-04-15"),
]


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def write_demo() -> None:
    target = ROOT / "demo"
    target.mkdir(parents=True, exist_ok=True)
    CLIP_ROWS.clear()
    OBS_ROWS.clear()
    build_clips()
    write_csv(target / "primary.csv", list(CLIP_ROWS[0]), [[r[k] for k in CLIP_ROWS[0]] for r in CLIP_ROWS])
    write_csv(target / "observations.csv", list(OBS_ROWS[0]), [[r[k] for k in OBS_ROWS[0]] for r in OBS_ROWS])
    write_csv(
        target / "reviews.csv",
        ["review", "clip", "reviewer", "tradition", "coarse", "fine", "verdict", "evidence_key", "reviewed_at"],
        [
            [rid, clip, reviewer, tradition, coarse, fine, verdict, "", reviewed_at]
            for rid, clip, reviewer, tradition, coarse, fine, verdict, reviewed_at in REVIEWS
        ],
    )
    write_csv(
        target / "findings.csv",
        ["finding", "clips", "verdict", "reviewer", "decided_at", "invalidates"],
        [[fid, clips, verdict, reviewer, decided_at, ""] for fid, clips, verdict, reviewer, decided_at in FINDINGS],
    )
    write_csv(
        target / "catalog.csv",
        ["code", "version", "level", "parent", "label", "scope_notes", "approved", "max_granularity"],
        [list(row) for row in CATALOG],
    )
    write_csv(
        target / "courses.csv",
        ["course", "chapter", "clip", "catalog_code", "explanation", "severity", "status", "valid_from", "valid_to"],
        [list(row) + [""] for row in COURSES],
    )
    write_csv(
        target / "support-1.csv",
        ["reviewer", "tradition", "skill", "minutes", "period"],
        [
            ["T1", "east", "join", "120", "2026-07"],
            ["T2", "east", "edge", "90", "2026-07"],
            ["T3", "west", "carve", "120", "2026-07"],
            ["T4", "west", "relief", "60", "2026-07"],
            ["T5", "north", "paint", "45", "2026-07"],
            ["T6", "west", "carve", "30", "2026-07"],
        ],
    )
    write_csv(
        target / "support-2.csv",
        ["policy", "item", "rule"],
        [
            ["P1", "scope", "reviewed"],
            ["P2", "version", "v2"],
            ["P3", "accuracy_threshold", "0.80"],
            ["P4", "confidence", "0.95"],
            ["P5", "review_minutes", "20"],
        ],
    )


def write_leaky() -> None:
    target = ROOT / "leaky"
    target.mkdir(parents=True, exist_ok=True)
    write_csv(
        target / "primary.csv",
        ["clip", "family", "tradition", "task", "context", "split", "coarse", "fine", "reviewer", "version"],
        [
            ["L1", "F_same", "east", "join", "oak", "train", "joinery", "edge", "", "v1"],
            ["L2", "F_same", "east", "join", "oak", "test", "joinery", "edge", "", "v1"],
            ["L3", "F_other", "east", "join", "oak", "train", "joinery", "lap", "", "v1"],
            ["L4", "F_third", "east", "join", "oak", "test", "joinery", "lap", "", "v1"],
        ],
    )
    write_csv(
        target / "observations.csv",
        ["clip", "task", "context", "family", "tradition", "duration_ms", "landmark_count", "sig_hold", "sig_sweep"],
        [
            ["L1", "join", "oak", "F_same", "east", "2200", "12", "0.80", "0.20"],
            ["L2", "join", "oak", "F_same", "east", "2220", "12", "0.79", "0.21"],
            ["L3", "join", "oak", "F_other", "east", "1500", "7", "0.20", "0.80"],
            ["L4", "join", "oak", "F_third", "east", "1520", "8", "0.21", "0.79"],
        ],
    )
    # 故意不提供复核人名册与策略表：演示使用手册默认阈值，且没有复核人
    # 资质信息时不会把无资质的人排进复核。


if __name__ == "__main__":
    write_demo()
    write_leaky()
    print("演示数据已生成：examples/demo, examples/leaky")
