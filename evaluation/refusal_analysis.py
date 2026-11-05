"""Refusal analysis over saved inference results.

A cases CSV names the expected refusal code and category for each saved result.
The report checks what the software actually emitted, shows every required
category (including categories with no cases), and separates clearly labelled
synthetic fixtures from real captures.

The current engine has no excessive-tilt threshold. That category is therefore
reported as not covered unless a future result actually carries an
``excessive_tilt`` code; the report never invents one.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from evaluation.common import (
    EvaluationError,
    ResultView,
    image_present,
    md_table,
    parse_optional_bool,
    read_csv,
    read_result,
    utc_now,
    write_csv,
    write_json,
    write_markdown,
)

ANALYSIS_SCHEMA = "measureit.refusal_analysis/1"

REQUIRED_CATEGORIES = (
    "missing_reference",
    "invalid_marker_geometry",
    "occlusion",
    "unreliable_segmentation",
    "invalid_calibration",
    "excessive_tilt",
)

CATEGORY_CODES: dict[str, set[str]] = {
    "missing_reference": {"missing_reference"},
    "invalid_marker_geometry": {"invalid_geometry"},
    "invalid_object_geometry": {"invalid_geometry"},
    "occlusion": {"partial_occlusion"},
    "unreliable_segmentation": {
        "missing_object",
        "partial_occlusion",
        "invalid_geometry",
        "low_reliability",
    },
    "invalid_calibration": {
        "missing_calibration",
        "invalid_calibration",
        "calibration_image_size_mismatch",
    },
    "excessive_tilt": {"excessive_tilt"},
    "reference_unavailable": {
        "missing_reference_config",
        "invalid_reference_config",
        "reference_size_unknown",
    },
    "checkpoint_unavailable": {"missing_checkpoint", "invalid_checkpoint"},
    "missing_result": {"missing_result"},
    "ok": {"ok"},
}

KNOWN_CODES = set().union(*CATEGORY_CODES.values()) | {
    "invalid_result",
    "low_reliability",
    "ok",
}


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    rows = read_csv(
        path,
        required=("case_id", "stem", "expected_code", "expected_category"),
        optional=("synthetic", "notes"),
        what="refusal cases CSV",
    )
    if not rows:
        raise EvaluationError("refusal cases CSV has no cases")

    cases: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    stems: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        case_id = row["case_id"]
        stem = row["stem"]
        expected_code = row["expected_code"]
        expected_category = row["expected_category"]
        if not case_id:
            raise EvaluationError(f"refusal case row {line_number}: case_id is empty")
        if not stem:
            raise EvaluationError(f"refusal case {case_id}: stem is empty")
        if case_id in case_ids:
            raise EvaluationError(f"refusal cases CSV repeats case_id {case_id!r}")
        if stem in stems:
            raise EvaluationError(
                f"refusal cases CSV repeats stem {stem!r}; one saved result "
                "cannot have two expected outcomes"
            )
        if expected_code not in KNOWN_CODES:
            raise EvaluationError(
                f"refusal case {case_id}: unknown expected_code {expected_code!r}"
            )
        if expected_category not in CATEGORY_CODES:
            raise EvaluationError(
                f"refusal case {case_id}: unknown expected_category "
                f"{expected_category!r}"
            )
        case_ids.add(case_id)
        stems.add(stem)
        cases.append(
            {
                "case_id": case_id,
                "stem": stem,
                "expected_code": expected_code,
                "expected_category": expected_category,
                "synthetic": parse_optional_bool(
                    row.get("synthetic"), f"case {case_id} synthetic"
                ),
                "notes": row.get("notes", ""),
            }
        )
    return cases


def observed_category(code: str, message: str) -> str:
    lowered = message.lower()
    if code == "missing_reference":
        return "missing_reference"
    if code == "partial_occlusion":
        return "occlusion"
    if code in {"missing_object", "low_reliability"}:
        return "unreliable_segmentation"
    if code in {"missing_calibration", "invalid_calibration", "calibration_image_size_mismatch"}:
        return "invalid_calibration"
    if code == "excessive_tilt":
        return "excessive_tilt"
    if code == "invalid_geometry":
        return (
            "invalid_marker_geometry"
            if "reference marker" in lowered
            else "invalid_object_geometry"
        )
    if code in {
        "missing_reference_config",
        "invalid_reference_config",
        "reference_size_unknown",
    }:
        return "reference_unavailable"
    if code in {"missing_checkpoint", "invalid_checkpoint"}:
        return "checkpoint_unavailable"
    if code == "missing_result":
        return "missing_result"
    if code == "ok":
        return "ok"
    return "other"


def category_matches(category: str, code: str, message: str) -> bool:
    lowered = message.lower()
    if category == "invalid_marker_geometry":
        return code == "invalid_geometry" and "reference marker" in lowered
    if category == "invalid_object_geometry":
        return code == "invalid_geometry" and "reference marker" not in lowered
    return code in CATEGORY_CODES[category]


def analyze_cases(
    cases: Sequence[dict[str, Any]],
    *,
    results_dir: str | Path,
    images_dir: str | Path | None = None,
) -> dict[str, Any]:
    images_path = None if images_dir is None else Path(images_dir)
    observations = []
    for case in cases:
        view: ResultView = read_result(results_dir, case["stem"])
        expected_code_match = view.code == case["expected_code"]
        category_match = category_matches(
            case["expected_category"], view.code, view.message
        )
        observations.append(
            {
                "case_id": case["case_id"],
                "stem": case["stem"],
                "expected_code": case["expected_code"],
                "expected_category": case["expected_category"],
                "observed_status": view.status,
                "observed_code": view.code,
                "observed_category": observed_category(view.code, view.message),
                "observed_reliability": view.reliability,
                "message": view.message,
                "expected_code_match": expected_code_match,
                "category_match": category_match,
                "passed": expected_code_match and category_match,
                "synthetic": case["synthetic"],
                "image_present": image_present(images_path, case["stem"]),
                "result_path": None if view.result_path is None else str(view.result_path),
                "notes": case["notes"],
            }
        )

    results_available = any(item["result_path"] is not None for item in observations)
    passed = sum(1 for item in observations if item["passed"])
    failed = len(observations) - passed
    synthetic_counts = {
        "synthetic": sum(1 for item in observations if item["synthetic"] is True),
        "real": sum(1 for item in observations if item["synthetic"] is False),
        "unknown": sum(1 for item in observations if item["synthetic"] is None),
    }
    code_counts = Counter(item["observed_code"] for item in observations)

    categories: list[dict[str, Any]] = []
    ordered = list(REQUIRED_CATEGORIES) + sorted(
        {
            item["expected_category"]
            for item in observations
            if item["expected_category"] not in REQUIRED_CATEGORIES
        }
    )
    for category in ordered:
        items = [item for item in observations if item["expected_category"] == category]
        category_passed = sum(1 for item in items if item["passed"])
        note = ""
        if category == "excessive_tilt" and not items:
            note = (
                "not covered: the current engine implements no excessive-tilt "
                "threshold and no case supplied an excessive_tilt result"
            )
        elif not items:
            note = "not covered: no case supplied for this category"
        elif category_passed != len(items):
            note = "some cases did not match their expected outcome"
        categories.append(
            {
                "category": category,
                "required": category in REQUIRED_CATEGORIES,
                "n_cases": len(items),
                "n_passed": category_passed,
                "n_failed": len(items) - category_passed,
                "covered": bool(items),
                "note": note,
            }
        )

    coverage_gaps = [
        category["category"] for category in categories if not category["covered"]
    ]

    if not results_available:
        status = "unavailable"
        code = "missing_results"
        message = f"no saved inference results were found in {results_dir}"
    elif failed:
        status = "mismatch"
        code = "case_mismatch"
        message = f"{failed} case(s) did not match the expected refusal outcome"
    else:
        status = "ok"
        code = None
        message = "every case matched its expected refusal outcome"

    return {
        "schema": ANALYSIS_SCHEMA,
        "created_utc": utc_now(),
        "status": status,
        "code": code,
        "message": message,
        "cases_path": None,
        "results_dir": str(results_dir),
        "images_dir": None if images_path is None else str(images_path),
        "results_available": results_available,
        "summary": {
            "n_cases": len(observations),
            "n_passed": passed,
            "n_failed": failed,
            "pass_rate_pct": 100.0 * passed / len(observations) if observations else None,
            "synthetic_counts": synthetic_counts,
            "observed_code_counts": dict(sorted(code_counts.items())),
            "coverage_gaps": coverage_gaps,
            "excessive_tilt_implemented": False,
        },
        "categories": categories,
        "cases": observations,
        "notes": [
            "a case passes only when both the observed code and the expected "
            "category match",
            "synthetic fixtures are labelled software checks, not real-world "
            "evidence",
            "the current engine has no excessive-tilt threshold; that category "
            "is not claimed as covered without an actual excessive_tilt result",
        ],
    }


def build_refusal_analysis(
    *,
    cases_path: str | Path,
    results_dir: str | Path,
    out_dir: str | Path,
    images_dir: str | Path | None = None,
) -> dict[str, Any]:
    cases = load_cases(cases_path)
    payload = analyze_cases(cases, results_dir=results_dir, images_dir=images_dir)
    payload["cases_path"] = str(Path(cases_path))

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    payload["report_paths"] = {
        "json": str(out_path / "refusal_analysis.json"),
        "csv": str(out_path / "refusal_cases.csv"),
        "markdown": str(out_path / "refusal_analysis.md"),
    }
    write_json(out_path / "refusal_analysis.json", payload)

    header = [
        "case_id",
        "stem",
        "expected_code",
        "expected_category",
        "observed_status",
        "observed_code",
        "observed_category",
        "observed_reliability",
        "expected_code_match",
        "category_match",
        "passed",
        "synthetic",
        "image_present",
        "result_path",
        "message",
        "notes",
    ]
    write_csv(
        out_path / "refusal_cases.csv",
        header,
        [
            [
                "" if item[name] is None else item[name]
                for name in header
            ]
            for item in payload["cases"]
        ],
    )

    summary = payload["summary"]
    pass_rate_text = (
        "—"
        if summary["pass_rate_pct"] is None
        else f"{summary['pass_rate_pct']:.2f} %"
    )
    lines = [
        "# Refusal analysis",
        "",
        f"- Generated: {payload['created_utc']}",
        f"- Cases: `{payload['cases_path']}`",
        f"- Results: `{payload['results_dir']}`",
        f"- Images: `{payload['images_dir'] or 'not supplied'}`",
        f"- Status: **{payload['status']}**"
        + ("" if payload["code"] is None else f" (`{payload['code']}`)"),
        f"- Cases: {summary['n_cases']}; passed: {summary['n_passed']}; "
        f"failed: {summary['n_failed']}; pass rate: {pass_rate_text}",
        "",
        "## Scope",
        "",
        "A case passes only when both the observed refusal code and the expected "
        "category match. Synthetic and real cases are counted separately. A "
        "synthetic fixture proves software behaviour, not real-world robustness.",
        "",
        "## Required category coverage",
        "",
    ]
    coverage_rows = []
    for category in payload["categories"]:
        coverage_rows.append(
            [
                category["category"],
                category["n_cases"],
                category["n_passed"],
                category["n_failed"],
                "yes" if category["covered"] else "no",
                category["note"] or "",
            ]
        )
    lines.extend(
        md_table(
            ["category", "cases", "passed", "failed", "covered", "note"],
            coverage_rows,
        )
    )
    lines.extend(["", "## Cases", ""])
    case_rows = []
    for item in payload["cases"]:
        case_rows.append(
            [
                item["case_id"],
                item["stem"],
                item["expected_code"],
                item["expected_category"],
                item["observed_code"],
                item["observed_category"],
                "yes" if item["passed"] else "no",
                "yes" if item["synthetic"] is True else ("no" if item["synthetic"] is False else "unknown"),
                item["notes"],
            ]
        )
    lines.extend(
        md_table(
            [
                "case",
                "stem",
                "expected code",
                "expected category",
                "observed code",
                "observed category",
                "passed",
                "synthetic",
                "notes",
            ],
            case_rows,
        )
    )
    lines.extend(["", "## Coverage gaps", ""])
    if summary["coverage_gaps"]:
        for category in summary["coverage_gaps"]:
            lines.append(f"- `{category}`: no case covers this category.")
    else:
        lines.append("- Every required category has at least one case.")
    lines.extend(["", "## Evidence type", ""])
    synthetic_counts = summary["synthetic_counts"]
    lines.append(
        f"- synthetic cases: {synthetic_counts['synthetic']}; real cases: "
        f"{synthetic_counts['real']}; unlabelled: {synthetic_counts['unknown']}"
    )
    if synthetic_counts["real"] == 0:
        lines.append(
            "- No case is labelled real. This report is not real-world evidence."
        )
    if synthetic_counts["unknown"]:
        lines.append(
            "- Some cases are unlabelled; do not treat them as real-world evidence."
        )
    lines.extend(
        [
            "",
            "## Excessive tilt",
            "",
            "The current engine has no excessive-tilt threshold. This category is "
            "reported as not covered unless a saved result actually carries an "
            "`excessive_tilt` code. Do not claim tilt refusal from this report "
            "without such a result.",
            "",
        ]
    )
    write_markdown(out_path / "refusal_analysis.md", lines)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.refusal_analysis",
        description="Check saved inference results against expected refusal "
        "codes and categories, including category coverage gaps.",
    )
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--images", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/refusal"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = build_refusal_analysis(
            cases_path=args.cases,
            results_dir=args.results,
            images_dir=args.images,
            out_dir=args.out,
        )
    except EvaluationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    summary = payload["summary"]
    print(
        f"{payload['status'].upper()}: {summary['n_passed']}/{summary['n_cases']} "
        f"case(s) matched; coverage gaps: "
        f"{', '.join(summary['coverage_gaps']) or 'none'}"
    )
    print(f"json: {args.out / 'refusal_analysis.json'}")
    print(f"csv: {args.out / 'refusal_cases.csv'}")
    print(f"markdown: {args.out / 'refusal_analysis.md'}")
    if payload["status"] == "unavailable":
        return 3
    return 0 if summary["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
