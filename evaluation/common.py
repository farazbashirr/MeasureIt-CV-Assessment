"""Shared, strict helpers for the Phase 4A evaluation tools.

Everything here is deliberately conservative: CSV schemas are checked, numbers
must be finite, ground truth must name the instrument that produced it, and
report writers store JSON atomically. Nothing in this module generates an
image, a measurement or a ground-truth value.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from dataset.splits import image_for_stem


class EvaluationError(RuntimeError):
    """A report input is missing, malformed or internally inconsistent."""


@dataclass(frozen=True)
class GroundTruth:
    stem: str
    width_mm: float
    height_mm: float
    instrument: str
    measured_utc: str = ""
    notes: str = ""


@dataclass(frozen=True)
class ResultView:
    """A normalised view of one saved inference result."""

    stem: str
    result_path: Path | None
    payload: dict[str, Any] | None
    status: str
    raw_code: str | None
    code: str
    message: str
    reliability: str | None
    width_mm: float | None
    height_mm: float | None
    ok: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def require_file(path: str | Path, what: str) -> Path:
    source = Path(path)
    if not source.is_file():
        raise EvaluationError(f"{what} not found: {source}")
    return source


def require_dir(path: str | Path, what: str) -> Path:
    source = Path(path)
    if not source.is_dir():
        raise EvaluationError(f"{what} not found: {source}")
    return source


def load_json(path: str | Path, *, what: str = "JSON file") -> dict[str, Any]:
    source = require_file(path, what)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvaluationError(f"cannot read {what} {source}: {exc}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"{what} {source} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvaluationError(f"{what} {source} must contain a JSON object")
    return payload


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, target)
    return target


def read_csv(
    path: str | Path,
    *,
    required: Sequence[str],
    optional: Sequence[str] = (),
    what: str = "CSV file",
) -> list[dict[str, str]]:
    """Read a strict CSV with an exact header.

    Unknown columns are rejected so that a typo cannot silently turn a measured
    field into an ignored one.
    """
    source = require_file(path, what)
    try:
        text = source.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise EvaluationError(f"cannot read {what} {source}: {exc}") from exc

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise EvaluationError(f"{what} {source} has no header row")

    fields = [str(field).strip() for field in reader.fieldnames]
    if len(fields) != len(set(fields)):
        raise EvaluationError(f"{what} {source} has duplicate column names")
    if len(fields) != len(reader.fieldnames):
        raise EvaluationError(f"{what} {source} has an unnamed column")

    missing = [name for name in required if name not in fields]
    if missing:
        raise EvaluationError(
            f"{what} {source} is missing column(s): {', '.join(missing)}"
        )
    allowed = set(required) | set(optional)
    unknown = [name for name in fields if name not in allowed]
    if unknown:
        raise EvaluationError(
            f"{what} {source} has unknown column(s): {', '.join(unknown)}"
        )

    rows: list[dict[str, str]] = []
    for line_number, record in enumerate(reader, start=2):
        if None in record:
            raise EvaluationError(f"{what} {source}:{line_number}: too many columns")
        row: dict[str, str] = {}
        for key, value in record.items():
            if key is None:
                raise EvaluationError(
                    f"{what} {source}:{line_number}: unexpected extra column"
                )
            row[key.strip()] = "" if value is None else str(value).strip()
        for name in required:
            row.setdefault(name, "")
        rows.append(row)
    return rows


def parse_float(
    value: Any,
    field: str,
    *,
    positive: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    text = "" if value is None else str(value).strip()
    if not text:
        raise EvaluationError(f"{field} is empty")
    try:
        number = float(text)
    except ValueError as exc:
        raise EvaluationError(f"{field} must be a number, got {text!r}") from exc
    if not math.isfinite(number):
        raise EvaluationError(f"{field} must be finite, got {number!r}")
    if positive and number <= 0:
        raise EvaluationError(f"{field} must be positive, got {number!r}")
    if minimum is not None and number < minimum:
        raise EvaluationError(f"{field} must be >= {minimum}, got {number!r}")
    if maximum is not None and number > maximum:
        raise EvaluationError(f"{field} must be <= {maximum}, got {number!r}")
    return number


def parse_optional_float(value: Any) -> float | None:
    """Best-effort finite positive float for reading result files."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def parse_optional_bool(value: Any, field: str) -> bool | None:
    text = "" if value is None else str(value).strip().lower()
    if not text:
        return None
    if text in {"true", "yes", "1", "y", "synthetic"}:
        return True
    if text in {"false", "no", "0", "n", "real"}:
        return False
    raise EvaluationError(
        f"{field} must be empty, true/false, yes/no or 1/0, got {value!r}"
    )


def load_ground_truth(path: str | Path) -> dict[str, GroundTruth]:
    """Load manual instrument measurements.

    The instrument column is required: a ground-truth value without a named
    source is not evidence and must not be used as one.
    """
    rows = read_csv(
        path,
        required=("stem", "width_mm", "height_mm", "instrument"),
        optional=("measured_utc", "notes"),
        what="ground-truth CSV",
    )
    if not rows:
        raise EvaluationError("ground-truth CSV has no instances")

    truth: dict[str, GroundTruth] = {}
    for line_number, row in enumerate(rows, start=2):
        stem = row["stem"]
        if not stem:
            raise EvaluationError(f"ground-truth row {line_number}: stem is empty")
        if stem in truth:
            raise EvaluationError(f"ground-truth CSV repeats stem {stem!r}")
        instrument = row["instrument"]
        if not instrument:
            raise EvaluationError(
                f"ground-truth row {line_number} ({stem}): instrument is empty; "
                "record the measuring instrument that produced the value"
            )
        truth[stem] = GroundTruth(
            stem=stem,
            width_mm=parse_float(
                row["width_mm"], f"ground truth {stem} width_mm", positive=True
            ),
            height_mm=parse_float(
                row["height_mm"], f"ground truth {stem} height_mm", positive=True
            ),
            instrument=instrument,
            measured_utc=row.get("measured_utc", ""),
            notes=row.get("notes", ""),
        )
    return truth


def find_result(results_dir: str | Path | None, stem: str) -> Path | None:
    if results_dir is None:
        return None
    root = Path(results_dir)
    if not root.is_dir():
        return None
    for name in (f"{stem}_result.json", f"{stem}.json", f"{stem}_measurement.json"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def read_result(results_dir: str | Path | None, stem: str) -> ResultView:
    result_path = find_result(results_dir, stem)
    if result_path is None:
        return ResultView(
            stem=stem,
            result_path=None,
            payload=None,
            status="missing",
            raw_code="missing_result",
            code="missing_result",
            message="no saved inference result was found for this stem",
            reliability=None,
            width_mm=None,
            height_mm=None,
            ok=False,
        )

    payload = load_json(result_path, what=f"inference result for {stem}")
    status = str(payload.get("status", "")).strip().lower() or "unknown"
    raw_code = payload.get("code")
    raw_code = None if raw_code in (None, "") else str(raw_code)
    message = str(payload.get("message", ""))
    measurement = payload.get("measurement")
    reliability = None
    width_mm = height_mm = None
    if isinstance(measurement, dict):
        reliability_value = measurement.get("reliability")
        reliability = None if reliability_value is None else str(reliability_value)
        if status == "ok":
            width_mm = parse_optional_float(measurement.get("width_mm"))
            height_mm = parse_optional_float(measurement.get("height_mm"))

    if status == "ok" and (width_mm is None or height_mm is None):
        code = "invalid_result"
        ok = False
    elif status == "ok" and reliability == "low" and raw_code is None:
        code = "low_reliability"
        ok = True
    elif raw_code is not None:
        code = raw_code
        ok = status == "ok"
    else:
        code = "ok" if status == "ok" else status
        ok = status == "ok"

    return ResultView(
        stem=stem,
        result_path=result_path,
        payload=payload,
        status=status,
        raw_code=raw_code,
        code=code,
        message=message,
        reliability=reliability,
        width_mm=width_mm,
        height_mm=height_mm,
        ok=ok,
    )


def error_summary(
    signed_errors: Sequence[float], truths: Sequence[float]
) -> dict[str, Any]:
    """Absolute-error metrics with an explicitly stated definition.

    ``signed_errors`` are ``system - ground truth`` in millimetres. Percentage
    error is ``abs(error) / ground truth * 100``. All metrics are pooled over
    the rows supplied, so callers decide whether a row is a width or a height.
    """
    if len(signed_errors) != len(truths):
        raise EvaluationError("signed_errors and truths must have the same length")
    if not signed_errors:
        return {
            "n": 0,
            "mae_mm": None,
            "rmse_mm": None,
            "mean_percentage_error_pct": None,
            "max_abs_error_mm": None,
            "median_abs_error_mm": None,
            "mean_signed_error_mm": None,
            "min_signed_error_mm": None,
            "max_signed_error_mm": None,
        }

    errors = np.asarray(signed_errors, dtype=np.float64)
    truth_values = np.asarray(truths, dtype=np.float64)
    if not np.isfinite(errors).all():
        raise EvaluationError("error list contains non-finite values")
    if not np.isfinite(truth_values).all() or (truth_values <= 0).any():
        raise EvaluationError("ground-truth values must be finite and positive")

    absolute = np.abs(errors)
    percentage = 100.0 * absolute / truth_values
    return {
        "n": int(errors.size),
        "mae_mm": float(absolute.mean()),
        "rmse_mm": float(np.sqrt(np.mean(np.square(errors)))),
        "mean_percentage_error_pct": float(percentage.mean()),
        "max_abs_error_mm": float(absolute.max()),
        "median_abs_error_mm": float(np.median(absolute)),
        "mean_signed_error_mm": float(errors.mean()),
        "min_signed_error_mm": float(errors.min()),
        "max_signed_error_mm": float(errors.max()),
    }


def latency_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "samples": 0,
            "median_s": None,
            "p95_s": None,
            "mean_s": None,
            "min_s": None,
            "max_s": None,
            "stdev_s": None,
        }
    timings = np.asarray(values, dtype=np.float64)
    if not np.isfinite(timings).all() or (timings < 0).any():
        raise EvaluationError("latency values must be finite and non-negative")
    return {
        "samples": int(timings.size),
        "median_s": float(np.median(timings)),
        "p95_s": float(np.percentile(timings, 95)),
        "mean_s": float(timings.mean()),
        "min_s": float(timings.min()),
        "max_s": float(timings.max()),
        "stdev_s": float(timings.std(ddof=0)),
    }


def md_cell(value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value).replace("|", "\\|").replace("\n", " ")


def md_table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(md_cell(cell) for cell in header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(md_cell(cell) for cell in row) + " |")
    return lines


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_csv(path: str | Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return target


def write_markdown(path: str | Path, lines: Iterable[str]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def image_present(images_dir: str | Path | None, stem: str) -> bool | None:
    if images_dir is None:
        return None
    return image_for_stem(images_dir, stem) is not None
