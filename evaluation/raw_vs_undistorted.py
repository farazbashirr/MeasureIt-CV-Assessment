"""Raw vs undistorted comparison experiment.

The command loads one real image and a real camera calibration, produces the
raw and undistorted frames, segments and measures both where the physical
artifacts exist, and writes the overlays plus a structured difference report.

If the calibration, measured reference size or checkpoint is missing, the
report says so and no millimetre difference is invented. The difference is
defined as ``undistorted - raw``: a positive value means the undistorted
measurement is larger.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from evaluation.common import EvaluationError, utc_now, write_json
from measureit.camera import CameraModel, CameraModelError
from measureit.io import ImageLoadError, load_image
from measurement.engine import COPLANARITY, MeasurementResult, measure
from measurement.infer import (
    EXIT_OK,
    EXIT_REJECTED,
    EXIT_UNAVAILABLE,
    EXIT_USAGE,
    draw_overlay,
    segment_with_model,
)
from measurement.reference import ReferenceError, ReferenceSpec
from segmentation.model import IMAGENET_MEAN, IMAGENET_STD
from segmentation.train import load_checkpoint

COMPARISON_SCHEMA = "measureit.raw_vs_undistorted/1"

SegmentFunction = Callable[[np.ndarray], np.ndarray | None]


@dataclass
class FrameComparison:
    raw_result: MeasurementResult
    undistorted_result: MeasurementResult
    difference: dict[str, Any]
    undistorted: np.ndarray


@dataclass
class ComparisonOutput:
    payload: dict[str, Any]
    exit_code: int
    json_path: Path | None
    raw_path: Path | None
    undistorted_path: Path | None
    raw_overlay_path: Path | None
    undistorted_overlay_path: Path | None
    side_by_side_path: Path | None


def measurement_difference(
    raw: MeasurementResult, undistorted: MeasurementResult
) -> dict[str, Any]:
    """Compare two successful measurements, or say why no difference exists."""
    if not raw.ok or not undistorted.ok:
        return {
            "available": False,
            "reason": (
                "both the raw and undistorted measurements must succeed before "
                "a difference is meaningful"
            ),
            "raw": {"status": raw.status, "code": raw.code},
            "undistorted": {
                "status": undistorted.status,
                "code": undistorted.code,
            },
        }

    raw_width = float(raw.width_mm)
    raw_height = float(raw.height_mm)
    undistorted_width = float(undistorted.width_mm)
    undistorted_height = float(undistorted.height_mm)
    width_delta = undistorted_width - raw_width
    height_delta = undistorted_height - raw_height
    width_delta_pct = 100.0 * width_delta / raw_width
    height_delta_pct = 100.0 * height_delta / raw_height

    return {
        "available": True,
        "definition": (
            "undistorted minus raw in millimetres; positive means the "
            "undistorted measurement is larger"
        ),
        "width_delta_mm": width_delta,
        "height_delta_mm": height_delta,
        "width_delta_pct": width_delta_pct,
        "height_delta_pct": height_delta_pct,
        "max_abs_delta_mm": max(abs(width_delta), abs(height_delta)),
        "max_abs_delta_pct": max(abs(width_delta_pct), abs(height_delta_pct)),
    }


def compare_frame(
    image_bgr: np.ndarray,
    *,
    camera: CameraModel,
    reference: ReferenceSpec,
    segment_fn: SegmentFunction,
) -> FrameComparison:
    """Measure raw and undistorted frames with an injected segmenter."""
    raw_mask = segment_fn(image_bgr)
    raw_result = measure(image_bgr, raw_mask, reference)

    undistorted = camera.undistort(image_bgr)
    undistorted_mask = segment_fn(undistorted)
    undistorted_result = measure(undistorted, undistorted_mask, reference)

    return FrameComparison(
        raw_result=raw_result,
        undistorted_result=undistorted_result,
        difference=measurement_difference(raw_result, undistorted_result),
        undistorted=undistorted,
    )


def _unavailable_measurement(code: str, message: str) -> MeasurementResult:
    return MeasurementResult(
        status="unavailable",
        code=code,
        message=message,
        reliability="invalid",
        warnings=[COPLANARITY],
    )


def _base_payload(image_path: Path, width: int, height: int) -> dict[str, Any]:
    return {
        "schema": COMPARISON_SCHEMA,
        "created_utc": utc_now(),
        "status": None,
        "code": None,
        "message": "",
        "image": {
            "path": str(image_path),
            "width": int(width),
            "height": int(height),
        },
        "camera": None,
        "reference": None,
        "checkpoint": None,
        "images_available": False,
        "measurements_available": False,
        "measurements": {"raw": None, "undistorted": None},
        "difference": None,
        "images": {
            "raw": None,
            "undistorted": None,
            "raw_overlay": None,
            "undistorted_overlay": None,
            "side_by_side": None,
        },
        "timings_s": {},
        "notes": [
            "raw and undistorted frames are measured independently; the "
            "difference is undistorted minus raw",
            "a measurement still assumes the marker and object are coplanar; "
            "one view cannot verify that",
            "reliability is a rule-based geometry grade, not a probability",
        ],
    }


def _labelled(image: np.ndarray, text: str) -> np.ndarray:
    overlay = image.copy()
    height, width = overlay.shape[:2]
    scale = max(0.7, min(height, width) / 1100.0)
    thickness = max(2, int(round(min(height, width) / 500)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, scale, thickness
    )
    cv2.rectangle(
        overlay,
        (0, 0),
        (text_width + 16, text_height + baseline + 14),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        overlay,
        text,
        (8, text_height + 7),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return overlay


def side_by_side(
    raw: np.ndarray,
    undistorted: np.ndarray,
    *,
    raw_label: str = "raw",
    undistorted_label: str = "undistorted",
) -> np.ndarray:
    if raw.shape[:2] != undistorted.shape[:2]:
        raise EvaluationError(
            "raw and undistorted images must have the same size to compare them"
        )
    height, width = raw.shape[:2]
    gap = 12
    canvas = np.full((height, width * 2 + gap, 3), 255, dtype=np.uint8)
    canvas[:, :width] = _labelled(raw, raw_label)
    canvas[:, width + gap :] = _labelled(undistorted, undistorted_label)
    return canvas


def _write_png(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"cannot write image: {path}")


def run_comparison(
    *,
    image_path: str | Path,
    camera_path: str | Path | None,
    reference_path: str | Path | None,
    checkpoint_path: str | Path | None,
    out_dir: str | Path,
    refine: bool = True,
    write: bool = True,
) -> ComparisonOutput:
    started = time.perf_counter()
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    stem = image_path.stem

    try:
        image = load_image(image_path)
    except (ImageLoadError, OSError) as exc:
        raise EvaluationError(f"cannot read image {image_path}: {exc}") from exc

    height, width = image.shape[:2]
    payload = _base_payload(image_path, width, height)
    paths: dict[str, Path | None] = {name: None for name in payload["images"]}

    def finish(status: str, code: str | None, message: str, exit_code: int) -> ComparisonOutput:
        payload["status"] = status
        payload["code"] = code
        payload["message"] = message
        payload["timings_s"]["total"] = round(time.perf_counter() - started, 6)
        json_path = None
        if write:
            json_path = write_json(out_dir / f"{stem}_comparison.json", payload)
        return ComparisonOutput(
            payload=payload,
            exit_code=exit_code,
            json_path=json_path,
            raw_path=paths["raw"],
            undistorted_path=paths["undistorted"],
            raw_overlay_path=paths["raw_overlay"],
            undistorted_overlay_path=paths["undistorted_overlay"],
            side_by_side_path=paths["side_by_side"],
        )

    def save_images(
        *,
        undistorted: np.ndarray,
        raw_result: MeasurementResult | None = None,
        undistorted_result: MeasurementResult | None = None,
        reference: ReferenceSpec | None = None,
    ) -> None:
        payload["images_available"] = True
        if not write:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_path = out_dir / f"{stem}_raw.png"
        undistorted_path = out_dir / f"{stem}_undistorted.png"
        _write_png(raw_path, image)
        _write_png(undistorted_path, undistorted)
        paths["raw"] = raw_path
        paths["undistorted"] = undistorted_path
        payload["images"]["raw"] = str(raw_path)
        payload["images"]["undistorted"] = str(undistorted_path)

        if raw_result is not None and reference is not None:
            raw_overlay = draw_overlay(image, raw_result, reference)
            raw_overlay_path = out_dir / f"{stem}_raw_overlay.png"
            _write_png(raw_overlay_path, raw_overlay)
            paths["raw_overlay"] = raw_overlay_path
            payload["images"]["raw_overlay"] = str(raw_overlay_path)

        if undistorted_result is not None and reference is not None:
            undistorted_overlay = draw_overlay(
                undistorted, undistorted_result, reference
            )
            undistorted_overlay_path = out_dir / f"{stem}_undistorted_overlay.png"
            _write_png(undistorted_overlay_path, undistorted_overlay)
            paths["undistorted_overlay"] = undistorted_overlay_path
            payload["images"]["undistorted_overlay"] = str(undistorted_overlay_path)

        comparison_path = out_dir / f"{stem}_side_by_side.png"
        _write_png(
            comparison_path,
            side_by_side(
                image,
                undistorted,
                raw_label=f"raw ({width}x{height})",
                undistorted_label=f"undistorted ({width}x{height})",
            ),
        )
        paths["side_by_side"] = comparison_path
        payload["images"]["side_by_side"] = str(comparison_path)

    camera_file = None if camera_path is None else Path(camera_path)
    if camera_file is None or not camera_file.is_file():
        return finish(
            "unavailable",
            "missing_calibration",
            f"camera calibration not found: {camera_file}; run the calibration stage first",
            EXIT_UNAVAILABLE,
        )
    try:
        camera = CameraModel.load(camera_file)
    except CameraModelError as exc:
        return finish(
            "unavailable", "invalid_calibration", str(exc), EXIT_UNAVAILABLE
        )
    payload["camera"] = {
        "path": str(camera_file),
        "image_size": [int(camera.image_size[0]), int(camera.image_size[1])],
        "rms_reprojection_px": float(camera.rms_reprojection_px),
        "n_views_used": int(camera.n_views_used),
        "distortion_model": f"opencv-radtan-{camera.dist_coeffs.size}",
        "max_undistort_displacement_px": float(
            camera.max_undistort_displacement_px()
        ),
    }
    if (width, height) != camera.image_size:
        return finish(
            "unavailable",
            "calibration_image_size_mismatch",
            f"image is {width}x{height} but the camera was calibrated at "
            f"{camera.image_size[0]}x{camera.image_size[1]}",
            EXIT_UNAVAILABLE,
        )

    try:
        started_undistort = time.perf_counter()
        undistorted = camera.undistort(image)
        payload["timings_s"]["undistort"] = round(
            time.perf_counter() - started_undistort, 6
        )
    except CameraModelError as exc:
        return finish(
            "unavailable", "invalid_calibration", str(exc), EXIT_UNAVAILABLE
        )

    reference: ReferenceSpec | None = None
    reference_file = None if reference_path is None else Path(reference_path)
    reference_code: str | None = None
    reference_message = ""
    if reference_file is None or not reference_file.is_file():
        reference_code = "missing_reference_config"
        reference_message = f"reference configuration not found: {reference_file}"
    else:
        try:
            reference = ReferenceSpec.load(reference_file)
            payload["reference"] = {
                "path": str(reference_file),
                "dictionary": reference.dictionary,
                "marker_id": int(reference.marker_id),
                "marker_size_mm": reference.marker_size_mm,
                "usable": bool(reference.usable),
                "notes": reference.notes,
            }
            if not reference.usable:
                reference_code = "reference_size_unknown"
                reference_message = (
                    "marker_size_mm is null: print the marker, measure its outer "
                    "black square, and set the value in the reference file"
                )
        except ReferenceError as exc:
            reference_code = "invalid_reference_config"
            reference_message = str(exc)

    model = None
    meta: dict[str, Any] = {}
    checkpoint_file = None if checkpoint_path is None else Path(checkpoint_path)
    checkpoint_code: str | None = None
    checkpoint_message = ""
    if checkpoint_file is None or not checkpoint_file.is_file():
        checkpoint_code = "missing_checkpoint"
        checkpoint_message = (
            f"checkpoint not found: {checkpoint_file}; train the model first"
        )
    else:
        try:
            model, meta = load_checkpoint(checkpoint_file)
            payload["checkpoint"] = {
                "path": str(checkpoint_file),
                "architecture": meta.get("architecture", {}).get("name"),
                "image_size": meta.get("image_size"),
                "epoch": meta.get("epoch"),
            }
        except Exception as exc:  # noqa: BLE001 - artifacts fail in loader-specific ways
            checkpoint_code = "invalid_checkpoint"
            checkpoint_message = str(exc)

    unavailable_code = reference_code or checkpoint_code
    if unavailable_code is not None:
        unavailable_message = reference_message or checkpoint_message
        unavailable = _unavailable_measurement(
            unavailable_code, unavailable_message
        ).to_dict()
        payload["measurements"] = {
            "raw": unavailable,
            "undistorted": dict(unavailable),
        }
        payload["difference"] = {
            "available": False,
            "reason": unavailable_message,
        }
        save_images(undistorted=undistorted)
        return finish(
            "unavailable", unavailable_code, unavailable_message, EXIT_UNAVAILABLE
        )

    assert reference is not None
    assert model is not None
    image_size = int(meta.get("image_size", 512))
    normalization = meta.get("normalization", {})
    mean = normalization.get("mean", IMAGENET_MEAN)
    std = normalization.get("std", IMAGENET_STD)

    def segment(image_bgr: np.ndarray) -> np.ndarray:
        return segment_with_model(
            model,
            image_bgr,
            image_size=image_size,
            mean=mean,
            std=std,
            refine=refine,
        )

    started_raw = time.perf_counter()
    raw_mask = segment(image)
    payload["timings_s"]["raw_segment"] = round(
        time.perf_counter() - started_raw, 6
    )
    started_raw_measure = time.perf_counter()
    raw_result = measure(image, raw_mask, reference)
    payload["timings_s"]["raw_measure"] = round(
        time.perf_counter() - started_raw_measure, 6
    )

    started_undistorted = time.perf_counter()
    undistorted_mask = segment(undistorted)
    payload["timings_s"]["undistorted_segment"] = round(
        time.perf_counter() - started_undistorted, 6
    )
    started_undistorted_measure = time.perf_counter()
    undistorted_result = measure(undistorted, undistorted_mask, reference)
    payload["timings_s"]["undistorted_measure"] = round(
        time.perf_counter() - started_undistorted_measure, 6
    )

    difference = measurement_difference(raw_result, undistorted_result)
    payload["measurements"]["raw"] = raw_result.to_dict()
    payload["measurements"]["undistorted"] = undistorted_result.to_dict()
    payload["measurements_available"] = True
    payload["difference"] = difference
    save_images(
        undistorted=undistorted,
        raw_result=raw_result,
        undistorted_result=undistorted_result,
        reference=reference,
    )

    if raw_result.ok and undistorted_result.ok:
        return finish(
            "ok",
            None,
            "both raw and undistorted measurements succeeded",
            EXIT_OK,
        )

    code = raw_result.code or undistorted_result.code
    message = (
        f"raw: {raw_result.status}"
        f"{'' if raw_result.code is None else '/' + raw_result.code}; "
        f"undistorted: {undistorted_result.status}"
        f"{'' if undistorted_result.code is None else '/' + undistorted_result.code}"
    )
    return finish("rejected", code, message, EXIT_REJECTED)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.raw_vs_undistorted",
        description="Compare measurement on a raw frame and the same frame "
        "undistorted with a real camera calibration.",
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument(
        "--camera", type=Path, default=Path("calibration/params/camera.json")
    )
    parser.add_argument(
        "--reference", type=Path, default=Path("measurement/reference.json")
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("runs/seg/best.pt")
    )
    parser.add_argument("--out", type=Path, default=Path("results/raw_vs_undistorted"))
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="skip the full-resolution crop refinement pass",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = run_comparison(
            image_path=args.image,
            camera_path=args.camera,
            reference_path=args.reference,
            checkpoint_path=args.checkpoint,
            out_dir=args.out,
            refine=not args.no_refine,
        )
    except (EvaluationError, OSError, ValueError, ImageLoadError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_USAGE

    payload = output.payload
    if output.json_path is not None:
        print(f"json: {output.json_path}")
    if output.side_by_side_path is not None:
        print(f"side by side: {output.side_by_side_path}")

    if payload["status"] == "ok":
        raw = payload["measurements"]["raw"]
        undistorted = payload["measurements"]["undistorted"]
        difference = payload["difference"]
        print(
            f"OK: raw {raw['width_mm']:.2f} x {raw['height_mm']:.2f} mm, "
            f"undistorted {undistorted['width_mm']:.2f} x "
            f"{undistorted['height_mm']:.2f} mm, "
            f"max |delta| {difference['max_abs_delta_mm']:.3f} mm"
        )
    else:
        print(f"{payload['status'].upper()}: {payload['message']}")
    return output.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
