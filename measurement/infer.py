"""CPU inference CLI: image in, millimetres out (or an explicit refusal).

Pipeline: load image -> load real camera calibration -> undistort -> segment
with the checkpoint -> detect the ArUco reference -> rectify with the marker's
homography -> oriented minimum-area rectangle -> millimetres. The command
refuses to emit a valid measurement when the camera calibration, the measured
reference size, or the checkpoint is missing, and refuses image-level results
when the reference or object is absent, occluded, or geometrically invalid.

Exit codes:
    0  a measurement was produced
    2  usage error (unreadable image, unwritable output)
    3  unavailable physical data or artifact (calibration/reference/checkpoint)
    4  image-level rejection (missing reference/object, occlusion, geometry)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np
import torch

from measureit.camera import CameraModel, CameraModelError
from measureit.io import ImageLoadError, load_image
from measurement.engine import MeasurementResult, measure
from measurement.reference import ReferenceError, ReferenceSpec
from segmentation.model import IMAGENET_MEAN, IMAGENET_STD
from segmentation.train import load_checkpoint

INFERENCE_SCHEMA = "measureit.inference/1"
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_UNAVAILABLE = 3
EXIT_REJECTED = 4

SegmentFunction = Callable[[np.ndarray], np.ndarray | None]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class FrameAnalysis:
    result: MeasurementResult
    mask: np.ndarray | None
    timings: dict[str, float]
    undistorted: np.ndarray


@dataclass
class InferenceOutput:
    payload: dict[str, Any]
    overlay: np.ndarray | None
    exit_code: int
    result_path: Path | None
    overlay_path: Path | None


def preprocess(image_bgr: np.ndarray, image_size: int, mean, std) -> torch.Tensor:
    resized = cv2.resize(
        image_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA
    )
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None]


def predict_probability(
    model: torch.nn.Module,
    image_bgr: np.ndarray,
    *,
    image_size: int,
    mean,
    std,
) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        logits = model(preprocess(image_bgr, image_size, mean, std))
        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return cv2.resize(
        probability,
        (image_bgr.shape[1], image_bgr.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )


def _expand_bbox(
    rect: tuple[int, int, int, int],
    width: int,
    height: int,
    *,
    fraction: float = 0.2,
    minimum: int = 16,
) -> tuple[int, int, int, int]:
    x, y, box_width, box_height = rect
    pad_x = max(minimum, int(round(box_width * fraction)))
    pad_y = max(minimum, int(round(box_height * fraction)))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(width, x + box_width + pad_x)
    y1 = min(height, y + box_height + pad_y)
    return x0, y0, x1 - x0, y1 - y0


def segment_with_model(
    model: torch.nn.Module,
    image_bgr: np.ndarray,
    *,
    image_size: int,
    mean,
    std,
    refine: bool = True,
) -> np.ndarray:
    """Coarse 512-square prediction, then a full-resolution crop refinement."""
    probability = predict_probability(
        model, image_bgr, image_size=image_size, mean=mean, std=std
    )
    mask = np.where(probability >= 0.5, 255, 0).astype(np.uint8)
    if not refine or not mask.any():
        return mask

    height, width = mask.shape[:2]
    x, y, box_width, box_height = _expand_bbox(
        cv2.boundingRect(mask), width, height
    )
    crop = image_bgr[y : y + box_height, x : x + box_width]
    crop_probability = predict_probability(
        model, crop, image_size=image_size, mean=mean, std=std
    )
    crop_mask = np.where(crop_probability >= 0.5, 255, 0).astype(np.uint8)
    if crop_mask.any():
        mask[y : y + box_height, x : x + box_width] = crop_mask
    return mask


def analyze_frame(
    image_bgr: np.ndarray,
    *,
    camera: CameraModel,
    reference: ReferenceSpec,
    segment_fn: SegmentFunction,
) -> FrameAnalysis:
    """Undistort, segment, and measure one frame with an injected segmenter."""
    started = time.perf_counter()
    undistorted = camera.undistort(image_bgr)
    after_undistort = time.perf_counter()
    mask = segment_fn(undistorted)
    after_segment = time.perf_counter()
    result = measure(undistorted, mask, reference)
    after_measure = time.perf_counter()
    timings = {
        "undistort": after_undistort - started,
        "segment": after_segment - after_undistort,
        "measure": after_measure - after_segment,
        "analysis_total": after_measure - started,
    }
    return FrameAnalysis(
        result=result,
        mask=None if mask is None else np.asarray(mask),
        timings=timings,
        undistorted=undistorted,
    )


def draw_overlay(
    image_bgr: np.ndarray, result: MeasurementResult, reference: ReferenceSpec
) -> np.ndarray:
    overlay = image_bgr.copy()
    height, width = overlay.shape[:2]
    thickness = max(2, int(round(min(height, width) / 400)))
    scale = max(0.6, min(height, width) / 1200.0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    if result.marker is not None:
        corners = np.round(result.marker.corners_px).astype(np.int32)
        cv2.polylines(overlay, [corners], True, (255, 160, 0), thickness)
        label = f"reference {float(reference.marker_size_mm):.1f} mm"
        origin = (
            max(10, int(corners[:, 0].min())),
            max(24, int(corners[:, 1].min()) - 10),
        )
        cv2.putText(overlay, label, origin, font, scale, (0, 0, 0), thickness + 2)
        cv2.putText(overlay, label, origin, font, scale, (255, 160, 0), thickness)

    if result.contour_px is not None:
        contour = np.round(result.contour_px).astype(np.int32).reshape(-1, 1, 2)
        cv2.drawContours(overlay, [contour], -1, (0, 200, 0), thickness)

    if result.box_corners_px is not None:
        box = np.round(result.box_corners_px).astype(np.int32)
        cv2.polylines(overlay, [box], True, (0, 0, 255), thickness)
        if result.ok:
            label = f"{result.width_mm:.2f} x {result.height_mm:.2f} mm"
            origin = (
                max(10, int(box[:, 0].min())),
                max(24, int(box[:, 1].min()) - 10),
            )
            cv2.putText(overlay, label, origin, font, scale * 1.15, (0, 0, 0), thickness + 2)
            cv2.putText(overlay, label, origin, font, scale * 1.15, (0, 0, 255), thickness)

    status = result.status.upper()
    if result.code:
        status = f"{status}: {result.code}"
    cv2.putText(overlay, status, (10, 32), font, scale, (0, 0, 0), thickness + 2)
    cv2.putText(overlay, status, (10, 32), font, scale, (255, 255, 255), thickness)
    return overlay


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _base_payload(
    image_path: Path, width: int, height: int
) -> dict[str, Any]:
    return {
        "schema": INFERENCE_SCHEMA,
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
        "segmentation": None,
        "measurement": None,
        "timings_s": {},
        "notes": [
            "a single planar reference marker supplies the metric scale; marker "
            "and object must be coplanar and one view cannot verify that",
            "reliability is a rule-based geometry grade, not a probability",
        ],
    }


def run_inference(
    *,
    image_path: str | Path,
    camera_path: str | Path | None,
    reference_path: str | Path | None,
    checkpoint_path: str | Path | None,
    out_dir: str | Path,
    refine: bool = True,
    write: bool = True,
) -> InferenceOutput:
    started = time.perf_counter()
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    stem = image_path.stem
    image = load_image(image_path)
    height, width = image.shape[:2]
    payload = _base_payload(image_path, width, height)

    def finish(
        status: str,
        code: str | None,
        message: str,
        *,
        exit_code: int,
        overlay: np.ndarray | None = None,
    ) -> InferenceOutput:
        payload["status"] = status
        payload["code"] = code
        payload["message"] = message
        payload["timings_s"]["total"] = round(time.perf_counter() - started, 6)
        result_path = overlay_path = None
        if write:
            out_dir.mkdir(parents=True, exist_ok=True)
            result_path = out_dir / f"{stem}_result.json"
            _atomic_json(result_path, payload)
            if overlay is not None:
                overlay_path = out_dir / f"{stem}_overlay.png"
                if not cv2.imwrite(str(overlay_path), overlay):
                    raise OSError(f"cannot write overlay: {overlay_path}")
        return InferenceOutput(
            payload=payload,
            overlay=overlay,
            exit_code=exit_code,
            result_path=result_path,
            overlay_path=overlay_path,
        )

    # 1. Real camera calibration is required: no calibration, no millimetres.
    camera_file = None if camera_path is None else Path(camera_path)
    if camera_file is None or not camera_file.is_file():
        return finish(
            "unavailable",
            "missing_calibration",
            f"camera calibration not found: {camera_file}; run the calibration stage first",
            exit_code=EXIT_UNAVAILABLE,
        )
    try:
        camera = CameraModel.load(camera_file)
    except CameraModelError as exc:
        return finish(
            "unavailable", "invalid_calibration", str(exc), exit_code=EXIT_UNAVAILABLE
        )
    payload["camera"] = {
        "path": str(camera_file),
        "image_size": [int(camera.image_size[0]), int(camera.image_size[1])],
        "rms_reprojection_px": float(camera.rms_reprojection_px),
        "n_views_used": int(camera.n_views_used),
        "distortion_model": f"opencv-radtan-{camera.dist_coeffs.size}",
    }
    if (width, height) != camera.image_size:
        return finish(
            "unavailable",
            "calibration_image_size_mismatch",
            f"image is {width}x{height} but the camera was calibrated at "
            f"{camera.image_size[0]}x{camera.image_size[1]}",
            exit_code=EXIT_UNAVAILABLE,
        )

    # 2. A measured printed reference size is required.
    reference_file = None if reference_path is None else Path(reference_path)
    if reference_file is None or not reference_file.is_file():
        return finish(
            "unavailable",
            "missing_reference_config",
            f"reference configuration not found: {reference_file}",
            exit_code=EXIT_UNAVAILABLE,
        )
    try:
        reference = ReferenceSpec.load(reference_file)
    except ReferenceError as exc:
        return finish(
            "unavailable",
            "invalid_reference_config",
            str(exc),
            exit_code=EXIT_UNAVAILABLE,
        )
    payload["reference"] = {
        "path": str(reference_file),
        "dictionary": reference.dictionary,
        "marker_id": int(reference.marker_id),
        "marker_size_mm": reference.marker_size_mm,
        "usable": bool(reference.usable),
        "notes": reference.notes,
    }
    if not reference.usable:
        return finish(
            "unavailable",
            "reference_size_unknown",
            "marker_size_mm is null: print the marker, measure its outer black "
            "square, and set the value in the reference file",
            exit_code=EXIT_UNAVAILABLE,
        )

    # 3. A trained checkpoint is required: no untrained model may measure.
    checkpoint_file = None if checkpoint_path is None else Path(checkpoint_path)
    if checkpoint_file is None or not checkpoint_file.is_file():
        return finish(
            "unavailable",
            "missing_checkpoint",
            f"checkpoint not found: {checkpoint_file}; train the model first",
            exit_code=EXIT_UNAVAILABLE,
        )
    try:
        model, meta = load_checkpoint(checkpoint_file)
    except Exception as exc:  # noqa: BLE001 - artifacts fail in loader-specific ways
        return finish(
            "unavailable", "invalid_checkpoint", str(exc), exit_code=EXIT_UNAVAILABLE
        )
    architecture = meta.get("architecture", {})
    payload["checkpoint"] = {
        "path": str(checkpoint_file),
        "architecture": architecture.get("name"),
        "image_size": meta.get("image_size"),
        "epoch": meta.get("epoch"),
        "torch_version": meta.get("torch_version"),
    }

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

    analysis = analyze_frame(
        image,
        camera=camera,
        reference=reference,
        segment_fn=segment,
    )
    payload["segmentation"] = {
        "method": architecture.get("name", "unet_mobilenetv3_small"),
        "input_size": image_size,
        "refine": bool(refine),
        "mask_area_px": analysis.result.mask_area_px,
        "mask_area_pct": analysis.result.mask_area_pct,
    }
    payload["measurement"] = analysis.result.to_dict()
    payload["timings_s"].update(
        {name: round(value, 6) for name, value in analysis.timings.items()}
    )
    overlay = draw_overlay(analysis.undistorted, analysis.result, reference)

    if analysis.result.ok:
        return finish(
            "ok", None, analysis.result.message, exit_code=EXIT_OK, overlay=overlay
        )
    return finish(
        "rejected",
        analysis.result.code,
        analysis.result.message,
        exit_code=EXIT_REJECTED,
        overlay=overlay,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m measurement.infer",
        description="Measure one image in millimetres using a calibrated camera "
        "and one printed ArUco reference marker.",
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
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="skip the full-resolution crop refinement pass",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = run_inference(
            image_path=args.image,
            camera_path=args.camera,
            reference_path=args.reference,
            checkpoint_path=args.checkpoint,
            out_dir=args.out,
            refine=not args.no_refine,
        )
    except (OSError, ValueError, ImageLoadError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_USAGE

    payload = output.payload
    if output.result_path is not None:
        print(f"json: {output.result_path}")
    if output.overlay_path is not None:
        print(f"overlay: {output.overlay_path}")

    if payload["status"] == "ok":
        measurement = payload["measurement"]
        print(
            f"OK: {measurement['width_mm']:.2f} x {measurement['height_mm']:.2f} mm "
            f"({measurement['reliability']} reliability)"
        )
    else:
        print(f"{payload['status'].upper()}: {payload['message']}")
    return output.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
