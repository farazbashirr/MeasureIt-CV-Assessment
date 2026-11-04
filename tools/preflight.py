"""Read-only preflight: what this machine can do, and what is still missing.

Run it before every phase. It changes nothing.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from importlib import metadata
from pathlib import Path

import numpy as np

REQUIRED = ["opencv-contrib-python", "numpy", "reportlab", "pypdf", "pytest"]
OPTIONAL = ["torch", "torchvision", "segmentation-models-pytorch", "albumentations", "timm"]


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _ram_gb() -> float | None:
    try:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return round(status.ullTotalPhys / 1024**3, 1)
    except Exception:
        return None


def collect() -> dict:
    import cv2

    board_spec = Path("calibration/board/board_spec.json")
    camera_file = Path("calibration/params/camera.json")
    image_dir = Path("calibration/images")

    board = None
    if board_spec.is_file():
        from calibration.board import BoardError, load_spec

        try:
            spec = load_spec(board_spec)
            board = {
                "spec": str(board_spec),
                "squares": f"{spec.squares_x}x{spec.squares_y}",
                "nominal_square_mm": spec.square_mm_nominal,
                "measured_square_mm": (
                    None
                    if spec.measured_square_x_mm is None
                    else round(spec.measured_square_mm, 3)
                ),
                "instrument": spec.measurement_instrument,
            }
        except BoardError as exc:
            board = {"spec": str(board_spec), "error": str(exc)}

    images = []
    if image_dir.is_dir():
        images = sorted(
            p.name
            for p in image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    disk = shutil.disk_usage(".")
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "cpu_count": __import__("os").cpu_count(),
        "ram_gb": _ram_gb(),
        "disk_free_gb": round(disk.free / 1024**3, 1),
        "packages": {name: _version(name) for name in REQUIRED + OPTIONAL},
        "opencv": {
            "version": cv2.__version__,
            "aruco": hasattr(cv2, "aruco"),
            "charuco_detector": hasattr(cv2.aruco, "CharucoDetector"),
        },
        "board": board,
        "calibration_images": images,
        "camera_model": camera_file.is_file(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only environment check")
    parser.add_argument("--json", action="store_true", help="print machine-readable output")
    args = parser.parse_args(argv)

    data = collect()
    if args.json:
        print(json.dumps(data, indent=2))
        return 0

    print("== Machine ==")
    print(f"  platform : {data['platform']}")
    print(f"  python   : {data['python']} ({data['executable']})")
    print(f"  cpu      : {data['cpu_count']} logical cores")
    print(f"  ram      : {data['ram_gb']} GB")
    print(f"  disk free: {data['disk_free_gb']} GB")

    print("== Packages ==")
    for name, version in data["packages"].items():
        print(f"  {name:32s} {version or 'not installed'}")
    opencv = data["opencv"]
    print(f"  opencv aruco: {opencv['aruco']}  charuco detector: {opencv['charuco_detector']}")

    print("== Calibration readiness ==")
    board = data["board"]
    if board is None:
        print("  board spec : missing (run calibration.make_board generate)")
    elif "error" in board:
        print(f"  board spec : ERROR {board['error']}")
    else:
        print(f"  board spec : {board['squares']} squares, nominal {board['nominal_square_mm']} mm")
        if board["measured_square_mm"] is None:
            print("  measured   : NOT RECORDED - calibration is blocked (physical task)")
        else:
            print(
                f"  measured   : {board['measured_square_mm']} mm "
                f"({board['instrument']})"
            )
    print(f"  images     : {len(data['calibration_images'])} in calibration/images")
    print(f"  camera.json: {'present' if data['camera_model'] else 'not produced yet'}")

    print("== Still physical ==")
    print("  - print and mount the board, measure the printed square")
    print("  - capture 30+ ChArUco views with the locked Pixel 8 Pro")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
