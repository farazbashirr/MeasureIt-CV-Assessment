"""The frozen 100-image split manifest.

Nothing here writes ``dataset/splits.csv``. The pre-registered SHA-256 is
verified before parsing, so an accidental edit stops every dataset command
instead of silently changing the experiment.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_SPLITS = Path("dataset/splits.csv")
FROZEN_SHA256 = "1D726C514B6D946F78607B9864C24567E787C88AF7DBF515BBCE3A57FE3DD75C"

HEADER = ["stem", "split", "session", "condition"]
EXPECTED_SPLIT_COUNTS = {"train": 70, "val": 20, "test": 10}
EXPECTED_SESSIONS = {f"S{index:02d}" for index in range(1, 11)}
IMAGES_PER_SESSION = 10
STEM_PATTERN = re.compile(r"^S\d{2}_\d{3}$")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


class SplitError(RuntimeError):
    """The split manifest is missing, altered or structurally invalid."""


@dataclass(frozen=True)
class SplitRow:
    stem: str
    split: str
    session: str
    condition: str


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def _parse(source: Path) -> list[SplitRow]:
    text = source.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != HEADER:
        raise SplitError(
            f"{source} has header {reader.fieldnames!r}; expected exactly {HEADER!r}"
        )
    rows: list[SplitRow] = []
    for line_number, record in enumerate(reader, start=2):
        try:
            rows.append(
                SplitRow(
                    stem=record["stem"].strip(),
                    split=record["split"].strip(),
                    session=record["session"].strip(),
                    condition=record["condition"].strip(),
                )
            )
        except (KeyError, AttributeError) as exc:
            raise SplitError(f"{source}:{line_number}: malformed row") from exc
    return rows


def validate_rows(rows: Iterable[SplitRow]) -> None:
    """Structural checks that must hold for the frozen manifest."""
    rows = list(rows)
    if len(rows) != sum(EXPECTED_SPLIT_COUNTS.values()):
        raise SplitError(
            f"expected {sum(EXPECTED_SPLIT_COUNTS.values())} rows, found {len(rows)}"
        )

    stems = [row.stem for row in rows]
    duplicates = sorted(stem for stem, count in Counter(stems).items() if count > 1)
    if duplicates:
        raise SplitError(f"duplicate stems: {', '.join(duplicates[:10])}")

    split_counts = Counter(row.split for row in rows)
    if dict(split_counts) != EXPECTED_SPLIT_COUNTS:
        raise SplitError(
            f"split counts are {dict(split_counts)}, expected {EXPECTED_SPLIT_COUNTS}"
        )

    session_splits: dict[str, set[str]] = defaultdict(set)
    session_counts: Counter[str] = Counter()
    for row in rows:
        if not STEM_PATTERN.fullmatch(row.stem):
            raise SplitError(f"stem {row.stem!r} does not match S01_001")
        if row.split not in EXPECTED_SPLIT_COUNTS:
            raise SplitError(f"stem {row.stem}: unknown split {row.split!r}")
        if not row.session:
            raise SplitError(f"stem {row.stem}: empty session")
        if not row.stem.startswith(f"{row.session}_"):
            raise SplitError(
                f"stem {row.stem}: session {row.session!r} disagrees with the stem"
            )
        if not row.condition:
            raise SplitError(f"stem {row.stem}: empty condition")
        session_splits[row.session].add(row.split)
        session_counts[row.session] += 1

    if set(session_counts) != EXPECTED_SESSIONS:
        raise SplitError(
            f"sessions are {sorted(session_counts)}, expected {sorted(EXPECTED_SESSIONS)}"
        )

    wrong = {
        session: count
        for session, count in session_counts.items()
        if count != IMAGES_PER_SESSION
    }
    if wrong:
        raise SplitError(f"sessions without exactly {IMAGES_PER_SESSION} images: {wrong}")

    leaks = {
        session: sorted(splits)
        for session, splits in session_splits.items()
        if len(splits) != 1
    }
    if leaks:
        raise SplitError(f"sessions split across splits: {leaks}")


def load_splits(path: str | Path = DEFAULT_SPLITS) -> list[SplitRow]:
    source = Path(path)
    if not source.is_file():
        raise SplitError(f"split manifest not found: {source}")

    digest = file_sha256(source)
    if digest != FROZEN_SHA256:
        raise SplitError(
            f"{source} has SHA-256 {digest}, but the frozen manifest is {FROZEN_SHA256}. "
            "The split must not change; restore the committed file."
        )

    rows = _parse(source)
    validate_rows(rows)
    return rows


def by_stem(rows: Iterable[SplitRow]) -> dict[str, SplitRow]:
    return {row.stem: row for row in rows}


def session_map(rows: Iterable[SplitRow]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in rows:
        mapping.setdefault(row.session, row.split)
    return mapping


def image_for_stem(images_dir: str | Path, stem: str) -> Path | None:
    """Return the processed image for a stem, whatever its extension."""
    root = Path(images_dir)
    for suffix in IMAGE_SUFFIXES:
        candidate = root / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None
