import shutil

import pytest

from dataset.splits import (
    DEFAULT_SPLITS,
    EXPECTED_SPLIT_COUNTS,
    FROZEN_SHA256,
    SplitError,
    SplitRow,
    file_sha256,
    image_for_stem,
    load_splits,
    session_map,
    validate_rows,
)


def _valid_rows() -> list[SplitRow]:
    rows: list[SplitRow] = []
    for index in range(1, 11):
        session = f"S{index:02d}"
        split = "train" if index <= 7 else ("val" if index <= 9 else "test")
        for shot in range(1, 11):
            rows.append(SplitRow(f"{session}_{shot:03d}", split, session, "condition"))
    return rows


def test_frozen_manifest_is_unchanged():
    assert file_sha256(DEFAULT_SPLITS) == FROZEN_SHA256
    rows = load_splits()
    assert len(rows) == 100
    counts = {
        split: sum(1 for row in rows if row.split == split)
        for split in EXPECTED_SPLIT_COUNTS
    }
    assert counts == EXPECTED_SPLIT_COUNTS


def test_sessions_never_cross_splits():
    rows = load_splits()
    mapping = session_map(rows)
    assert mapping == {
        "S01": "train",
        "S02": "train",
        "S03": "train",
        "S04": "train",
        "S05": "train",
        "S06": "train",
        "S07": "train",
        "S08": "val",
        "S09": "val",
        "S10": "test",
    }
    for session in mapping:
        splits = {row.split for row in rows if row.session == session}
        assert len(splits) == 1


def test_tampered_manifest_is_refused(tmp_path):
    copy = tmp_path / "splits.csv"
    shutil.copyfile(DEFAULT_SPLITS, copy)
    copy.write_bytes(copy.read_bytes() + b"\r\n")
    with pytest.raises(SplitError, match="SHA-256"):
        load_splits(copy)


def test_session_leak_is_refused():
    rows = _valid_rows()
    train_row = next(index for index, row in enumerate(rows) if row.session == "S07")
    val_row = next(index for index, row in enumerate(rows) if row.session == "S08")
    rows[train_row] = SplitRow(rows[train_row].stem, "val", "S07", "condition")
    rows[val_row] = SplitRow(rows[val_row].stem, "train", "S08", "condition")
    with pytest.raises(SplitError, match="split across splits"):
        validate_rows(rows)


def test_wrong_row_count_is_refused():
    with pytest.raises(SplitError, match="expected 100 rows"):
        validate_rows(_valid_rows()[:-1])


def test_duplicate_stem_is_refused():
    rows = _valid_rows()
    rows[1] = SplitRow(rows[0].stem, rows[1].split, rows[1].session, rows[1].condition)
    with pytest.raises(SplitError, match="duplicate stems"):
        validate_rows(rows)


def test_bad_stem_shape_is_refused():
    rows = _valid_rows()
    rows[0] = SplitRow("frame-1", rows[0].split, rows[0].session, rows[0].condition)
    with pytest.raises(SplitError, match="does not match"):
        validate_rows(rows)


def test_image_for_stem_finds_either_extension(tmp_path):
    assert image_for_stem(tmp_path, "S01_001") is None
    png = tmp_path / "S01_001.png"
    png.write_bytes(b"x")
    assert image_for_stem(tmp_path, "S01_001") == png
    jpg = tmp_path / "S01_002.jpg"
    jpg.write_bytes(b"x")
    assert image_for_stem(tmp_path, "S01_002") == jpg
