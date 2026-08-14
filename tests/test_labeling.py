"""Tests for the human-label loader that feeds judge calibration."""

from __future__ import annotations

from pathlib import Path

from trustgate.labeling import load_labels


def test_load_labels_reads_filled_rows_only(tmp_path: Path) -> None:
    csv = tmp_path / "sheet.csv"
    csv.write_text(
        "item_id,adjudicated_pass,notes\n"
        "a,1,ok\n"
        "b,0,ok\n"
        "c,,not yet labeled\n"      # blank -> skipped
        "d,x,bad value\n",          # non 0/1 -> skipped
        encoding="utf-8",
    )
    labels = load_labels(csv)
    assert labels == {"a": 1, "b": 0}


def test_load_labels_custom_column(tmp_path: Path) -> None:
    csv = tmp_path / "sheet.csv"
    csv.write_text("item_id,rater_A_pass\ng1,1\ng2,0\n", encoding="utf-8")
    assert load_labels(csv, column="rater_A_pass") == {"g1": 1, "g2": 0}
