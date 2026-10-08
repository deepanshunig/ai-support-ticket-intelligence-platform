"""Fixture-only checks for reproducible dataset boundaries and input integrity."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ticket_intelligence.data.prepare import (
    audit_frame,
    load_categories,
    normalize_text,
    prepare_dataset,
    sha256_file,
    split_training_frame,
)


@pytest.fixture
def categories() -> list[str]:
    return ["second_intent", "first_intent"]


@pytest.fixture
def training_frame(categories: list[str]) -> pd.DataFrame:
    rows = []
    for category in categories:
        for group in range(30):
            text = f"{category} request {group:02d}"
            rows.extend(
                [
                    {"text": text, "category": category},
                    {"text": "  " + text.upper().replace(" ", "   ") + "\t", "category": category},
                ]
            )
    return pd.DataFrame(rows)


def test_grouped_splits_are_disjoint_complete_and_deterministic(
    training_frame: pd.DataFrame, categories: list[str]
) -> None:
    first = split_training_frame(training_frame, categories, seed=42)
    second = split_training_frame(training_frame, categories, seed=42)
    original_groups = {normalize_text(text) for text in training_frame["text"]}
    seen_groups: set[str] = set()
    seen_rows: set[str] = set()
    for name, records in first.items():
        pd.testing.assert_frame_equal(records, second[name])
        assert set(records["category"]) == set(categories)
        assert not seen_groups & set(records["group_hash"])
        assert not seen_rows & set(records["row_id"])
        seen_groups.update(records["group_hash"])
        seen_rows.update(records["row_id"])
    assert len(seen_groups) == len(original_groups)
    assert len(seen_rows) == len(training_frame)
    assert len(first["fit"]) > len(first["development"])


def test_unknown_label_is_rejected(training_frame: pd.DataFrame, categories: list[str]) -> None:
    frame = training_frame.copy()
    frame.loc[0, "category"] = "unexpected_intent"
    with pytest.raises(ValueError, match="unknown labels"):
        split_training_frame(frame, categories)


def test_missing_label_is_rejected(training_frame: pd.DataFrame, categories: list[str]) -> None:
    frame = training_frame.loc[training_frame["category"] == categories[0]].copy()
    with pytest.raises(ValueError, match="missing labels"):
        split_training_frame(frame, categories)


@pytest.mark.parametrize("text", ["", "   \t\n", None])
def test_empty_record_is_rejected(
    training_frame: pd.DataFrame, categories: list[str], text: str | None
) -> None:
    frame = training_frame.copy()
    frame.loc[0, "text"] = text
    with pytest.raises(ValueError, match="empty or invalid text"):
        split_training_frame(frame, categories)


def test_conflicting_duplicate_groups_are_audited_and_kept_together(
    training_frame: pd.DataFrame, categories: list[str]
) -> None:
    frame = training_frame.copy()
    frame.loc[1, "category"] = categories[1]
    audit = audit_frame(frame, categories)
    assert audit["conflicting_label_groups"] == 1
    assert audit["conflicting_label_details"][0]["records"] == 2
    partitions = split_training_frame(frame, categories)
    matching_partitions = [
        name
        for name, records in partitions.items()
        if records["row_id"].isin(["train:000000", "train:000001"]).any()
    ]
    assert len(matching_partitions) == 1


def test_category_order_is_preserved_and_duplicate_labels_rejected(tmp_path: Path) -> None:
    category_file = tmp_path / "categories.json"
    category_file.write_text('["z_label", "a_label"]', encoding="utf-8")
    assert load_categories(category_file, expected_count=2) == ["z_label", "a_label"]
    category_file.write_text('["z_label", "z_label"]', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate labels"):
        load_categories(category_file, expected_count=2)


def test_preparation_preserves_raw_data_mapping_and_official_test(
    tmp_path: Path, training_frame: pd.DataFrame, categories: list[str]
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    training_frame.to_csv(raw_dir / "train.csv", index=False)
    test_rows = [training_frame.iloc[0].to_dict()]
    test_rows.extend(
        {"text": f"new request for {label}", "category": label} for label in categories
    )
    pd.DataFrame(test_rows).to_csv(raw_dir / "test.csv", index=False)
    (raw_dir / "categories.json").write_text(json.dumps(categories), encoding="utf-8")
    (raw_dir / "LICENSE").write_text("Fixture data only", encoding="utf-8")
    before = {path.name: sha256_file(path) for path in raw_dir.iterdir()}
    output_dir = tmp_path / "processed"
    report_dir = tmp_path / "reports"
    audit = prepare_dataset(raw_dir, output_dir, report_dir, expected_label_count=2)
    assert {path.name: sha256_file(path) for path in raw_dir.iterdir()} == before
    assert audit["official_train"]["records"] == len(training_frame)
    assert audit["official_train_test_hash_overlap"]["unique_groups"] == 1
    assert audit["official_train_test_hash_overlap"]["training_records"] == 2
    assert audit["official_train_test_hash_overlap"]["test_records"] == 1
    mapping = json.loads((output_dir / "label_mapping.json").read_text(encoding="utf-8"))
    assert mapping == [
        {"label_id": index, "category": label} for index, label in enumerate(categories)
    ]
    manifest = pd.read_csv(output_dir / "split_manifest.csv")
    assert len(manifest) == len(training_frame) + len(test_rows)
    assert manifest["row_id"].is_unique
    official_test = pd.read_csv(output_dir / "official_test.csv")
    assert official_test["text"].tolist() == [row["text"] for row in test_rows]
    assert official_test["category"].tolist() == [row["category"] for row in test_rows]
    for name in ("fit", "development", "calibration", "policy"):
        assert (output_dir / f"{name}.csv").is_file()
    checksums = json.loads((output_dir / "checksums.json").read_text(encoding="utf-8"))
    assert checksums["raw_files"] == before
    for filename, expected_hash in checksums["processed_files"].items():
        assert sha256_file(output_dir / filename) == expected_hash
    for filename, expected_hash in checksums["report_files"].items():
        assert sha256_file(report_dir / filename) == expected_hash
    assert (report_dir / "training_class_counts.png").read_bytes().startswith(b"\x89PNG")
    assert (report_dir / "training_text_lengths.png").read_bytes().startswith(b"\x89PNG")
