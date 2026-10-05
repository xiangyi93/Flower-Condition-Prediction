from pathlib import Path

import pytest

from pipeline_evaluate import (
    binary_counts,
    classification_breakdown,
    load_stage_truth,
    parse_sample_number,
    stage_metrics,
    threshold_rows,
)


def test_parse_sample_number_supports_all_current_filename_styles() -> None:
    assert parse_sample_number("000012.png") == 12
    assert parse_sample_number("daylily__000003__abc.png") == 3
    with pytest.raises(ValueError, match="Cannot recover"):
        parse_sample_number("no-source-number.png")


def test_load_stage_truth_rejects_duplicate_source_species(tmp_path: Path) -> None:
    directory = tmp_path / "cherry_green"
    directory.mkdir()
    (directory / "000001.png").touch()
    (directory / "cherry__000001__copy.png").touch()
    with pytest.raises(ValueError, match="Duplicate stage truth"):
        load_stage_truth(tmp_path)


def test_pipeline_metrics_separate_coverage_from_classifier_accuracy() -> None:
    rows = [
        {"gt_present": True, "true_stage": "green", "predicted_stage": "green", "foreground_pixels": 10, "foreground_ratio": .02},
        {"gt_present": True, "true_stage": "half", "predicted_stage": "full", "foreground_pixels": 10, "foreground_ratio": .02},
        {"gt_present": True, "true_stage": "full", "predicted_stage": None, "foreground_pixels": 0, "foreground_ratio": 0.0},
        {"gt_present": False, "true_stage": None, "predicted_stage": "green", "foreground_pixels": 2, "foreground_ratio": .0005},
    ]
    presence = binary_counts(rows, .001)
    stages = stage_metrics(rows, .001)
    assert presence == {"tp": 2, "fp": 0, "fn": 1, "tn": 1, "precision": 1.0, "recall": pytest.approx(2 / 3)}
    assert stages["coverage"] == pytest.approx(2 / 3)
    assert stages["conditional_accuracy"] == .5
    assert stages["system_accuracy"] == pytest.approx(1 / 3)
    assert threshold_rows(rows, [0.0, .001])[0]["fp"] == 1


def test_unlabelled_present_region_affects_presence_but_not_stage_denominator() -> None:
    rows = [
        {"gt_present": True, "true_stage": None, "predicted_stage": "green", "foreground_pixels": 5, "foreground_ratio": .01},
        {"gt_present": True, "true_stage": "half", "predicted_stage": "half", "foreground_pixels": 5, "foreground_ratio": .01},
    ]
    assert binary_counts(rows, .001)["tp"] == 2
    stages = stage_metrics(rows, .001)
    assert stages["labelled"] == 1
    assert stages["system_accuracy"] == 1.0


def test_classification_breakdown_preserves_requested_orders_and_confidence_errors() -> None:
    rows = [
        {"species": "cherry", "true_stage": "green", "predicted_stage": "green", "status": "ok", "stage_confidence": .8},
        {"species": "cherry", "true_stage": "full", "predicted_stage": "half", "status": "ok", "stage_confidence": .95},
        {"species": "hydrangeas", "true_stage": "half", "predicted_stage": "half", "status": "ok", "stage_confidence": .7},
        {"species": "daylily", "true_stage": None, "predicted_stage": "green", "status": "ok", "stage_confidence": .9},
    ]
    species, stages, confidence = classification_breakdown(rows)
    assert list(species) == ["cherry", "hydrangeas", "daylily"]
    assert list(stages) == ["green", "half", "full"]
    assert species["cherry"]["system_accuracy"] == .5
    assert confidence["wrong_at_least_0_90"] == 1
