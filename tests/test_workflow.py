import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from mammo_lens.cli import train
from mammo_lens.core import coverage_curve, features, load_manifest, patient_split, prepare_image



def test_split_keeps_all_views_of_patient_together(cohort):
    rows = load_manifest(cohort)
    parts = patient_split(rows)
    groups = [{rows[i]["patient_id"] for i in ids} for ids in parts.values()]
    assert all(not (groups[a] & groups[b]) for a, b in [(0, 1), (0, 2), (1, 2)])
    assert sorted(i for ids in parts.values() for i in ids) == list(range(len(rows)))


def test_duplicate_content_rejected(cohort):
    with cohort.open("a") as handle:
        handle.write("other_patient,0_0.png,1,1\n")
    with pytest.raises(ValueError, match="Duplicate image"):
        load_manifest(cohort)


def test_sixteen_bit_input_and_stress(tmp_path):
    path = tmp_path / "16bit.tiff"
    image = np.tile(np.linspace(0, 65535, 300, dtype=np.uint16), (300, 1))
    Image.fromarray(image).save(path)
    clean = prepare_image(path)
    degraded = prepare_image(path, blur=2, noise=.05)
    assert clean.shape == (256, 256)
    assert clean.std() > .2
    assert not np.allclose(clean, degraded)
    for mode, size in [("global", 23), ("patch", 46), ("combined", 69)]:
        vector, boxes = features(clean, mode)
        assert len(vector) == size
        assert np.isfinite(vector).all()
        assert len(boxes) == 4


def test_abstention_reports_empty_coverage():
    curve = coverage_curve([0, 1], [.5, .5])
    assert curve[0]["coverage"] == 1
    assert curve[-1]["coverage"] == 0
    assert curve[-1]["error_rate"] is None


def test_training_produces_evaluation_and_preserves_previous_run(cohort, tmp_path):
    output = tmp_path / "run"
    report = train(cohort, output)
    assert report["splits"]["test"]["metrics"]["n"] > 0
    assert set(report["stress_test"]) == {"blur_radius_2", "noise_std_0.05"}
    assert (output / "model.joblib").is_file()
    assignments = json.loads((output / "split_manifest.json").read_text())
    assert len(assignments) == 120
    with pytest.raises(ValueError, match="preserve"):
        train(cohort, output)


def test_app_starts_without_model():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=20)
    assert not app.exception
    assert "Question the signal" in app.title[0].value
    assert len(app.slider) >= 2
    assert float(app.metric[0].value) == 0
    app.slider[1].set_value(.1).run()
    assert not app.exception
    assert float(app.metric[0].value) > 0
    app.radio[0].set_value("Upload mammogram").run()
    assert not app.exception
    assert len(app.metric) == 0
