import csv
import numpy as np
from PIL import Image
import pytest

@pytest.fixture
def cohort(tmp_path):
    """Synthetic images test plumbing only, never disease discrimination."""
    rng = np.random.default_rng(7)
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patient_id", "image_path", "label", "density"])
        for patient in range(60):
            for view in range(2):
                filename = f"{patient}_{view}.png"
                Image.fromarray(rng.integers(0, 256, (80, 80), dtype=np.uint8)).save(tmp_path / filename)
                writer.writerow([f"patient_{patient}", filename, patient % 2, patient % 4 + 1])
    return path


