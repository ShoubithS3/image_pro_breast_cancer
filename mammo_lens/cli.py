import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .core import MODES, coverage_curve, features, load_manifest, metrics, patient_split, prepare_image


def train(manifest, output, mode="combined", seed=42):
    output = Path(output)
    if output.exists():
        raise ValueError("Choose a new output directory to preserve previous experiments")
    rows = load_manifest(manifest)
    parts = patient_split(rows, seed)
    x = np.stack([features(prepare_image(r["image_path"]), mode)[0] for r in rows])
    y = np.array([r["label"] for r in rows])
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    model.fit(x[parts["train"]], y[parts["train"]])
    report = dict(mode=mode, seed=seed, model="handcrafted texture + logistic regression",
                  clinical_validation=False, splits={})
    for name, ids in parts.items():
        p = model.predict_proba(x[ids])[:, 1]
        report["splits"][name] = dict(metrics=metrics(y[ids], p),
                                     patient_count=len({rows[i]["patient_id"] for i in ids}),
                                     coverage_curve=coverage_curve(y[ids], p))
    test = parts["test"]
    report["density_subgroups"] = {}
    for density in ("1", "2", "3", "4", ""):
        ids = [i for i in test if rows[i]["density"] == density]
        if ids:
            report["density_subgroups"][density or "unknown"] = metrics(y[ids], model.predict_proba(x[ids])[:, 1])
    report["stress_test"] = {}
    for name, kwargs in {"blur_radius_2": {"blur": 2}, "noise_std_0.05": {"noise": .05}}.items():
        stressed = np.stack([features(prepare_image(rows[i]["image_path"], seed=seed+int(i), **kwargs), mode)[0]
                             for i in test])
        report["stress_test"][name] = metrics(y[test], model.predict_proba(stressed)[:, 1])
    output.mkdir(parents=True)
    joblib.dump(dict(model=model, mode=mode, schema_version=1), output / "model.joblib")
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    assignments = {int(i): name for name, ids in parts.items() for i in ids}
    (output / "split_manifest.json").write_text(json.dumps([
        dict(row, split=assignments[i]) for i, row in enumerate(rows)], indent=2) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


def main():
    parser = argparse.ArgumentParser(description="MAMMO-LENS handcrafted research baseline")
    parser.add_argument("manifest", help="CSV with patient_id,image_path,label and optional density")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=MODES, default="combined")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.manifest, args.output, args.mode, args.seed)


if __name__ == "__main__":
    main()
