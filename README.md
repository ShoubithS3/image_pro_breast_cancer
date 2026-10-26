# MAMMO-LENS

**Density-aware evaluation, patch evidence, and reliability experiments for mammogram classification.**

Image Processing and Analysis (CAI3400), Presidency University, Bengaluru. This repository starts with a runnable **handcrafted baseline**. The proposed deep model is future work; no real-data training results or novelty claims are supplied.

## Research question

Do local image features improve benign/malignant classification over global features, and how do breast density and image degradation affect performance and selective prediction?

The assigned topic is “Breast Cancer Risk Prediction from Mammograms,” including density assessment and patch-based deep features. CBIS-DDSM supports retrospective abnormality classification; future short-term cancer risk requires longitudinal outcomes and a defined prediction horizon. Discuss that scope distinction with the coordinator. Density subgroup evaluation is not automated density assessment. A deep patch encoder and density assessment remain required extensions if the assigned scope is retained.

## Run

Python 3.11+; the tested cloud runtime uses Python 3.12. From this checkout:

```bash
UV_CACHE_DIR=/workspace/.uv-cache uv venv /workspace/mammo-lens-venv
UV_CACHE_DIR=/workspace/.uv-cache uv pip sync --python /workspace/mammo-lens-venv/bin/python requirements.lock
source /workspace/mammo-lens-venv/bin/activate
python -m pytest -q
streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
```

No API key is needed. The viewer can normalize images and display patches without a model; predictions appear only when a locally trained artifact exists. Never load untrusted joblib/pickle artifacts.

A self-contained browser sample is available at [demo/index.html](demo/index.html). Download and open it directly to explore the workflow without installing Python. It demonstrates browser image processing only; it does not perform trained inference.

## Dataset preparation

Obtain CBIS-DDSM from the assignment's Kaggle dataset (`awsaf49/cbis-ddsm-breast-cancer-image-dataset`), respecting its usage conditions. Keep downloaded data under ignored `data/`. No dataset is bundled or downloaded automatically.

Prepare a CSV with the following schema; paths resolve relative to the CSV:

```csv
patient_id,image_path,label,density
P_00001,images/example.png,0,2
P_00002,images/another.png,1,3
```

`0` = benign, `1` = malignant. Optional density: categories 1–4; blank means unknown. Verify actual source metadata and label semantics before mapping. Preserve original patient IDs across all lesions, sides, and views. Use full mammogram exports consistently; mixing lesion crops and full images can confound the study. Map CBIS-DDSM metadata explicitly, including the treatment of `BENIGN_WITHOUT_CALLBACK`, and document exclusions. Do not map labels using filenames. DICOM needs a separately validated export step with polarity/orientation checks; this baseline accepts PNG/JPEG/TIFF, not DICOM.

The loader rejects duplicate image bytes and missing files. This does not detect recompressed duplicates; manually audit source patient identifiers and export provenance too. Train-only augmentation must be added after splitting, never as new patients.

## Train and compare

```bash
python -m mammo_lens.cli data/manifest.csv --mode global --output artifacts/global
python -m mammo_lens.cli data/manifest.csv --mode patch --output artifacts/patch
python -m mammo_lens.cli data/manifest.csv --mode combined --output artifacts/combined
```

These commands use the same fixed patient-group split (approximately 60/20/20 by patient, seed 42). Each partition must contain both classes. At least ten patients are required; a scientifically useful cohort needs substantially more. Preserve the official held-out partition for the final study once its patient overlap has been audited; the current random grouped split is a development baseline, not official benchmark reproduction. Do not tune the seed to improve test performance.

Each run writes a local model, metrics, and split manifest. Outputs contain patient identifiers and local paths; keep them private. Existing output directories are never overwritten. Metrics include AUROC, sensitivity/specificity at 0.5, Brier score, density subgroup counts, confidence-based coverage/error curves, and blur/noise stress tests. Single-class subgroups have undefined AUROC; absent classes have undefined sensitivity/specificity. Small subgroups are exploratory.

The baseline uses normalized 256×256 grayscale histograms, intensity statistics, and gradient features. Patch mode pools four high-texture 64×64 grid patches. These patches are not lesion detections or explanations of model causality. Noise/blur are controlled synthetic degradations at the resized resolution, not validated simulations of acquisition conditions. Scores and the demo's 0.70 confidence flag are uncalibrated research outputs.

## Submission plan

See [the research plan](docs/research-plan.md) for review deliverables, the deep-model extension, evaluation safeguards, and the evidence still needed. Tests use synthetic images solely to verify software behavior; their scores must never enter the results section.
