# MAMMO-LENS

**An interactive mammography research workbench: global/local evidence, density assessment, and reliability under image degradation.**

Built for Image Processing and Analysis (CAI3400), Presidency University, Bengaluru. Full source includes a web frontend, FastAPI backend, CBIS-DDSM adapter, handcrafted baselines, a PyTorch global/patch model, training jobs, calibration, evaluation, and tests.

> This is a research project, not a clinical tool. CBIS-DDSM supports retrospective abnormality classification; it does **not** establish future short-term cancer risk. No real-data performance is claimed until dataset preparation and training have completed.

## Move to your local desktop

See [LOCAL-PC.md](LOCAL-PC.md). Clone this repository on your PC, then run `python run_local.py` (Windows: `py -3 run_local.py`; macOS/Linux: `python3 run_local.py`). The launcher prepares a local environment and opens the app in your browser. Continue development through a Codex session attached to that local folder.

## Start the full application

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync --locked --all-extras --python 3.12
uv run --all-extras python -m uvicorn mammo_lens.api:app --host 127.0.0.1 --port 8000
```

Open the server's root page in your local browser. The server serves both the frontend and API; no Node build is needed. Interactive API documentation is at `/docs`. For remote deployment, use an authenticated HTTPS reverse proxy and set appropriate upload/concurrency limits; operator training is disabled by default.

For this cloud instance, the prepared virtual environment is `/workspace/mammo-lens-venv`; invoke its Python directly or use `UV_PROJECT_ENVIRONMENT=/workspace/mammo-lens-venv uv run --all-extras ...`. Set `UV_CACHE_DIR=/workspace/.uv-cache` when the home cache directory is read-only.

### Frontend workflow

1. Explore the procedural phantom, or upload a de-identified PNG/JPEG. The phantom is texture only and never receives a disease score.
2. Inspect four texture-selected patches and controlled blur/noise changes in the browser.
3. Select a completed model, then click **Run server analysis**. The API normalizes the original upload consistently with training and returns model scores, score changes, optional density probabilities, and patch-masking responses. No model means image processing only.
4. Review held-out results in the experiment ledger, inspect coverage/error curves and patient confidence intervals, or export your observation report.
5. Use the authenticated operator console to audit the dataset and launch/monitor training jobs, or train from the CLI.

Uploads are not retained after the request. The multipart parser may use temporary files for larger uploads; request files are closed after reading. API analysis additionally supports TIFF exports. Scores and images are returned to the requesting browser; no external prediction service is used. The UI clearly labels unvalidated confidence flags and synthetic experiments.

A separate [standalone sample](demo/index.html) opens without Python. The earlier Streamlit viewer is retained in `app.py` as an optional exploratory interface; the FastAPI frontend is the primary application.

## Get and prepare CBIS-DDSM

Assignment dataset: [awsaf49/cbis-ddsm-breast-cancer-image-dataset](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset). Respect its source usage terms. Download needs `www.kaggle.com` and the redirect host `storage.googleapis.com`; the public download does not normally require an API key. Keep at least 15 GiB free. The archive is approximately 5.3 GB compressed.

```bash
uv run python scripts/download_dataset.py --destination data/cbis-ddsm
uv run python -m mammo_lens.data data/cbis-ddsm --output data/manifest.csv
```

The downloader preserves TLS verification, bounds archive size, rejects unsafe extraction paths/symlinks, and lets ZIP extraction check CRC. Its SHA-256 receipt is a local provenance record, not a publisher-authenticated checksum. Existing destinations/archives are preserved rather than overwritten; interrupted `.part` files are not automatically resumed.

The adapter maps case-description DICOM series identifiers to full-image JPEGs via `dicom_info.csv`. It aggregates repeated lesion descriptions into one full-image label: any malignant lesion → 1; `BENIGN` and `BENIGN_WITHOUT_CALLBACK` → 0. It checks conflicting metadata and duplicate image bytes. Density 0 is treated as unknown; 1–4 are auxiliary targets, not clinical categories inferred from image texture.

Original test images are held out. If a test patient also has training images, their **training views are excluded** and counted in the audit; they are never leaked into training or added to the official test set. Validation is a fixed patient-group split of the retained training patients. This protocol differs from studies that retain patient overlap; report the exact audit counts. Missing JPEGs are counted as exclusions. For official benchmark claims, require zero missing files and report all remaining exclusions.

The adapter has fixture-based integration tests but still needs validation against the actual downloaded export. It refuses ambiguous UID mappings instead of guessing. DICOM decoding is outside this JPEG-export workflow; independently verify polarity/orientation if exporting DICOM yourself.

### Bring your own manifest

CSV fields: `patient_id,image_path,label,density,source_split`. Paths resolve relative to the CSV; density may be blank. Labels: benign=0, malignant=1. `source_split` must be `train`/`test` for every row, or blank for every row. Without official partitions, the development split is approximately 60/20/20 by patient. Each partition must contain both classes; at least ten patients are required, although useful research needs many more.

Keep all views, sides, and lesions belonging to one patient under the same ID. Duplicate hashes catch exact copies, not recompressed duplicates. Audit export provenance, target definitions, image types, and source identifiers before training. Never mix lesion crops with full mammograms without a declared experiment.

## Train the baselines

```bash
uv run python -m mammo_lens.cli data/manifest.csv --mode global --output artifacts/global
uv run python -m mammo_lens.cli data/manifest.csv --mode patch --output artifacts/patch
uv run python -m mammo_lens.cli data/manifest.csv --mode combined --output artifacts/combined
```

These use intensity histograms, statistics, and gradients with standardized logistic regression. Patch mode pools four texture-rich 64×64 grid patches. Selection is heuristic, not lesion localization. Every run fits a temperature on validation scores only, then evaluates the held-out partition. New output directories are required to preserve prior runs.

## Train the deep model

```bash
uv run --all-extras python -m mammo_lens.deep data/manifest.csv \
  --output artifacts/deep-combined --mode combined --epochs 10 --batch-size 8

# Density-conditioned fusion ablation; needs training density labels:
uv run --all-extras python -m mammo_lens.deep data/manifest.csv \
  --output artifacts/deep-density --mode combined --density-aware --epochs 10
```

Architecture: an ImageNet-initialized shared ResNet18 encodes a normalized global image and four selected patches. Gated attention pools local vectors; the classifier uses global, patch-only, or combined features. A supervised four-class density head is trained where density labels exist. Optional fusion uses **predicted** density probabilities, never held-out ground-truth density at inference. Compare density conditioning on/off as an ablation.

Pretrained weights are cached under ignored `state/torch/` by default (`TORCH_HOME` overrides this), preserving checksum verification.

The encoder, including its BatchNorm statistics, is frozen by default for modest CPU compute. `--fine-tune` trains it with a lower learning rate; `--device cuda` requires a separately installed compatible GPU build of PyTorch. The locked environment uses CPU wheels. `--no-pretrained` disables ImageNet initialization; appropriate for software smoke tests, not an equivalent research baseline.

Validation log loss selects the checkpoint; validation scores fit scalar temperature calibration. The test set is not used for selection or calibration. Outputs include safe-loadable tensor weights, model configuration, epoch history, image hashes, patient split assignments, held-out metrics, paired blur/noise tests, and density summaries. Whole images are reduced to 256² before 224² global / 128² patch encoding: this is a resource-conscious starting point that can discard small calcifications. Measure higher-resolution alternatives before claiming clinical relevance.

The architecture composes established techniques; originality must come from a literature-supported research gap and measured experiments, not a renamed network.

## Metrics and evidence

- Image-level AUROC, sensitivity/specificity at threshold 0.5, Brier score, and coverage versus error.
- Patient-level exploratory endpoint: maximum image score, with a positive label if any included image is malignant.
- Image-level 95% intervals from **200 patient-cluster bootstrap resamples**; valid resample counts are included. Small studies need careful interpretation and potentially more resamples.
- Density subgroup counts/metrics and deep density-head accuracy when labels exist.
- Blur/noise at the working resolution, paired with clean test images. These are controlled perturbations, not validated acquisition simulations.
- API patch masking: replace a selected region with median image intensity and measure the score delta. Patch selection is recomputed after masking; this measures the full pipeline response, not isolated fixed-patch importance. Attention/masking is not causal or clinical explanation.

The frontend's low-confidence flag uses 0.70, an explicitly unvalidated exploratory threshold. Temperature fitting is implemented; it does not guarantee calibration on a shifted population. Single-class subgroups have undefined AUROC, and absent classes yield undefined sensitivity/specificity.

## Operator training console

CLI training needs no token. Web training requires explicit opt-in:

```bash
export MAMMO_ENABLE_TRAINING=1
# Generate a token locally without printing or committing it:
export MAMMO_ADMIN_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
uv run --all-extras python -m uvicorn mammo_lens.api:app --host 127.0.0.1 --port 8000
```

Use the securely obtained token in the console; never put it in Git or share it in chat. The browser holds it in memory only. Do not enable operator mode on an unauthenticated public deployment. One server worker supervises one training subprocess at a time. Jobs persist under ignored `state/`; logs are operator-only. Interrupted records are marked after restart; do not restart the server while a training run is active. Dataset/artifact paths are operator-configured, never accepted from browser requests.

Optional environment variables are documented in `.env.example`. They are read from the process environment; `.env` is not automatically loaded. Artifacts are trusted local outputs; uploading models or arbitrary pickle files is not supported.

## Verify the entire pipeline

```bash
uv run --all-extras python -m pytest -q
node --check frontend/app.js
# Optional, with Chromium installed and the API running:
uv run python scripts/browser_smoke.py
uv run python scripts/smoke_train.py
uv run --all-extras python scripts/smoke_train.py --deep \
  --root data/synthetic-deep --output artifacts/synthetic-deep
```

Smoke commands generate random textures with arbitrary labels. They validate software plumbing only, are marked `synthetic-smoke`, and are blocked from disease scoring by the model registry. Their metrics must never be used as mammogram results. Real trained artifacts are intentionally not committed.

CI runs the Python suite and JavaScript syntax check on pushes and pull requests. Docker files are supplied as a deployment option:

```bash
docker compose up --build
```

The container serves the app on port 8000, runs as a non-root user, and defaults to training off. Docker Compose configuration validates, but the cloud Docker build could not resolve the package host / egress proxy from its build container. The container image remains unverified; Python/API execution is the tested primary workflow here. See [validation record](docs/validation.md).

## Project layout

```text
frontend/                  Responsive same-origin research UI
mammo_lens/api.py          FastAPI uploads, analysis, experiments, operator jobs
mammo_lens/deep.py         ResNet18, gated attention, density head, training CLI
mammo_lens/cli.py          Handcrafted baseline training CLI
mammo_lens/data.py         CBIS-DDSM adapter and audit
mammo_lens/evaluation.py   Calibration, patient evaluation, clustered intervals
mammo_lens/inference.py    Completed-model registry and shared inference
mammo_lens/jobs.py         Persistent operator training supervisor
scripts/                   Dataset download and synthetic smoke runners
tests/                     Data, training, checkpoint, API and interaction checks
docs/                      Research plan, literature starting points, report guide
demo/                      Self-contained concept website
```

See [research plan](docs/research-plan.md), [literature notes](docs/literature.md), and [report guide](docs/report-guide.md). Keep data, patient identifiers, model binaries, local secrets, and generated results out of Git. Final deliverables must include measured real-data results and model availability through an appropriate, permitted artifact channel.
