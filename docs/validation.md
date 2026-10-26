# Validation record — 26 October 2026

## Verified in the current cloud machine

- Locked installation with Python 3.12, uv 0.12.23, CPU PyTorch 2.6.0 and torchvision 0.21.0.
- Full Python suite: **15 passed**. Tests cover patient grouping, exact-image duplicate checks, 16-bit preprocessing, CBIS UID mapping and patient exclusions, training, gradients, checkpoints/reload, validation calibration, clustered intervals, upload analysis, patch masking, operator authentication, persistent training jobs, and the secondary Streamlit viewer.
- After the last API/job/doc adjustments, the five API/data tests passed again; JavaScript syntax and Git whitespace checks passed.
- Headless Chromium: live frontend initialization, patch selection, noise/reset controls, report dialog, file upload, server-side image analysis, stale-result clearing, operator authentication, and mobile layout without horizontal overflow. Reproduce with `python scripts/browser_smoke.py` against a running local server.
- One-epoch CPU deep training smoke command completed, saved a checkpoint and evaluation artifacts, and was recorded as `synthetic-smoke`. The registry blocks it from disease scoring. These random-texture results have no medical meaning.
- Official ImageNet ResNet18 encoder weights downloaded through torchvision with its hash verification enabled.
- FastAPI root and health responses succeeded. No real-data model is currently available.
- Docker Compose configuration parsed successfully.

## Blocked or unverified

- Kaggle's dataset-list endpoint succeeds, but the actual dataset download redirects to `storage.googleapis.com`, where the cloud proxy returns 403. That host has been added to the environment configuration draft along with `www.kaggle.com`; runtime access still needs to be enabled and rechecked.
- Real CBIS-DDSM import/training, benchmark metrics, and full clinical/longitudinal validation are **not** complete. The adapter is tested on representative metadata fixtures, not the actual downloaded export.
- The Docker build fetched the Python base image, then failed because the build container could not resolve the package host / configured egress proxy. Passing existing proxy build arguments did not restore DNS. The native Python workflow works; the container image is not validated.
- GitHub Actions is configured and source is pushed, but its hosted run result has not been independently inspected here.
- Publisher/arXiv full-text pages are blocked; the literature document is a starting bibliography, not a completed five-paper review.

## Remaining work for the submission

Enable dataset download, validate the actual export and exclusions, run the predefined baselines and deep-model ablations, inspect failures, complete literature reading, and write the report/presentations using measured results. The software stack is implemented; those research deliverables cannot be replaced by smoke tests or a polished interface.

## Local-PC handoff launcher

`run_local.py --setup-only --test` created a fresh checkout-local `.venv` and passed all 15 tests on Linux. The launcher started the primary application on a separate test port; root and API health returned HTTP 200. Windows CMD and macOS execution remain untested until run on those systems.
