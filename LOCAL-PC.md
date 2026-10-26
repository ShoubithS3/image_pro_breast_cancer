# Continue on your local PC

The project is saved in GitHub. This chat's current tools control a cloud machine, not your desktop. Connecting a PC to ChatGPT does not by itself give this chat filesystem or terminal access to that PC.

## Save the project locally

For continued GitHub pushes, install Git and clone from a terminal on your PC:

```bash
cd Desktop
git clone https://github.com/ShoubithS3/image_pro_breast_cancer.git
cd image_pro_breast_cancer
```

If your Desktop is elsewhere, choose that folder in your file explorer and open a terminal there. A Git clone is preferred over the ZIP because it includes Git history and the configured remote. Authenticate Git locally when GitHub requires it; never paste tokens into chat.

Alternatively, download and extract the supplied project ZIP into a folder on your Desktop. The ZIP includes source and documentation, not Git history, a Python environment, a dataset, or trained medical models. Do not merge it over an existing modified project without reviewing differences.

## Run locally

Install Python 3.12 from python.org if needed. Then, from the project folder:

```bash
# Windows:
py -3 run_local.py

# macOS / Linux:
python3 run_local.py
```

Windows users can also double-click `START-WINDOWS.cmd`. The launcher installs locked dependencies into this folder's `.venv`, using a separate `.bootstrap` installer environment if uv is absent. uv downloads Python 3.12 if required. The launcher binds the server to this PC only and opens the application in your browser after its health check succeeds. First setup needs Internet access and may take several minutes. Keep the terminal open while using the app; Ctrl+C stops it.

To install and test without starting the app:

```bash
python run_local.py --setup-only --test
```

Use `py -3` or `python3` instead of `python` when appropriate. If port 8000 is busy, use `--port 8001`. On a machine with an existing uv installation, the ordinary `uv sync --locked --all-extras --python 3.12` workflow from README.md also works.

The portable launcher has been exercised on Linux; Windows/macOS launch behavior must still be checked on those systems. The locked CPU PyTorch wheels must support your OS/architecture; common Windows x64, Linux x64 and Apple Silicon targets are covered by the upstream packages, while older Intel Macs may require a compatible PyTorch configuration.

## Let Codex continue there

Open the **local project folder** in a Codex client or IDE integration that provides local filesystem and terminal access. Select a local session, not this cloud environment. Give that session:

> Continue MAMMO-LENS from this local checkout. Read LOCAL-PC.md, README.md and docs/validation.md. Run the local setup and tests, finish CBIS-DDSM download and real-data training, and commit/push completed changes to the existing GitHub main branch. Keep datasets, patient information, secrets and model binaries out of Git. Do not claim synthetic smoke scores as medical results.

That local session can inspect your PC, resume training, and push through your locally authenticated Git installation. This cloud chat cannot switch itself to that machine or verify files were saved there.

## Finish real-data training locally

Once the app and tests run locally, follow README.md to download CBIS-DDSM, build the audited manifest, and run baseline/deep-model experiments. The cloud proxy's storage-domain restriction may not apply on your PC. No real CBIS-DDSM model or medical accuracy result is included in the handoff.

Use CPU defaults first; use a separately configured compatible CUDA/PyTorch environment if training with your NVIDIA GPU. GPU installation depends on the actual machine and should be checked by the local session.
