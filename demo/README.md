# Standalone sample website

Open `index.html` in a modern browser. The file contains all styles and JavaScript; no installation, API keys, network requests, or server are required.

Workflow: choose the illustrative phantom or upload a PNG/JPEG, inspect selected texture patches, adjust blur/noise, then view or download the JSON observations. Uploaded images are processed within the browser.

This sample demonstrates image processing and the proposed interaction design. It does not load the Python model, predict cancer, or reproduce Python percentile normalization. All reported numbers are computed from the displayed pixels. The default phantom is procedural texture, not a patient image or a disease simulation.

Validated in headless Chromium: initialization, patch selection, noise control, reset, report dialog, image upload, phantom reset, and mobile layout without horizontal overflow.
