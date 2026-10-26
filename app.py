"""MAMMO-LENS interactive stress lab. Run: streamlit run app.py."""
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, UnidentifiedImageError
import streamlit as st

from mammo_lens.core import features, prepare_image

st.set_page_config(page_title="MAMMO-LENS · Signal Lab", page_icon="◉", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
.stApp {background:#101918;color:#e6eeeb;font-family:'DM Sans',sans-serif}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.04em}
h1 {font-size:3.7rem!important;font-weight:500!important;line-height:1.05!important;max-width:850px}
[data-testid="stSidebar"] {background:#152320;border-right:1px solid #2b4039}
[data-testid="stHeader"] {background:#101918e8}
.block-container {padding-top:2.4rem;max-width:1500px}
.eyebrow {font-family:monospace;letter-spacing:.19em;color:#a8ddbb;font-size:.72rem;margin-bottom:20px}
.hero-copy {color:#9eafa8;max-width:640px;font-size:1rem;line-height:1.75;margin:12px 0 28px}
.brand {font-family:'Space Grotesk',sans-serif;font-size:1.5rem;letter-spacing:-.06em;font-weight:700}
.brand span {color:#bce8a4}
.rule {border-top:1px solid #31443d;margin:26px 0}
.label {font:11px monospace;letter-spacing:.14em;color:#91a99e;margin-bottom:14px}
.panel {border:1px solid #344a40;padding:22px;border-radius:12px;background:#15231e;min-height:135px}
.panel .number {font-size:1.7rem;color:#d7f6c4;font-family:'Space Grotesk',sans-serif}
.panel p {font-size:.83rem;color:#a8bbb0;line-height:1.5;margin:8px 0 0}
.pill {border:1px solid #49694e;border-radius:30px;padding:7px 12px;font:11px monospace;color:#c0eaa6;display:inline-block}
[data-testid="stMetric"] {background:#182820;border:1px solid #344a40;padding:18px;border-radius:10px}
[data-testid="stFileUploader"] {border:1px dashed #58765b;border-radius:10px}
[data-baseweb="tab-list"] {gap:30px}
[data-baseweb="tab"] {color:#b5c8bc}
.stButton>button {border-radius:30px;background:#c9efa7;color:#12201a;border:0}
a {color:#c9efa7!important}
footer {visibility:hidden}
</style>
""", unsafe_allow_html=True)


def phantom():
    """Illustrative texture phantom, not a mammogram or a simulated disease case."""
    y, x = np.mgrid[-1:1:256j, -1:1:256j]
    mask = ((x+.7)**2 / 1.1**2 + y**2 / .95**2) < 1
    rng = np.random.default_rng(12)
    texture = .3 + .13*np.sin(25*x + 10*y) + .08*np.cos(32*y - 9*x)
    tissue = .25*np.exp(-((x+.25)**2 + y*y)*4)
    pixels = np.clip((texture+tissue+rng.normal(0,.035,x.shape))*mask, 0, 1)
    buffer = BytesIO()
    Image.fromarray((pixels*255).astype(np.uint8)).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


with st.sidebar:
    st.markdown('<div class="brand">MAMMO<span>—</span>LENS</div>', unsafe_allow_html=True)
    st.caption("IMAGE PROCESSING / SIGNAL LAB")
    st.markdown('<div class="rule"></div><div class="label">01 / INPUT SOURCE</div>', unsafe_allow_html=True)
    source = st.radio("Choose input", ["Illustrative phantom", "Upload mammogram"], label_visibility="collapsed")
    uploaded = st.file_uploader("De-identified image export", type=["png","jpg","jpeg","tif","tiff"])
    st.markdown('<div class="rule"></div><div class="label">02 / CONTROLLED DEGRADATION</div>', unsafe_allow_html=True)
    blur = st.slider("Blur radius", 0., 5., 0., .5, help="Applied at 256 × 256 working resolution")
    noise = st.slider("Noise standard deviation", 0., .2, 0., .01)
    st.caption("Adjust one factor at a time to isolate its effect. These are synthetic transformations, not acquisition simulations.")
    with st.expander("Local research model"):
        artifact_path = st.text_input("Model artifact", "artifacts/combined/model.joblib")
        st.caption("Only load artifacts you trained and trust.")
    st.markdown('<div class="rule"></div><span class="pill">RESEARCH PROTOTYPE / v0.1</span>', unsafe_allow_html=True)

st.markdown('<div class="eyebrow">MAMMOGRAPHY × IMAGE PROCESSING × RELIABILITY</div>', unsafe_allow_html=True)
st.title("Look closer.\nQuestion the signal.")
st.markdown('<div class="hero-copy">Explore image texture, inspect local patches, and test what changes when the signal gets harder to read. An interactive research workbench for evidence and model reliability.</div>', unsafe_allow_html=True)
workspace, protocol, experiments = st.tabs(["01  /  Signal workspace", "02  /  Research protocol", "03  /  Experiment ledger"])

with workspace:
    selected = phantom() if source == "Illustrative phantom" else uploaded
    if source == "Illustrative phantom":
        st.info("ILLUSTRATIVE PHANTOM · Procedural texture for exploring the tools. Not a mammogram; disease scoring is disabled.")
    else:
        st.caption("Research-only benign/malignant classification. Future cancer risk and clinical reliability are not established.")
    if selected is None:
        st.markdown('<div class="panel"><div class="number">Your evidence starts here.</div><p>Upload a de-identified PNG, JPEG or TIFF in the sidebar. Keep full images and lesion crops separate in your study.</p></div>', unsafe_allow_html=True)
    else:
        try:
            selected.seek(0)
            clean = prepare_image(selected)
            selected.seek(0)
            pixels = prepare_image(selected, blur=blur, noise=noise)
            _, boxes = features(pixels)
            overlay = Image.fromarray((pixels*255).astype(np.uint8)).convert("RGB")
            draw = ImageDraw.Draw(overlay)
            patch_rows = []
            for index, (x0,y0,x1,y1) in enumerate(boxes, 1):
                draw.rectangle((x0,y0,x1-1,y1-1), outline="#cdf5a6", width=1)
                draw.text((x0+4,y0+3), f"P{index}", fill="#cdf5a6")
                patch_rows.append({"Patch":f"P{index}", "Texture spread":float(pixels[y0:y1,x0:x1].std())})
            a,b,c = st.columns([1,1,.8], gap="large")
            with a:
                st.markdown('<div class="label">A / NORMALIZED REFERENCE</div>', unsafe_allow_html=True)
                st.image(clean, clamp=True, width="stretch")
                st.caption("Percentile normalization · 256 × 256 grayscale")
            with b:
                st.markdown('<div class="label">B / STRESSED + PATCH OVERLAY</div>', unsafe_allow_html=True)
                st.image(overlay, width="stretch")
                st.caption("Four texture-rich patches · not lesion annotations")
            with c:
                st.markdown('<div class="label">C / SIGNAL READOUT</div>', unsafe_allow_html=True)
                st.metric("Mean absolute image change", f"{np.abs(pixels-clean).mean():.4f}")
                st.metric("Working resolution", "256²")
                st.caption("Image-change measurements describe pixels, not malignancy or diagnostic quality.")
            st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
            detail, model_panel = st.columns([1,1], gap="large")
            with detail:
                st.subheader("Local texture profile")
                st.bar_chart(pd.DataFrame(patch_rows).set_index("Patch"), color="#c9efa7", height=210)
                st.caption("Texture spread = standard deviation of normalized patch intensities. Background filtering and grid selection are heuristic.")
            with model_panel:
                st.subheader("Does the score hold up?")
                if source == "Illustrative phantom":
                    st.write("Explore the image tools above. A phantom cannot demonstrate cancer classification.")
                elif not Path(artifact_path).is_file():
                    st.write("Awaiting a trained model. Predictions unlock after a real-data training run; no example scores are substituted.")
                    st.code("python -m mammo_lens.cli data/manifest.csv \\\n  --mode combined --output artifacts/combined", language="bash")
                else:
                    bundle = joblib.load(artifact_path)
                    model, mode = bundle["model"], bundle["mode"]
                    original = float(model.predict_proba([features(clean, mode)[0]])[0,1])
                    changed = float(model.predict_proba([features(pixels, mode)[0]])[0,1])
                    st.metric("Malignant-class model score", f"{changed:.3f}", f"{changed-original:+.3f} vs reference", delta_color="off")
                    st.caption(f"{mode} handcrafted baseline. Uncalibrated research score, not a clinical probability.")
                    st.write("Exploratory low-confidence flag:", "flagged" if max(changed,1-changed)<.7 else "not flagged")
                    st.caption("Threshold 0.70 is unvalidated. No flag does not establish reliability.")
        except (ValueError, OSError, KeyError, UnidentifiedImageError) as exc:
            st.error(f"Unable to process input: {exc}")

with protocol:
    st.subheader("A research question with measurable answers.")
    st.write("Do local image features improve abnormality classification, and how do density and degraded inputs affect model reliability?")
    cols = st.columns(3)
    cards = [("01 / Evidence", "Global + local", "Compare whole-image features, local patches, and their combination on the same patient split."),
             ("02 / Reliability", "Know when to pause", "Measure coverage versus error across confidence thresholds. Fit calibration using validation data in the deep-model phase."),
             ("03 / Robustness", "Stress the signal", "Pair clean and degraded test images, report subgroup counts, and examine density-related performance differences.")]
    for col, (label, title, desc) in zip(cols, cards):
        col.markdown(f'<div class="panel"><div class="label">{label}</div><div class="number">{title}</div><p>{desc}</p></div>', unsafe_allow_html=True)
    st.markdown("### The validity rules")
    st.write("Keep all views and lesions from one patient in one partition. Never choose the split or tune thresholds using test results. Report uncertainty and sample counts alongside metrics.")
    st.warning("CBIS-DDSM is retrospective and enriched for abnormalities. It cannot establish future short-term cancer risk or population screening performance. Density subgroup analysis is not automatic density assessment.")
    st.caption("Current implementation: handcrafted baseline. Planned: deep patch/global encoders, validation-fitted calibration, patient bootstrap intervals, density assessment, and multi-view aggregation. Literature review must establish novelty.")

with experiments:
    st.subheader("Measured results. Traceable runs.")
    st.write("Train each mode on the same manifest and seed. The CLI records model artifacts, image hashes, patient partitions, density summaries, and controlled degradation results.")
    import json
    records = []
    for name in ("global", "patch", "combined"):
        path = Path("artifacts") / name / "metrics.json"
        if path.exists():
            try:
                report = json.loads(path.read_text())
                records.append({"Mode":name, **report["splits"]["test"]["metrics"]})
            except (ValueError, KeyError, TypeError):
                st.error(f"Invalid metrics artifact for {name}")
    if records:
        st.dataframe(pd.DataFrame(records).drop(columns=["confusion_matrix"]), hide_index=True)
        st.caption("Local run outputs; inspect split manifests and dataset provenance before interpreting these values. Synthetic test outputs must not be used as research results.")
    else:
        st.markdown('<div class="panel"><div class="number">No measured runs yet.</div><p>This space fills with real experiment artifacts. Training data, verified labels, and an audited patient split come first.</p></div>', unsafe_allow_html=True)
    st.markdown("### Review milestones")
    st.dataframe(pd.DataFrame([
        {"Review":"01 / Proposal", "Date":"05 Nov 2026", "Evidence":"5+ verified papers · dataset audit · methodology"},
        {"Review":"02 / Progress", "Date":"17 Nov 2026", "Evidence":"Working demo · baseline results · deep-model progress"},
        {"Review":"03 / Final", "Date":"27 Nov 2026", "Evidence":"Trained models · ablations · report · full demo"},
    ]), hide_index=True, width="stretch")

st.markdown('<div class="rule"></div><div class="label">MAMMO-LENS / CAI3400 · RESEARCH PROTOTYPE · NOT FOR CLINICAL USE</div>', unsafe_allow_html=True)
