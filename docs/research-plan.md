# Research and submission plan

## Proposed contribution

Evaluate global/local mammogram evidence jointly with density subgroup performance and selective prediction under image degradation. Treat the contribution as a hypothesis until literature review and experiments establish what is new. Avoid claiming an original architecture or clinical utility without supporting evidence.

## Review deliverables

Use the numeric dates printed in the assignment; its weekday labels do not match the 2026 calendar. Confirm dates with faculty/LMS.

| Review | Date in assignment | Weight | Deliverables |
| --- | --- | --- | --- |
| Proposal and literature | 5 November 2026 | 20% | Problem statement, at least five verified papers, dataset audit, methodology diagram, experiment plan, timeline; 8–10 minute presentation and report |
| Progress and intermediate results | 17 November 2026 | 30% | Working baseline, patient split audit, initial real-data results, model implementation progress, challenges and revised plan; 10–12 minute presentation and demo |
| Final evaluation | 27 November 2026 | 50% | Trained models, quantitative comparison, ablations, limitations, conclusion, full demo, source code and final report; 12–15 minutes |

The assignment requires a 15–20 page final report, source code and results by 27 November 2026, 11:59 PM (confirm the LMS timezone). All reviews are mandatory. Review 1/2 list a 10%-per-day late penalty; final review permits no late submission. Acknowledge generative-AI assistance as required, and understand the implementation for the viva.

## Work sequence

1. Before Review 1: verify five or more primary papers; record task, dataset, split unit, architecture, metrics, leakage safeguards, limitations, and the exact gap our experiment addresses. Audit Kaggle files and metadata, obtain original dataset references, and inspect polarity, orientation, duplicate images, class balance, and density definitions. Do not invent citations.
2. Before Review 2: run all three handcrafted modes; implement a pretrained patch encoder, a global encoder, and simple feature fusion using training data only. Establish real image preprocessing and patient-safe official partitions before comparing results. Treat high resolution and compute cost as measured design choices.
3. Before final review: freeze the method using validation data, then run held-out evaluation and ablations. Add patient-level bootstrap confidence intervals, validation-fitted probability calibration, multi-view aggregation, and density assessment if feasible and required. Training scripts, checkpoints, manifests, dataset provenance, and measured results must accompany claims.

## Required comparisons

- Global-only, patch-only, and global-plus-patch, holding split and training budget fixed.
- Handcrafted versus deep features, without presenting them as equivalent implementations.
- Density subgroup results; optionally a density-conditioned model if metadata coverage permits. Ground-truth density input must be distinguished from density predicted from images.
- Clean versus predefined degraded inputs, paired per test image.
- Coverage versus error as the model abstains; choose operational thresholds on validation data. Evaluate calibration before interpreting probabilities.

Report image-level and, once aggregation is implemented, patient-level metrics separately. Cluster bootstrap resampling by patient, retaining all views, to avoid falsely narrow intervals. Never select patches using held-out lesion annotations for a claimed full-image screening experiment. CBIS-DDSM's selected abnormal cases do not establish population screening prevalence, positive predictive value, or future-risk performance.

## Final report outline

Abstract and problem; literature review; dataset and ethics/usage; preprocessing; methodology; experimental protocol; measured results and ablations; error analysis and limitations; conclusion and future work; verified references; reproducibility appendix and AI-use acknowledgement.

## Current implementation status

Implemented: full FastAPI/frontend application, authenticated training jobs, UID-based CBIS-DDSM JPEG adapter, handcrafted feature modes, global/patch ResNet18 with gated attention, supervised density head and optional predicted-density fusion, patient-safe official/test or development splitting, duplicate-content checks, validation-only checkpoint selection and temperature fitting, patient-cluster bootstrap intervals, patient aggregation, density subgroup metrics, synthetic blur/noise evaluation, and patch-masking response analysis.

The standalone website and Streamlit viewer remain available as secondary interfaces. CI and Docker deployment definitions are included.

Still outstanding: real dataset download/access, adapter validation on the downloaded export, real-data training and measured results, higher-resolution experiments, recent-paper/full-text literature review, final report/presentations, external validation, and clinical validation. Longitudinal risk prediction cannot be completed with these pathology labels alone. Viewers and synthetic tests must never substitute illustrative numbers for missing results.
