# Final report and review package

Target the assignment's 15–20 page final report. Write results only after the corresponding real-data command has completed. Never copy synthetic smoke metrics into research tables.

| Section | Suggested pages | Required evidence |
| --- | --- | --- |
| Abstract and problem | 1–2 | Assigned topic, retrospective endpoint, risk-label limitation, precise hypothesis |
| Literature | 3–4 | At least five read primary papers; task/split/metric comparison and specific gap |
| Dataset and preprocessing | 2–3 | Source/usage terms, counts, label mapping, missing/duplicate audit, patient overlap exclusions, density availability, image examples |
| Method | 3–4 | Global/local architecture, gated attention, density auxiliary task, loss, freeze/fine-tune choices, preprocessing and patch selection |
| Experimental protocol | 1–2 | Fixed patient split, validation-only selection/calibration, same cohort for ablations, compute budget, seeds |
| Results and discussion | 3–4 | Actual held-out metrics, patient-cluster intervals, density subgroup sample sizes, clean/degraded comparison, coverage/error curves, failures |
| Conclusion, limitations, references | 2–3 | Supported findings, low-resolution/selected-cohort limitations, verified citations, reproducibility and AI-use acknowledgement |

## Required results tables

1. Handcrafted global / patch / combined, then deep global / patch / combined using the same audited cohort and seed.
2. Deep combined with density conditioning off/on; report density head accuracy separately from classification.
3. Uncalibrated versus validation-temperature-scaled Brier score and AUROC. Temperature scaling is monotonic, so AUROC should generally remain unchanged; investigate any clipping/tie differences.
4. Density groups 1–4 and unknown: counts, class counts, AUROC where defined, sensitivity/specificity, and limitations for small groups.
5. Clean / blur / noise comparisons at declared severity; show paired error examples.
6. Coverage/error at declared confidence thresholds. Do not select an operational threshold on the test set.

## Reproducibility record

Retain Git commit, `uv.lock`, dataset receipt, manifest audit, all exclusions, seed, exact CLI commands, hardware, epoch histories, model configuration, held-out JSON, and private split manifests. Share patient data and images only under source terms. Checkpoints are ignored in Git; provide an authorized artifact download or documented training reproduction.

## Demo script

Open the real application, explain the illustrative phantom, upload a de-identified real image, inspect patches, select an actual trained run, apply one degradation, and compare score response. Demonstrate the low-confidence flag and masking caveats. Open the experiment ledger and show corresponding held-out results. Show a known failure rather than hiding it.

## Oral defence checklist

Explain why future risk needs longitudinal outcomes; how patient leakage is prevented; why exact-image hashes are insufficient for all duplicates; what gated attention computes; how density supervision differs from density conditioning; where calibration is fitted; what a patient-cluster bootstrap resamples; and why neither attention nor perturbation is a clinical explanation. Acknowledge AI assistance according to the assignment.
