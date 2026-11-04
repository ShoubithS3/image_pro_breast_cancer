# Review 1 viva and demonstration

## What is your project actually predicting

Whether an included abnormal mammogram has any described malignant lesion, using retrospective pathology labels. It does not predict future cancer risk because we lack longitudinal follow-up outcomes.

## What is new if global and local fusion already exists

GMIC already combines global and local evidence, so we credit that prior work. Our proposed investigation compares reliability across recorded density groups and controlled image degradation, using one audited cohort, fixed patient partitions, calibration and coverage/error analysis. This experimental focus is a hypothesis to investigate, not a proven novelty claim.

## Why CBIS-DDSM

It has public mammography images with pathology and density metadata, which supports a reproducible retrospective study. Our selected abnormal-case JPEG cohort limits screening and generalization claims. Cite the original TCIA collection and Lee et al. paper as well as the Kaggle export.

## How did you avoid leakage

We keep every patient in one partition, exclude 64 training views whose patients occur in the official test cohort and exclude all 84 images in 42 exact-duplicate groups. Final train/validation/test patient overlap is zero. Exact byte hashing does not detect all near duplicates. The held-out set has already been examined for baselines, so subsequent changes based on it are exploratory and need independent validation.

## Why are the results modest

The completed models are simple handcrafted texture baselines, and the current pipeline reduces mammograms to 256 by 256. This may discard small findings. Texture selection can miss the lesion. These are measured limitations, not reasons to invent better scores.

## Did combining patches improve results

No improvement has been demonstrated. Test AUROC is global 0.6249, patch 0.5907 and combined 0.6198. The confidence intervals overlap. We have not performed a paired test establishing a significant difference. Deep models may or may not improve on these results.

## What does AUROC mean

It measures ranking across classification thresholds. Roughly, AUROC is the probability that a randomly chosen positive example receives a higher score than a randomly chosen negative example. It is different from accuracy at a chosen threshold. Sensitivity is the fraction of malignant images detected, and specificity is the fraction of benign images correctly classified.

## What is calibration and Brier score

Calibration asks whether predicted probabilities agree with observed outcome frequencies. We fit a scalar temperature using validation predictions. Brier score is mean squared probability error and lower is better. Brier reflects more than calibration alone, so it does not replace a calibration curve. Calibration can fail after image or population shift.

## What did the density and noise results show

Global density category 4 had AUROC 0.4895 on 113 test images. Under noise standard deviation 0.05 at working resolution, sensitivity at threshold 0.5 fell to zero. These findings motivate subgroup and stress evaluation. They do not establish a causal density effect or a validated scanner-noise model.

## Are highlighted patches explanations

They show texture-selected regions. Attention can show model weighting, and masking shows the pipeline's score response. Masking recomputes patch selection, so it does not isolate fixed-region causal importance. We do not claim that a highlighted patch is a confirmed lesion.

## What is the deep model status

ResNet18 with gated attention, a density head and optional predicted-density fusion is implemented. CUDA works on the RTX 4050. The real run was stopped before it saved a checkpoint. We still need preprocessing optimization, training, ablations and held-out deep evaluation.

## Why do six papers have different datasets

The literature includes methodological foundations and close mammography precedents. He supplies the residual encoder, Ilse attention pooling and Guo calibration. Lee defines CBIS-DDSM, while Li Shen and Yiqiu Shen provide mammography precedents. Their tasks and protocols differ, so we compare ideas rather than rank their headline scores against ours.

## Optional two-minute demonstration

1. Open `http://127.0.0.1:8000` after launching `START-WINDOWS.cmd`. Start the app before the review so dependency setup does not interrupt you.
2. Briefly show the experiment ledger with the three real baseline runs.
3. Use a permitted, de-identified local CBIS-DDSM JPEG. Select `real-baseline-global` and run server analysis.
4. Show the four selected patches, research score and masking responses. Explain the heuristic selection.
5. Change only noise to 0.05 and rerun. Explain score shift and the aggregate robustness finding.
6. Reset the controls. Explain that deep models remain pending.

If the live demo fails, return to the measured-results slide and describe the workflow. The dataset and model artifacts are local and excluded from Git. Keep a permitted demo image and the deck available offline. The procedural phantom demonstrates image processing only and cannot demonstrate a medical classifier.

## One-minute opening if time is cut

MAMMO-LENS studies benign versus malignant mammogram classification with global and local image evidence. We focus on whether model reliability changes across density categories and image degradation. The audited cohort has 2,955 images from 1,554 patients with zero patient overlap across train, validation and test. Three completed real handcrafted baselines give AUROC 0.6249, 0.5907 and 0.6198. Local fusion has not yet shown an improvement. We will next evaluate ResNet18 variants, calibration and selective prediction on the same cohort, with higher-resolution experiments as an explicit limitation and next step.
