# MAMMO-LENS Review 1 speaking notes

Target: 10 minutes plus questions. Slide 13 contains references and is for questions.

## Slide 1 MAMMO-LENS

Target 25 seconds. Good morning. MAMMO-LENS is a research workbench for retrospective benign versus malignant mammogram classification. The central question is whether combining whole-image context with local patches remains reliable across breast-density categories and image degradation. Today I will explain the proposal, its literature basis, our dataset safeguards and completed preliminary baselines.

## Slide 2 Problem and scope

Target 45 seconds. The task is classification of existing findings from pathology labels. It does not estimate a person's future cancer risk. Our current 256 by 256 working resolution may discard small calcifications, so this is a measurable design limitation. We want to compare feature approaches under the same patient split, then inspect density subgroups, probability quality and performance when images degrade. The dataset does not represent a general screening population.

## Slide 3 Literature foundation

Target 65 seconds. The dataset paper by Lee defines provenance and annotations. He provides residual encoders. Ilse provides attention pooling for bags of patches, while Guo motivates probability calibration. These are established components. Our implementation combines them for a controlled research experiment. Their original datasets and tasks differ, so none of their reported scores can substitute for our own evaluation. References and links are included at the end and in the proposal.

## Slide 4 Closest mammography work and proposed focus

Target 60 seconds. The closest precedent is GMIC, which already uses global and local information plus gated attention. Therefore I would not claim that combining patches and a whole image is a new architecture. Our project focuses on whether any benefit survives density variation and controlled degradation on a strictly audited cohort. Our texture-selected patches are also much simpler than GMIC's learned saliency and high-resolution patches. A positive or negative result is useful if the comparison is reproducible.

## Slide 5 Audited CBIS-DDSM cohort

Target 55 seconds. We downloaded the Kaggle JPEG export and map case-description series identifiers to JPEGs through dicom_info. The audited cohort contains 2,955 full mammograms from 1,554 patients. Labels are malignant if any described lesion is malignant; both benign categories map to zero. We found no missing mapped JPEGs. We excluded 64 training views from patients appearing in the official test cohort, and all 84 members of 42 duplicate-content groups. Density is known for 2,953 images. These exclusions mean our results use a declared derived cohort rather than an unchanged official benchmark.

## Slide 6 Patient partitions and leakage controls

Target 45 seconds. Training has 1,740 images from 905 patients. Validation has 578 images from 302 patients. The held-out official test subset has 637 images from 347 patients. There is zero patient overlap across partitions. We use seed 42, fit preprocessing statistics for the classifier on training features, fit temperature only on validation scores, and never use test scores for checkpoint selection. Test results already exist for the handcrafted baselines, so future method changes based on these observations must be acknowledged as exploratory and ultimately need a fresh external evaluation.

## Slide 7 Methodology and planned deep model

Target 65 seconds. The completed handcrafted baseline extracts intensity histograms, summary statistics and gradients. Patch features pool four regions selected by intensity variation with background filtering. The implemented deep model resizes the global image to 224 square and the selected patches to 128 square before ResNet18 encoding. Gated attention pools patch vectors. A density head provides auxiliary supervision, and the density-conditioned variant consumes predicted density probabilities at inference. Real deep training remains unfinished. Patch selection after heavy downsampling is a major limitation that the next experiments should address.

## Slide 8 Preliminary real-data baseline results

Target 60 seconds. These are completed real CBIS-DDSM runs, not synthetic smoke scores. Global AUROC is approximately 0.625, patch-only 0.591 and combined 0.620 on the same 637 test images. Global has the lowest Brier score of the three. The intervals overlap, and we have not performed a paired significance test. The correct conclusion is that these preliminary handcrafted features have not demonstrated a fusion improvement. We will keep these baselines as references when evaluating the deep variants.

## Slide 9 Reliability findings motivate the next experiments

Target 50 seconds. The density-4 result is a subgroup observation, not proof of a clinical density effect. It could reflect sample composition, resolution or model weakness. Under deterministic added noise, all images receive a benign thresholded prediction, although ranking AUROC is still above one half. This shows why accuracy or confidence alone is inadequate. We will assess coverage versus error and choose any abstention policy on validation data. The noise and blur tests are controlled perturbations, not validated simulations of scanner acquisition.

## Slide 10 Experiment plan and evaluation

Target 50 seconds. The initial deep configuration uses ImageNet-initialized ResNet18 with a frozen encoder, ten epochs and batch size eight on the local RTX 4050. We will confirm runtime after improving preprocessing. Validation log loss selects the checkpoint. For calibration we will compare raw and calibrated probabilities, and define operational thresholds on validation data. Patient-level maximum scores are a separate exploratory endpoint. Masking is a response test of the whole pipeline because it recomputes selected patches, not a causal explanation.

## Slide 11 Progress and review milestones

Target 45 seconds. The local software foundation and real baselines are complete. The previous session passed fifteen automated tests. A CUDA-capable environment is configured, but the interrupted deep run saved no checkpoint. The immediate engineering task is caching deterministic preprocessing and adding robust checkpoint resume so long runs are practical. The next review is listed for 17 November and the final review for 27 November in the project plan. These are work targets, and faculty or LMS instructions remain the scheduling authority.

## Slide 12 Expected contribution and limitations

Target 35 seconds. The contribution we aim to defend is an auditable experiment that connects image evidence to measured reliability. The current results already show that extra local features can fail to improve held-out performance. The work is retrospective and research only. We need higher-resolution experiments and external validation before making broader claims. I welcome feedback on the experimental scope and the proposed evaluation priorities.

## Slide 13 References

Reference slide for questions. Primary sources were checked on 4 November 2026. The literature summary covers foundational methods and close mammography precedents. It is not a systematic or exhaustive survey.
