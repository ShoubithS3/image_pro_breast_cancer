# Two minute speaking notes

Present slides 1 to 4 only. Slides 5 and 6 are backup for questions. Timings: 25 / 30 / 35 / 30 seconds.

## Slide 1

MAMMO-LENS studies benign versus malignant classification of abnormal mammograms. Our question is whether whole-image context and local patches improve reliability across recorded breast-density groups and image degradation. The proposed output is an evidence viewer showing scores, selected patches and uncertainty. The dataset supports retrospective classification, rather than future cancer-risk prediction.

## Slide 2

Our literature credits six primary papers. Lee defines CBIS-DDSM, He supplies residual encoders, Ilse attention pooling and Guo calibration. Li Shen and Yiqiu Shen provide mammography precedents. GMIC already combines global and local features. Our proposed investigation therefore focuses on controlled density and degradation evaluation, and makes no proven architecture-novelty claim.

## Slide 3

We audited 2,955 full mammograms from 1,554 patients. Train, validation and test have zero patient overlap. Normalization produces a whole image and four texture-selected patches. We compare global, patch and combined features. The planned ResNet18 version pools patches with attention and optionally consumes predicted density. We select and calibrate models using validation data.

## Slide 4

Three real handcrafted baselines are complete. Test AUROC is 0.625 for global, 0.591 for patches and 0.620 for combined features. Fusion has not demonstrated improvement. Weak density-four results and noise failures motivate further reliability testing. Deep training remains pending. The next milestone is deep-model comparisons for Review 2, followed by complete evaluation and the final report.
