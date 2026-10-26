# Literature starting points and contribution boundaries

This is a reading list for the proposal, not a completed systematic review. Read the original papers and verify the exact experiments before presenting comparisons or quoting numerical results. The cloud network currently blocks the publisher/arXiv pages; full-text review has not been completed here.

| Primary work | Why it matters | What it does not establish for this project |
| --- | --- | --- |
| Lee et al. (2017), **A curated mammography data set for use in computer-aided detection and diagnosis research**, Scientific Data. [DOI: 10.1038/sdata.2017.177](https://doi.org/10.1038/sdata.2017.177) | CBIS-DDSM source, annotations and intended use; read before defining cohorts. | Longitudinal future-risk labels, population screening prevalence, or performance of our model. |
| He et al. (2016), **Deep Residual Learning for Image Recognition**, CVPR. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) | Residual image encoder used as the backbone. | Mammogram-specific suitability or diagnostic validation of ImageNet features. |
| Ilse, Tomczak and Welling (2018), **Attention-based Deep Multiple Instance Learning**, ICML. [PMLR 80](https://proceedings.mlr.press/v80/ilse18a.html) | Attention pooling and the gated-attention form used to combine patch representations. | Attention weights as causal explanations or lesion annotations. |
| Guo et al. (2017), **On Calibration of Modern Neural Networks**, ICML. [PMLR 70](https://proceedings.mlr.press/v70/guo17a.html) | Temperature scaling and the need to assess confidence separately from accuracy. | Guaranteed calibration on an external mammography population. |
| Yala et al. (2021), **Toward robust mammography-based models for breast cancer risk**, Science Translational Medicine. [DOI: 10.1126/scitranslmed.aba4373](https://doi.org/10.1126/scitranslmed.aba4373) | A relevant example of longitudinal mammography risk modeling; inspect outcomes and study design to distinguish risk from retrospective classification. | The ability to train future-risk prediction with CBIS-DDSM pathology labels alone. |

## Literature matrix to complete before Review 1

For each paper, record the task, cohort, sample counts, outcome horizon, unit of splitting, views, preprocessing, model, metrics, external validation, and limitations. Add at least two recent mammography-specific global/local or density-aware papers. Do not compare headline AUROCs across different cohorts as if they were controlled experiments.

## Proposed project contribution

Study **whether the benefit of global/local fusion remains reliable across breast-density groups and controlled image degradations**, with validation-fitted calibration and explicit coverage/error analysis. Use patch masking to inspect score response, while keeping interpretation claims limited.

This combination is a proposed experimental focus, not an established novelty claim. Literature review may reveal close prior work; revise the hypothesis accordingly. The ResNet backbone, gated attention, multitask density head, and temperature scaling are existing techniques and must be credited.
