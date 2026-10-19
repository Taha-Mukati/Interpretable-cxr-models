# Vision-Language Models for Interpretable Chest X-Ray Diagnosis

**Author:** Muhammad Taha Mukati (Student ID: 730099705), University of Exeter
**Dataset:** MIMIC-CXR-JPG v2.1.0 (PhysioNet)

A systematic comparison of interpretability techniques applied to two biomedical vision-language models, **BioViL-T** and **CXR-CLIP**, on chest X-ray diagnosis. The project curates two MIMIC-CXR-JPG subsets, builds end-to-end inference pipelines for both models, implements four families of interpretability methods, and evaluates them with a four-metric framework.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Aims and Objectives](#aims-and-objectives)
3. [Repository Structure](#repository-structure)
4. [Setup and Prerequisites](#setup-and-prerequisites)
5. [Step 1: Dataset Curation](#step-1-dataset-curation)
6. [Step 2: Inference Pipelines](#step-2-inference-pipelines)
7. [Step 3: Interpretability Techniques](#step-3-interpretability-techniques)
8. [Step 4: Evaluation Framework](#step-4-evaluation-framework)
9. [Step 5: Running the Experiments](#step-5-running-the-experiments)
10. [Results Summary](#results-summary)
11. [Limitations](#limitations)
12. [Future Work](#future-work)
13. [References](#references)

---

## Project Overview

Vision-language models (VLMs) are promising tools for automated chest X-ray interpretation, but they largely operate as black boxes. This limits clinical adoption, and regulatory frameworks such as the EU AI Act and the FDA's Good Machine Learning Practice increasingly require transparency.

This project compares:

| Model | Image encoder | Text encoder | Latent space | Notes |
|-------|---------------|--------------|--------------|-------|
| **CXR-CLIP** | ResNet-50 | BioClinicalBERT | 512-d | Dual-encoder, prompt-based, zero-shot |
| **BioViL-T** | ResNet-50 + temporal Vision Transformer pooler | CXR-BERT | 120-d | Supports single and temporal (prior + current) inference |

## Aims and Objectives

**Core aims**

1. Curate two high-quality MIMIC-CXR-JPG subsets (single and temporal), removing duplicates, low-quality images/reports and incorrect views, and balancing pathology distributions.
2. Implement functional inference pipelines for BioViL-T and CXR-CLIP and obtain embeddings.
3. Implement four interpretability families: attention-based, attribution-based, concept-based, counterfactual.
4. Ensure a fair comparison between BioViL-T and CXR-CLIP (single inference) and between BioViL-T single and temporal modes.

**Stretch goals**

1. Expose per-layer attention, per-patch projections, per-token text projections and pre/post-fusion temporal representations from BioViL-T (enabling cross-modal alignment and four-quadrant temporal rollout).
2. Develop a four-metric evaluation framework (LQ, SA, F, CP) with quantitative success criteria.
3. Diagnose scorer-geometry confounds in perturbation-based faithfulness testing by reporting blur and zero-mask modes side by side.

## Repository Structure

```
interpretable-cxr-models/
├── Dataset/
│   ├── Single/                  # single_dataset.csv, images/, Reports/, image_paths.txt
│   └── Temporal/                # temporal_dataset.csv, images/, Reports/, image_paths_temporal.txt
├── models/
│   ├── biovil_t/                # cloned hi-ml image encoder sources + load.py (BioVILTWrapper)
│   └── cxr_clip/                # cloned CXR-CLIP repo + load.py
├── data_loader.py               # SingleInferenceDataset, TemporalDataset, CXRClipInferenceDataset
├── preprocessors/
│   ├── biovil_t.py
│   └── cxr_clip.py
├── collate/
│   └── biovil_t.py              # collate_biovil_single(), collate_biovil_temporal()
├── pipelines/
│   ├── biovil_t_pipeline.py
│   └── cxr_clip_pipeline.py
├── calibration/
│   └── calibration.py           # threshold fitting for CXR-CLIP
├── utils/
│   └── prompts.py               # CXR-CLIP prompt dictionary + embedding cache
├── interpretability/
│   ├── biovil_t/                # attention_rollout, cross_modal_alignment, grad_cam, tcav,
│   │                            # prototype_retrieval, counterfactuals, attention_control
│   └── cxr_clip/                # similarity_relevance, grad_cam, counterfactuals
├── evaluation/
│   ├── localisation_quality.py
│   ├── semantic_alignment.py
│   ├── faithfulness.py
│   └── clinical_plausibility.py
├── embeddings/biovil_t/         # cached chunked embeddings + manifest
└── run_experiments.ipynb        # main experiment notebook
```

> Adjust file names to match your actual repository layout.

## Setup and Prerequisites

1. **Python environment** with PyTorch, Hugging Face Transformers, pandas, NumPy, scikit-learn, Captum, matplotlib, Jupyter.
2. **MIMIC-CXR-JPG v2.1.0** access via [PhysioNet](https://physionet.org/) (credentialed access required): metadata table, CheXpert label table, free-text reports.
3. **CheXmask** MIMIC-CXR-JPG segmentation CSV (anatomical masks used as ground truth for the Pointing Game).
4. **Model sources and weights:**
   - BioViL-T image-encoder source from Microsoft's [hi-ml-multimodal](https://github.com/microsoft/hi-ml) repository (`hi-ml-multimodal/src/health_multimodal/image/model`), cloned into `models/biovil_t`. CXR-BERT tokenizer and text model are loaded from Hugging Face.
   - CXR-CLIP official repository cloned into `models/cxr_clip`, plus the **MIMIC-CXR (ResNet-50)** checkpoint listed in its `readme.md`.
5. Create the directory `interpretable-cxr-models` following the structure above.

## Step 1: Dataset Curation

Both subsets are built from MIMIC-CXR-JPG v2.1.0 using the metadata table, CheXpert labels and radiology reports.

### 1.1 Single-inference dataset (10,000 samples)

1. Load metadata and CheXpert tables into pandas (377,110 rows).
2. Keep **frontal views only** (PA or AP) via a `ViewPosition` mask, giving 243,334 studies; drop duplicate `dicom_id` rows.
3. Attach radiology reports with `load_report()`; drop rows with missing reports and reports under 100 characters.
4. Extract the **FINDINGS** and **IMPRESSION** sections; exclude impressions that are empty, underscores only, or under 50 characters.
5. Join CheXpert labels at the study level, then:
   - discard rows with uncertain (-1) or excluded labels;
   - remove rows where any of Enlarged Cardiomediastinum, Fracture, Lung Lesion, Pleural Other or Pneumonia is positive;
   - discard rows with no positive target label ("No Finding" serves as the control group).
6. Apply a **text-based exclusion** comparing impressions against a dictionary of excluded labels.
7. Deduplicate with `deduplicate_mimic_cxr()`: quality score from pixel area with a `PixelSpacing` penalty, view rank (PA preferred over AP), acquisition datetime for tie-breaking; keep the best row per study (23,340 rows remain).
8. **Stratified sampling** of 10,000 samples across the eight target conditions (pneumothorax, pleural effusion, support devices, consolidation, cardiomegaly, atelectasis, edema, no finding): 900 each for the rarer Pneumothorax and Consolidation, 1,200 each for the other six, with remaining slots filled by uniform sampling.
9. Save `Dataset/Single/single_dataset.csv` (columns: `subject_id`, `study_id`, `dicom_id`, CheXpert labels, `image_path`, `report_path`).
10. Export artefacts: GCS image URIs to `image_paths.txt` (used to download JPGs into `Dataset/Single/images/`), copy report `.txt` files into `Dataset/Single/Reports/`, and build absolute image and report path columns.

### 1.2 Temporal-inference dataset (2,000 pairs)

1. Reuse the single-pipeline filtering logic for consistency.
2. Build an `acquisition_time` column from `StudyDate` + zero-padded `StudyTime`; drop rows that fail parsing.
3. Deduplicate to **one image per study** (not per patient) with `deduplicate_to_one_image_per_study()`.
4. Construct temporal pairs: latest study per patient is the *current* image; join with the **closest prior** study with a gap of **1 to 180 days** (same-day pairs excluded).
5. Apply the excluded-text scan to current impressions only (reduces pairs from 8,252 to 7,099).
6. Priority stratified sampling: retain all Consolidation and Pneumothorax pairs, distribute the remainder of the 2,000-pair budget proportionally across other strata.
7. Save `Dataset/Temporal/temporal_dataset.csv` (columns: `subject_id`, `prior_study_id`, `prior_dicom_id`, `current_study_id`, `current_dicom_id`, `gap_days`, `current_report_text`, CheXpert labels, `prior_image_path`, `current_image_path`, `report_path`).
8. Export image URIs to `image_paths_temporal.txt`, download into `Dataset/Temporal/images/`, and copy current-study reports into `Dataset/Temporal/Reports/`.

## Step 2: Inference Pipelines

### 2.1 BioViL-T

1. Clone the hi-ml image-encoder sources into `models/biovil_t`.
2. **Datasets** (`data_loader.py`): `SingleInferenceDataset` and `TemporalDataset` return raw paths and strings; tensor construction is deferred to collate.
3. **Preprocessors** (`preprocessors/biovil_t.py`): image transform resizes to 512 then centre-crops to 448; text uses the CXR-BERT tokeniser from Hugging Face.
4. **Collate functions**: `collate_biovil_single()` and `collate_biovil_temporal()` (temporal stacks prior and current images separately and tokenises the current report only). Batch sizes: 8 (single), 4 (temporal).
5. **Model wrapper** (`BioVILTWrapper`): text model via `AutoModel.from_pretrained`; image model via `get_biovil_t_image_encoder()` (ResNet-50 + `VisionTransformerPooler`).
6. **Single inference** (`extract_single_outputs_chunked()` under `torch.inference_mode()`): global embedding, patch projections, per-block attention maps, CLS-projected text embedding, last hidden state and per-token projections, and image-text similarity.
7. **Temporal inference** (`extract_temporal_outputs_chunked()`): encodes prior and current independently, then fuses via the temporal module (captured with a forward pre-hook, pooler re-invoked with `return_attentions=True`). Four representations preserved: independent global embeddings, independent patch projections, current-image patch features, and fused patch projections. Three similarity scores: prior-text, current-text, prior-current.
8. **Chunked storage**: flush every 100 (single) or 50 (temporal) samples; `save_manifest()` records chunk layout; cache-aware loaders (`load_or_extract_biovil_t_single_embeddings()`, `load_or_extract_biovil_t_temporal_embeddings()`) skip extraction when a manifest exists.

### 2.2 CXR-CLIP

1. Clone the official repo into `models/cxr_clip` and download the MIMIC-CXR (ResNet-50) checkpoint.
2. Wrapper prepends the repo path to `sys.path` via `add_repo_to_path`.
3. Preprocessing via `load_cxr_clip_preprocessor()` (reads transform config from the checkpoint).
4. `CXRClipInferenceDataset` fills missing labels with zero and sets "No Finding" to zero when any abnormal label is positive.
5. Build prompts per pathology (training-style format) with **prompt ensembling** to produce positive and negative composite directions.
6. Helpers: `encode_image()`, `encode_text()` (L2-normalised), `score_prompts()` (cosine similarity).
7. `init_cxr_clip_pipeline()` builds the text-embedding cache; `extract_dataset_image_embeddings()` / `get_or_create_dataset_image_embeddings()` cache image embeddings to disk.
8. `score_image_embedding()` records mean positive similarity, mean negative similarity and **margin** (positive minus negative).
9. **Calibration** (`calibration/calibration.py`): subject-level 80/20 calibration/test split; `find_best_threshold()` sweeps 400 thresholds per pathology to maximise F1; thresholds are fitted on the calibration split only and applied unchanged to the test split; "No Finding" is derived as the negation of any abnormal prediction. Metrics: precision, recall, specificity, accuracy, balanced accuracy, F1.

## Step 3: Interpretability Techniques

### 3.1 BioViL-T (`interpretability/biovil_t/`)

| Family | Method | Details |
|--------|--------|---------|
| Attention | **Spatial attention rollout** | `fuse_heads()` → `normalize_attention()` (adds identity, row-normalises) → chained rollout; 14×14 grid upsampled bilinearly to 448×448 |
| Attention | **Temporal rollout** | `split_temporal_rollout()` into four quadrants (current→current, current→previous, previous→current, previous→previous) |
| Attention | **Cross-modal alignment** | Patch-token similarity cube; `token_heatmap()`, `phrase_heatmap()`, `topk_tokens_for_patch_and_token()` |
| Attribution | **Grad-CAM / Grad-CAM++** | Hooks on last `conv2d` layer; target = image-text similarity |
| Concept | **TCAV** | `train_linear_cav()` (logistic regression, Adam + weight decay); embedding TCAV and similarity TCAV (finite-difference directional derivative) |
| Concept | **Prototype retrieval** | `retrieve_prototypes()` top-k nearest neighbours by dot product |
| Counterfactual | **Textual** | Report substitution, measure score drop |
| Counterfactual | **Temporal** | True prior vs mismatched prior vs no prior |
| Counterfactual | **Latent** | Perturb embedding along CAV (e ± εc) and re-normalise |
| Counterfactual | **Attention suppression** | Top-k, head ablation, or temporal-quadrant ablation via `AttentionSuppressionConfig` |

### 3.2 CXR-CLIP (`interpretability/cxr_clip/`)

| Family | Method | Details |
|--------|--------|---------|
| Attention proxy | **Similarity-relevance map** | Layer-4 ResNet features scored against mean positive/negative prompt embeddings; margin reshaped and upsampled (CXR-CLIP has no self-attention) |
| Attribution | **Grad-CAM / Grad-CAM++** | Target scalar built from prompt embeddings; matched architectural depth with BioViL-T |
| Concept | **TCAV** | Same pathology set as BioViL-T for fair comparison |
| Counterfactual | **Negation** | Swap positive and negative prompt lists |
| Counterfactual | **Generic prompt** | Replace disease prompts with neutral templates |
| Counterfactual | **Latent** | Perturb embedding along CAV, rescore with cached prompts |
| Counterfactual | **Masking** | Mask top-k% relevant pixels and recompute margin |

**Baseline (both models):** Integrated Gradients via Captum.

## Step 4: Evaluation Framework

Four complementary metric families, each with a success criterion (`evaluation/`):

| Metric | Module | What it measures | Success criterion |
|--------|--------|------------------|-------------------|
| **Localisation Quality (LQ)** | `localisation_quality.py` | Pointing-game hit-rate against CheXmask anatomical proxy masks | Per-pathology rate exceeds uniform random-peak baseline |
| **Semantic Alignment (SA)** | `semantic_alignment.py` | Heatmap entropy, peak score (max/mean), sparsity (fraction > 0.7) plus TCAV concept sensitivity | TCAV > 0.5 (chance); > 0.7 = strong |
| **Faithfulness (F)** | `faithfulness.py` | Deletion/insertion AUC (11 steps), saliency vs 10 random masks at 20%, zero and blur modes | Insertion > deletion; saliency drop > random drop; win-rate > 50% |
| **Clinical Plausibility (CP)** | `clinical_plausibility.py` | Artefact checks: border focus, rib overfocus, centre bias, activation spread | Pass rate ≥ 90% (centre bias excluded as saturated) |

Implementation notes:

- **LQ:** `load_chexmask_mimic_csv()` streams the CheXmask CSV in 50,000-row chunks; `decode_rle()`, `build_chexmask_lookup()`, `attach_chexmask_to_samples()`, `build_proxy_mask()` map pathology to anatomical region; `pointing_game_mask()` records a hit if the heatmap peak lies in the mask.
- **SA:** `build_tcav_concepts()` partitions samples into positives/negatives and fits CAVs.
- **F:** scorer abstraction (`make_biovilt_scorer()`, `make_cxr_clip_scorer()`); `evaluate_faithfulness_for_heatmap()` composes deletion-insertion, zero-mask and blur-mask tests.
- **CP:** border mask of 10% (flag if > 50% activation), rib check (top 5% of rows > 40% activation), activation spread flagged below 0.01 or above 0.5.

## Step 5: Running the Experiments

All experiments run in `run_experiments.ipynb`:

1. Create the `interpretable-cxr-models` directory structure.
2. Confirm dataset curation produced valid `single_dataset.csv` and `temporal_dataset.csv` with correct image/report paths.
3. Run imports.
4. Extract BioViL-T embeddings with `load_or_extract_biovil_t_single_embeddings()` and `load_or_extract_biovil_t_temporal_embeddings()` (cached as chunks under `embeddings/biovil_t`).
5. Initialise CXR-CLIP with `init_cxr_clip_pipeline()` and run `run_cxr_clip_calibration_and_evaluation()` with `test_size=0.2`.
6. Attach CheXmask masks (`min_dice_mean = 0.7`) to the sample lists.
7. Run the three comparisons:
   - BioViL-T (single) vs CXR-CLIP (single)
   - BioViL-T (single) vs BioViL-T (temporal)
   - Attribution methods within each model
8. Run the evaluation framework:
   - LQ and SA are pathology-anchored (Cardiomegaly, Pleural Effusion, Pneumothorax); pools balanced with `balance_pathology_sample()`; SA uses 100 positive + 100 negative samples.
   - F and CP are model-level diagnostics (CP uses 60 heatmaps per model).
9. CXR-CLIP is evaluated on the held-out 20% test split; BioViL-T uses the full single-inference set.

## Results Summary

**Localisation Quality (Pointing Game)**

| Method | Overall |
|--------|---------|
| CXR-CLIP Grad-CAM++ | 0.3501 |
| CXR-CLIP Grad-CAM | 0.2622 |
| BioViL-T Grad-CAM | 0.1974 |
| BioViL-T Grad-CAM++ | 0.1657 |
| BioViL-T Spatial Rollout | 0.1299 |
| BioViL-T Temporal Rollout | 0.1017 |

CXR-CLIP's lead is driven by Cardiomegaly (a large central structure), partly reflecting centre bias. BioViL-T performs better on peripheral pathologies (e.g. spatial rollout 22.4% vs 7.2% on Pneumothorax).

**Semantic Alignment**

- Median entropy: BioViL-T **5.19** vs CXR-CLIP **10.69** (non-overlapping distributions).
- Peak score: 2.04 vs 2.17. Sparsity: 0.15 vs 0.18.
- Embedding TCAV: Pneumothorax BioViL-T 0.735 vs CXR-CLIP 0.590; Cardiomegaly 0.575 vs 0.610; Pleural Effusion 0.480 vs 0.455.

**Faithfulness**

- Insertion curve dominates deletion in both models.
- Under hard masking BioViL-T appears inverted (win-rate 6.7%), a methodological edge artefact rather than a faithfulness failure.
- Under blur, BioViL-T saliency drop 0.5781 vs random -0.0800 (win-rate 86.7%); CXR-CLIP 0.0514 vs -0.0104 (win-rate 53.0%).

**Clinical Plausibility**

- Border focus pass rate: BioViL-T **96.7%** vs CXR-CLIP **61.7%**.
- Excluding centre bias: BioViL-T passes 178/180 checks (98.9%), CXR-CLIP 156/180 (86.7%).

**Key takeaways**

1. Single-metric comparisons are biased; triangulate across the four metrics.
2. Perturbation-based faithfulness is confounded by scorer geometry and edge artefacts; report blur and zero modes side by side.
3. Architecture shapes the explanations available, not only their quality.

## Limitations

- CheXmask provides anatomical, not pathology-specific, masks, which biases the Pointing Game toward large central pathologies.
- CXR-CLIP is evaluated zero-shot with calibrated thresholds while BioViL-T uses cosine similarity directly; scorer scales differ by an order of magnitude, so comparisons are normalised and framed by direction of effect.
- All evaluation uses MIMIC-CXR-JPG; generalisation to CheXpert, NIH ChestX-ray14 or VinDr-CXR is untested.

## Future Work

- Replace CheXmask proxies with lesion-level ground truth (CheXlocalize, RSNA Pneumonia).
- Add a blinded radiologist evaluation (10-20 cases per pathology) to validate CP and SA.
- Run synthetic temporal-pair experiments with progressively perturbed priors.
- Build a learned ensemble of attention and gradient methods.
- Extend to other medical VLMs (CheXagent, RAD-DINO, MedCLIP).
- Mitigate CXR-CLIP's border-focus failure via attention-based pooling, anatomical priors or post-hoc spatial constraints.

## References

Key sources (see the dissertation for the full list):

- You et al., *CXR-CLIP: Toward Large Scale Chest X-ray Language-Image Pre-training*, 2023.
- Boecking et al., *Making the Most of Text Semantics to Improve Biomedical Vision-Language Processing* (BioViL), 2022.
- Bannur et al., *Learning to Exploit Temporal Structure for Biomedical Vision-Language Processing* (BioViL-T), 2023.
- Selvaraju et al., *Grad-CAM*, 2019.
- Chefer et al., *Transformer Interpretability Beyond Attention Visualization*, 2021.
- Johnson et al., *MIMIC-CXR Database*, PhysioNet, v2.1.0.
