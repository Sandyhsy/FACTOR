# Multimodal Fashion Recommendation Framework Using Fine-Grained Attribute-Aware Retrieval

**Shao-Yu Huang**  
Master of Science in Applied Data Science  
San José State University · May 2026

**Thesis Committee:** Mohammad Masum (Chair), Guannan Liu, Saptarshi Sengupta

---

## Abstract

Accurate garment attribute recognition is essential for fashion recommendation systems, trend analysis, and large-scale retrieval applications. However, most current approaches rely on coarse visual features and fail to capture the fine-grained semantic attributes that determine clothing style and compatibility. This research investigates visual-to-textual garment labeling using CLIP and extends its capabilities through fine-tuning on a curated subset of the Polyvore dataset. We introduce **FACTOR** (*Feature-Aware Contextual Optimization and Refinement*), a two-stage pipeline that employs linear probing over CLIP's embedding structure to predict five core attributes—category, subcategory, color, material, and pattern—while preserving CLIP's rich semantic representations through frozen encoders and lightweight classification heads.

The framework further incorporates style as a higher-level semantic attribute and evaluates attribute-aware retrieval under image-only, text-only, native multimodal, and late-fusion settings. FACTOR achieves 93% category accuracy, 72% subcategory accuracy, and semantic similarity scores of 87%, 87%, and 85% for color, material, and pattern, respectively. By integrating CLIP-based semantic similarity into the evaluation pipeline, this work provides a more nuanced and interpretable assessment of attribute recognition that captures partial correctness and semantic alignment rather than strict lexical matching. Together, these contributions advance fine-grained fashion understanding and provide a practical, explainable framework for multimodal fashion recommendation.

> Full thesis: [`Paper.pdf`](Paper.pdf)

---

## 1. Introduction

### 1.1 Background and Motivation

E-commerce has transformed how people discover and consume fashion, yet existing recommendation systems primarily suggest items similar to prior purchases rather than addressing users' broader styling needs. When users specify detailed preferences—style, occasion, or material—these queries often fail to align with seller-provided metadata. Systems that rely on global visual similarity retrieve visually related items but miss the underlying semantic intent and stylistic preference.

This limitation motivates a shift from **visual perception** to **semantic understanding**: transforming garment images into structured, interpretable attributes that bridge user intent and system output.

### 1.2 Problem Statement

Three interconnected challenges define this work:

1. **Fine-grained attribute extraction.** Given a single garment image, the system must jointly predict multiple attributes (category, subcategory, color, material, pattern). Prior work rarely treats this as a central problem within vision-language models.
2. **Higher-level style representation.** Style captures abstract aesthetic intent that low-level attributes alone cannot express, yet it is often absent from traditional recommendation pipelines.
3. **Attribute-aware retrieval.** Recommendation should evaluate similarity at the attribute level—not only through global visual embeddings—enabling semantically aligned and interpretable results.

### 1.3 Research Objectives

| Objective | Description |
|-----------|-------------|
| Attribute prediction | Build a robust model that jointly predicts five core garment attributes in single-label and multi-label settings |
| Style modeling | Extend the representation with a 17-class style classifier to capture higher-level fashion semantics |
| Attribute-aware retrieval | Design a retrieval mechanism that leverages structured attributes alongside visual embeddings |
| Semantic evaluation | Employ CLIP-based semantic similarity to assess prediction quality beyond exact lexical matching |

---

## 2. Proposed Framework

The framework comprises three sequential stages that move from **image understanding** to **semantic representation** and finally to **recommendation generation**.

```
┌─────────────────┐     ┌──────────────────────┐      ┌─────────────────────────────────────────┐
│  Stage 1        │     │  Stage 2             │      │  Stage 3                                │
│  FACTOR         │ ──▶ │  Style Classification│ ──▶ │  Attribute-Aware Retrieval              │
│  (5 attributes) │     │  (17 style classes)  │      │  (image / text / fusion / Multimodal)   │
└─────────────────┘     └──────────────────────┘      └─────────────────────────────────────────┘
```

### Stage 1: FACTOR — Fine-Grained Attribute Extraction

FACTOR is a two-stage learning framework built upon CLIP ViT-B/32:

- **Stage 1 (Color-aware encoder refinement).** The visual encoder is fine-tuned on 1,000 color-focused samples from the Second-Hand Fashion dataset, targeting the most frequently misclassified color labels identified through error analysis.
- **Stage 2 (Multi-attribute probing).** The refined encoder is frozen; independent linear heads predict category and subcategory (single-label) and color, material, and pattern (multi-label) on the Polyvore dataset.

Among single-stage adaptation strategies—zero-shot CLIP, FashionCLIP, linear probe, prompt tuning, hybrid tuning, and full fine-tuning—**linear probing consistently delivers the strongest and most stable performance**. FACTOR's two-stage design further improves multi-label semantic similarity while maintaining high single-label accuracy.

### Stage 2: Style Classification

A separate 17-class style classifier captures higher-level aesthetic intent. Experiments compare ResNet50, OpenFashionCLIP, and FashionCLIP under multiple adaptation strategies (partial fine-tuning, prompt classification, LoRA, full fine-tuning). **Partial fine-tuning** provides the best trade-off between performance, stability, and computational efficiency.

### Stage 3: Attribute-Aware Multimodal Retrieval

Each item is represented by six attributes: category, subcategory, color, material, pattern, and style. Retrieval is evaluated across four paradigms:

| Setting | Models | Role |
|---------|--------|------|
| Image-only | FashionCLIP, OpenFashionCLIP, SigLIP2 | Visual similarity baseline |
| Text-only | BGE-M3, Voyage | Semantic matching via structured attribute text |
| Native multimodal | Voyage Multimodal, SigLIP, BGE-VL | Joint image-text encoding |
| Late fusion | FashionCLIP + Voyage | Score fusion and reciprocal rank fusion |

---

## 3. Datasets

| Dataset | Role | Key Details |
|---------|------|-------------|
| **Polyvore** (Han et al., 2017) | Attribute learning & retrieval gallery | 35,990 cleaned instances; 80/10/10 split; five attributes with label library |
| **Second-Hand Fashion** (Nauman, 2024) | Color-aware encoder refinement | 28,248 training samples; multi-label color, material, pattern |
| **FashionStyle14K** (Takagi et al., 2017) | Style classification (core) | 17 style classes; Japanese fashion aesthetics |
| **Fashion Product Images** (Aggarwal, 2019) | Style augmentation & query images | Structured product images with clean backgrounds |
| **Pinterest** (collected) | Query images | Real-world user-style inputs for retrieval evaluation |

**Retrieval setup.** Query images are drawn from Pinterest and Fashion Product Images. The retrieval gallery is built from the Polyvore training split across **five random seeds** to reduce sampling bias and assess robustness.

**Attribute taxonomy.**

| Attribute | Type | Examples |
|-----------|------|----------|
| Category | Single-label | tops, bottoms, shoes, outerwear |
| Subcategory | Single-label | sweater, jeans, heels, blazer |
| Color | Multi-label | black, gold, ecru, denim stonewash |
| Material | Multi-label | cotton, silk, leather, polyester |
| Pattern | Multi-label | floral, striped, plain, jacquard |
| Style | Single-label | formal, street, minimalist, vintage |

---

## 4. Evaluation Metrics

### 4.1 Attribute Prediction

- **Single-label:** Accuracy, Precision, Recall, F1, Semantic Similarity
- **Multi-label:** Relaxed Match, Semantic Similarity (CLIP-based embedding cosine similarity)

### 4.2 Retrieval Ranking

Precision@k, Recall@k, MRR@k, MAP@k, nDCG@k

### 4.3 Attribute Matching Quality

| Metric | Description |
|--------|-------------|
| **AMR** (Attribute Match Rate) | Fraction of retrieved items with semantically matching attribute values |
| **EAM** (Exact Attribute Match) | Token-set exact matching for single- and multi-value attributes |
| **SS** (Semantic Similarity) | CLIP-based similarity between query and retrieved attribute values |
| **WAS** (Weighted Attribute Score) | Attribute-weighted aggregate consistency across all six attributes |

---

## 5. Key Findings

### Attribute Recognition (FACTOR)

| Attribute | Accuracy / Relaxed Match | Semantic Similarity |
|-----------|--------------------------|---------------------|
| Category | 93% | 99% |
| Subcategory | 72% | 95% |
| Color | 43% (relaxed) | 87% |
| Material | 40% (relaxed) | 87% |
| Pattern | 25% (relaxed) | 85% |

Linear probing outperforms prompt tuning, hybrid tuning, and full fine-tuning. Full fine-tuning exhibits representation drift and degraded multi-label performance. FACTOR's color-aware two-stage design yields consistent improvements over single-stage linear probing.

### Style Classification

Partial fine-tuning of OpenFashionCLIP achieves the strongest overall performance (macro F1 ≈ 86.1%), surpassing ResNet50 (≈ 81.1%) while requiring substantially fewer trainable parameters. Results remain stable across five data splits.

### Attribute-Aware Retrieval

- **Text-only retrieval** achieves the highest Weighted Attribute Score (≈ 99%), demonstrating that accurate structured semantics alone drive effective recommendation.
- **Structured text** performs comparably to sentence-based templates, confirming that retrieval quality depends on semantic accuracy rather than linguistic fluency.
- **Image-only models** lag on multi-label attributes (color, pattern) and abstract concepts (style).
- **Fusion and multimodal methods** provide a balanced trade-off between visual consistency and semantic precision.

---

## 6. Repository Structure

This repository implements the complete experimental pipeline described in the thesis. Notebooks should be executed in the order below.

| Notebook | Thesis Chapter | Description |
|----------|----------------|-------------|
| [`01_EDA_Preprocessing.ipynb`](01_EDA_Preprocessing.ipynb) | Ch. 3 | Exploratory data analysis, LLaMA3-based attribute extraction from Polyvore metadata, label library construction, and train/validation/test splitting |
| [`02_FACTOR_Methodology.ipynb`](02_FACTOR_Methodology.ipynb) | Ch. 5 | FACTOR development: CLIP/FashionCLIP baselines, linear probe, prompt tuning, hybrid tuning, full fine-tuning, and two-stage FACTOR training |
| [`03_FACTOR_ErrorAnalysis.ipynb`](03_FACTOR_ErrorAnalysis.ipynb) | Ch. 5 | Systematic error analysis on color prediction; identifies misclassification patterns that motivate the color-aware refinement stage |
| [`04_StyleClassification_Methodology.ipynb`](04_StyleClassification_Methodology.ipynb) | Ch. 6 | Style classification across ResNet50, OpenFashionCLIP, and FashionCLIP; baseline and fine-tuning experiments with robustness evaluation over five splits |
| [`05_Attribute_Inference.ipynb`](05_Attribute_Inference.ipynb) | Ch. 7 (Stage 1) | Inference bridge: applies trained FACTOR and style models to query and gallery datasets; generates six-attribute CSV representations for retrieval |
| [`06_Recommend_Methodology.ipynb`](06_Recommend_Methodology.ipynb) | Ch. 7 (Stage 2–3) | Attribute-aware retrieval experiments: image-only, text-only, native multimodal, and fusion-based recommendation with comprehensive evaluation |

### Experimental Pipeline

```
01 EDA & Preprocessing
        │
        ▼
02 FACTOR Methodology ──▶ 03 Error Analysis
        │                        │
        └────────┬───────────────┘
                 ▼
        04 Style Classification
                 │
                 ▼
        05 Attribute Inference
                 │
                 ▼
        06 Recommendation & Retrieval
```

---

## 7. Notebook Guide

### `01_EDA_Preprocessing.ipynb`

Prepares the Polyvore dataset for attribute learning. Raw item metadata is cleaned and standardized into five structured labels (category, subcategory, color, material, pattern) using LLaMA3 for multi-label extraction from unstructured descriptions. A label library is constructed, class distributions are analyzed, and data is split into training (80%), validation (10%), and test (10%) sets with balanced class proportions.

### `02_FACTOR_Methodology.ipynb`

Develops and compares CLIP-based adaptation strategies for fine-grained attribute prediction:

- Zero-shot CLIP and FashionCLIP baselines
- Linear probe on frozen embeddings
- Prompt tuning, hybrid tuning, and full fine-tuning
- Two-stage FACTOR framework with color-aware encoder refinement

Evaluation reports accuracy, precision, recall, F1, relaxed match, and CLIP-based semantic similarity across five random seeds.

### `03_FACTOR_ErrorAnalysis.ipynb`

Conducts systematic error analysis focused on the color attribute—the most error-prone prediction target. False-positive distributions are extracted across all adaptation strategies to identify systematic biases (e.g., over-prediction of "black" and confusion among subtle hues such as ecru). Findings directly inform the Stage 1 color-aware fine-tuning in FACTOR.

### `04_StyleClassification_Methodology.ipynb`

Trains and evaluates 17-class style classifiers using ResNet50, OpenFashionCLIP, and FashionCLIP. Compares baseline performance with partial fine-tuning, prompt classification, LoRA, and full fine-tuning. Repeats all experiments across five data splits and reports mean ± std for robustness assessment.

### `05_Attribute_Inference.ipynb`

Serves as the inference bridge between model training and retrieval. Loads trained FACTOR checkpoints and the style classifier, runs inference on recommendation query and retrieval gallery images, and merges predictions into unified six-attribute CSV files. Generates both structured and sentence-form text representations for downstream retrieval experiments.

### `06_Recommend_Methodology.ipynb`

Implements the final attribute-aware recommendation stage. Compares four retrieval paradigms (image-only, text-only, native multimodal, fusion) across five seed-based galleries. Computes ranking metrics (Precision@k, Recall@k, MRR@k, MAP@k, nDCG@k) and attribute-level quality metrics (AMR, EAM, semantic similarity, WAS) to evaluate both visual relevance and semantic faithfulness.

---

## 8. Limitations and Future Work

**Limitations.**

- Attribute annotation relies on LLaMA3 extraction from product descriptions, which may introduce parsing noise and semantic granularity inconsistencies.
- Color-aware refinement uses 1,000 sampled images from selected confusing categories, yielding localized rather than global color-space improvement.
- Polyvore exhibits long-tailed class distributions; style datasets are limited in gender and cultural diversity.

**Future directions.**

- Extend toward full outfit recommendation and compatibility modeling
- Incorporate personal wardrobe data for sustainable, reuse-oriented styling
- Introduce occasion-based categories (wedding, business, casual) for context-aware recommendation
- Expand the query pool and systematically optimize fusion hyperparameters
- Integrate large language models for conversational query understanding and explanation generation

---

## 9. Citation

If you use this code or reference this work, please cite:

```bibtex
@mastersthesis{huang2026factor,
  author       = {Huang, Shao-Yu},
  title        = {Multimodal Fashion Recommendation Framework Using Fine-Grained Attribute-Aware Retrieval},
  school       = {San Jos{\'e} State University},
  year         = {2026},
  type         = {Master's Thesis},
  address      = {San Jos{\'e}, CA},
  month        = {May}
}
```

---

## Author & Copyright

© 2026 Shao-Yu Huang. All rights reserved.

**Author:** Shao-Yu Huang  
**University:** San José State University  
**Department:** Applied Data Science

**Committee:**
- Dr. Mohammad Masum (Chair)
- Dr. Guannan Liu (Committee Member)
- Dr. Saptarshi Sengupta (Committee Member)
