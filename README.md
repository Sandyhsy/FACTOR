# 0. Experiment Summary (Total = 11)

## 0.1 Baseline (5)

### Text-only (2)

* **T1: Text-only / BGE**
* **T2: Text-only / Voyage**

### Image-only (3)

* **I1: Image-only / GCL**
* **I2: Image-only / SigLIP**
* **I3: Image-only / FashionCLIP**

## 0.2 Proposed (Fusion, 6)

Fusion = **2 (Text) × 3 (Image) = 6**

* **F1: BGE + GCL**
* **F2: BGE + SigLIP**
* **F3: BGE + FashionCLIP**
* **F4: Voyage + GCL**
* **F5: Voyage + SigLIP**
* **F6: Voyage + FashionCLIP**

---

# 1. Data Setup

## 1.1 Query Set

* Dataset: `recom_dataset`
* Query images: **50**
* Each query image → Factor model predicts 5 attributes:
  * `category, sub_category, color, pattern, material`
* Generate canonical sentence from predicted attributes (template-based)

## 1.2 Retrieval Database (per seed)

* Seeds: 5
* Database per seed = that seed’s `train/images`
* For each db image: 
    * apply the **same embedding pipeline** as query (text-only / image-only / fusion)
    * ensure the **same projection and same final dimension** are used

---

# 2. Pipeline Overview

## 2.1 End-to-end Flow (per query image)

1. **Input image**
2. **Factor model** → predicted attributes
    Output: (category, sub_category, color, pattern, material)
3. **Canonical sentence generation** (template-based)
4. **Embedding branch** (one of 11 experiments)
5. **Dimension alignment** (for fusion projection to `R^1024` + L2 normalize)
6. **Retrieval** using cosine similarity
7. **Evaluation** (Relaxed Match, AMR, WAM, Attribute-specific Recall@K, MAP)
8. **Logging + Save outputs** under `output/`


---

# 3. Embeddings & Dimension Alignment (Critical)

## 3.1 Shared embedding dimension

* **Fusion / retrieval dimension: `D_fuse = 1024`**
* All text embeddings and image embeddings must be projected into `R^1024`
* Always apply **L2-normalization** after projection

## 3.2 Projection rule (query and database must share the same projector)

For any embedding `e ∈ R^d`:

* `e_proj = normalize( W · e )`, where `W ∈ R^(1024×d)`

> This ensures **query fusion embedding** and **database fusion embedding** have identical dimensions and are comparable.

---

# 4. Experiment Definitions (11 Groups)

## 4.1 Baseline: Text-only (2)

### (T1) Text-only: BGE

* Input: canonical sentence
* `t = normalize( Proj_BGE( BGE(sentence) ) )` (BGE 768 → project → 1024)

### (T2) Text-only: Voyage

* Input: canonical sentence
* `t = normalize( Voyage(sentence, output_dimension=1024) )`

---

## 4.2 Baseline: Image-only (3)

### (I1) Image-only: GCL

* Input: image
* `v = normalize( Proj_GCL( GCL(image) ) )`

### (I2) Image-only: SigLIP

* Input: image
* `v = normalize( Proj_SigLIP( SigLIP(image) ) )`

### (I3) Image-only: FashionCLIP

* Input: image
* `v = normalize( Proj_FashionCLIP( FashionCLIP(image) ) )`

---

## 4.3 Proposed: Fusion (6)

### Fusion rule (Late Fusion)

Given:

* `t ∈ R^1024` from text encoder
* `v ∈ R^1024` from image encoder

Define fusion embedding:

* `f = normalize( λ·t + (1-λ)·v )`
* Default: `λ = 0.5`

### Fusion combinations (6)

* **F1**: BGE + GCL
* **F2**: BGE + SigLIP
* **F3**: BGE + FashionCLIP
* **F4**: Voyage + GCL
* **F5**: Voyage + SigLIP
* **F6**: Voyage + FashionCLIP

---

# 5. Retrieval Scoring

Compute scores between query embedding and each db embedding:

* Cosine similarity

---

# 6. Evaluation Metrics

## 6.1 Relaxed Match (binary relevance)
Rule :
* A retrieved item is relevant if it matches (category AND sub_category AND color) with the query predictions.
* Report relaxed match at Top3 
* Save relaxed match Top10 to support “failure cases = Recall@10 = 0”.

## 6.2 AMR - Attribute Match Rate 
AMR measures how many attributes match between a query and a candidate, allowing partial credit.
### Per-attribute score: 
Let `s_a ∈ [0,1]` for each attribute:
* category, sub_category, color, pattern: exact match → {0,1}
* material: partial match allowed (e.g., Jaccard over material tokens)
### Definition: `AMR = (s_cat + s_subcat + s_color + s_pattern + s_material) / 5`
### Example:
Query canonical attributes:

* color = white, pattern = striped, sub_category = blouse, category = tops, material = {cotton, polyester}

Candidate 1:

* color = white (1)
* pattern = striped (1)
* sub_category = blouse (1)
* category = tops (1)
* material = {cotton} → partial (1/2 = 0.5)

AMR = (1 + 1 + 1 + 1 + 0.5) / 5 = **0.90**

Candidate 2:

* color = blue (0)
* pattern = striped (1)
* sub_category = blouse (1)
* category = tops (1)
* material = {cotton, polyester} (1)

AMR = (0 + 1 + 1 + 1 + 1) / 5 = **0.80**

## 6.3 WAM - Weighted Attribute Matching

Weights:

* category 30%
* sub_category 20%
* color 25%
* material 15%
* pattern 10%

WAM:
* `WAM = 0.30*s_cat + 0.20*s_subcat + 0.25*s_color + 0.15*s_mat + 0.10*s_pattern`

## 6.4 Attribute-Specific Recall@K (K ∈ {1,3,5,10})

For each attribute `a` ∈ {category, sub_category, color, material, pattern}:

* Define relevance set:
  * `GT_a = { db_item | db_item[a] == query_pred[a] }`
* Compute Recall@K independently for each attribute:
  * `Recall@K(a) = |TopK ∩ GT_a| / |GT_a|`

## 6.5 MAP - Mean Average Precision

MAP requires a **binary relevance** definition per query.

### Relevance used for MAP (recommended and consistent)

Use the same Relaxed Match relevance:

* relevant if **(category AND sub_category AND color)** matches.

### Computation

* For each query: compute **AP** based on ranked list and relevant set.
* For each seed: compute **MAP = mean(AP over 50 queries)**.
* Final reporting: **mean ± std across 5 seeds**.

---

# 7. Required Saved Outputs (All under `output/`)

```
output/
├── text-only/
│   ├── BGE/
│   │   ├── retrieval_results.csv
│   │   ├── model_performance_summary.csv
│   │   └── attribute_recall_breakdown.csv
│   └── Voyage/
├── image-only/
│   ├── GCL /
│   │   ├── retrieval_results.csv
│   │   ├── model_performance_summary.csv
│   │   └── attribute_recall_breakdown.csv
│   ├── SigLIP /
│   └── FashionCLIP /
├── fusion/
│   ├── BGE + GCL /
│   │   ├── retrieval_results.csv
│   │   ├── model_performance_summary.csv
│   │   └── attribute_recall_breakdown.csv
│   ├── BGE + SigLIP /
│   ├── BGE + FashionCLIP /
│   ├── Voyage + GCL /
│   ├── Voyage + SigLIP /
│   └── Voyage + FashionCLIP /
├── image_predictions.csv
└── mean_std_report.csv
```

---

# 8. CSV Schemas (Tables)

## 8.1 image_predictions.csv

**Per query image Factor model outputs**

* `image`
* `category`
* `sub_category`
* `color`
* `pattern`
* `material`

## 8.2 retrieval_results.csv

**Top-3 retrieval per (image, model, seed)**

* `image`
* `model`
* `seed`
* `top1_image_id`, `top1_score`, `top1_description`
* `top2_image_id`, `top2_score`, `top2_description`
* `top3_image_id`, `top3_score`, `top3_description`

## 8.3 model_performance_summary.csv

**Overall performance per (model, seed)**

* `model`
* `seed`
* `Recall@1`, `Recall@3`, `Recall@5`, `Recall@10`
* `MAP`
* `AMR`
* `WAM`

## 8.4 attribute_recall_breakdown.csv

**Five attributes × Recall@{1,3,5,10}**
Per model, per seed, per attribute

* `model`
* `seed`
* `attribute` ∈ {category, sub_category, color, material, pattern}
* `Recall@1`, `Recall@3`, `Recall@5`, `Recall@10`

## 8.5 mean_std_report.csv 

* `model` ∈ {BGE,Voyage,GCL,SigLIP,FashionCLIP,BGE+GCL,BGE+SigLIP,BGE+FashionCLIP,Voyage+GCL,Voyage+SigLIP,Voyage+FashionCLIP}
* `Recall@1 (mean±std)`
* `Recall@3 (mean±std)`
* `Recall@5 (mean±std)`
* `Recall@10 (mean±std)`
* `MAP (mean±std)`
* `AMR (mean±std)`
* `WAM (mean±std)`

---

# 9. Printed Outputs 

For each query image:

1. **Image name**
2. **Prediction row**
   `image,category,sub_category,color,pattern,material`
3. **Retrieval Top-3 for each seed** (same fields as `retrieval_results.csv`)
4. **Attribute recall breakdown** (table)
5. **Mean±Std report** (final paper table)

At last print:
1. mean_std_report
2. Text vs Image vs Fusion
3. Top-K Failure Cases
4. Embedding Time Cost
---

# 10. Additional Analyses 

## 10.1 [Ablation] Text vs Image vs Fusion

Store Recall@1, @5, @10 for:

* Text-only: {BGE, Voyage}
* Image-only: {GCL, SigLIP, FashionCLIP}
* Fusion: {6 combinations}

## 10.2 [Error Analysis] Top-K Failure Cases

* Automatically list query images with **Recall@10 = 0**
* Save their image paths for qualitative inspection

## 10.3 [Efficiency] Embedding Time Cost

Measure time cost per model/branch:

* factor model time
* text embedding time (BGE, Voyage)
* image embedding time (GCL, SigLIP, FashionCLIP)
* projection + fusion time
* retrieval time
