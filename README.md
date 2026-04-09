# Multimodal Fashion Recommendation Framework Using Fine-Grained Attribute-Aware Retrieval
By: Shao-Yu Huang

In summary, the research follows the same logic as the proposed framework. 
The first notebook develops FACTOR for fine-grained attribute extraction, the second notebook applies the trained attribute and style models to downstream datasets, the third and fourth notebooks establish and validate the style classification module, and the final notebook performs attribute-aware fashion retrieval under image-only, text-only, fusion, and native multimodal settings. 
Together, these notebooks form a complete experimental pipeline from garment understanding to recommendation generation.


## Codebase Overview

## 1. Proposed Framework

This project proposes a **multimodal fashion recommendation framework** that connects **fine-grained garment understanding** with **attribute-aware retrieval**. Instead of relying only on global visual similarity, the system first predicts structured semantic attributes from a clothing image, then extends the representation with a higher-level style label, and finally uses these outputs for retrieval and recommendation.

The full pipeline contains three major stages:

1. **FACTOR: Fine-Grained Attribute Extraction**
   This stage predicts low-level garment semantics from a single clothing image, including:

   * category
   * subcategory
   * color
   * material
   * pattern

2. **Style Classification**
   This stage predicts a higher-level semantic style label from the same garment image.
   Style complements the five FACTOR attributes by capturing more abstract fashion concepts that are not directly visible from attribute labels alone.

3. **Attribute-Aware Retrieval / Recommendation**
   This stage uses predicted structured attributes, image features, or both to retrieve similar items from a gallery.
   The goal is not only to find visually similar products, but also to return items that are **semantically aligned** with the query garment.

Overall, the project moves from **image understanding** to **semantic representation**, and finally to **recommendation generation**, forming an end-to-end pipeline for interpretable fashion retrieval. 

---

## 2. Notebook-by-Notebook Explanation

---

# 01_FACTOR.ipynb

## Purpose

This notebook develops the **FACTOR module**, which is the first stage of the proposed framework. Its main purpose is to learn fine-grained fashion attributes from garment images and compare several adaptation strategies on CLIP-based models.

The notebook is not just one model run. It is an experimental notebook that compares multiple approaches, including:

* CLIP zero-shot baseline
* FashionCLIP baseline
* Linear Probe
* Prompt Tuning
* Hybrid Tuning
* Full Fine-Tuning
* Two-stage / color-enhanced refinement

This notebook is therefore the **core model development notebook** for attribute prediction.

## Dataset Used

This notebook mainly uses the **Polyvore-based attribute dataset** across multiple random splits.
From your slide and notebook structure, the attributes include:

* category
* sub_category
* color
* material
* pattern

The notebook also uses **5 seeds / 5 splits** for robustness and fair comparison.

## Input

* Single garment image
* Corresponding training labels for the five attributes
* Label library for all valid attribute classes

## Output

* Predicted attribute labels for each image
* Evaluation reports for each seed
* Aggregated mean ± std tables across seeds
* Saved prediction CSV / JSON files
* Performance comparison across different tuning strategies

## What the code is doing

### Part A. Environment and data loading

The notebook first mounts Google Drive, copies image folders locally, and loads the label library. This part prepares the experiment environment and keeps training I/O efficient.

### Part B. CLIP and FashionCLIP baselines

The notebook tests pretrained vision-language models as initial baselines. These sections show how well pretrained models can perform before task-specific adaptation.

### Part C. Linear Probe

A lightweight classifier head is trained on top of frozen image embeddings.
This section is important because it provides a strong and stable adaptation baseline.

### Part D. Prompt Tuning / Hybrid Tuning / Full Fine-Tuning

These sections compare different transfer learning strategies:

* Prompt tuning adapts the text side
* Hybrid tuning combines prompt-related adjustments with classification learning
* Full fine-tuning updates more of the model parameters

This allows you to study the trade-off between efficiency, stability, and final accuracy.

### Part E. Two-stage design and color refinement

This part addresses the observation that **color is the most error-prone attribute**.
The code adds an additional refinement stage to improve difficult color-related prediction performance.

### Part F. Inference and evaluation

The final section runs prediction on the test split and computes:

* accuracy
* precision
* recall
* F1
* relaxed matching for multi-label attributes
* semantic similarity

This is especially important because your thesis does not rely only on exact lexical match, but also evaluates whether predicted labels are **semantically close** to the ground truth.

## Small Summary from the Results

This notebook supports the main conclusion that:

* **Linear probe is the strongest and most stable single-stage method**
* Prompt tuning and hybrid tuning do not consistently outperform linear probing
* Full fine-tuning can become unstable
* Additional targeted refinement is particularly useful for **color**

So for the thesis narrative, this notebook shows that **high-quality attribute prediction is the foundation of the whole recommendation system**.

---

# 02_attributes_classifiactions.ipynb

## Purpose

This notebook is the **inference bridge notebook** between model training and recommendation.
Its purpose is to run the trained FACTOR model and the trained style classifier on the datasets used later in recommendation and retrieval.

In other words, this notebook converts raw images into the **final six-attribute structured representation** used by the retrieval stage.

## Dataset Used

This notebook uses the datasets needed for downstream retrieval and recommendation, including:

* recommendation query images
* retrieval gallery images
* outputs from multiple seed folders

It also loads:

* the trained FACTOR checkpoints
* the trained style classification model
* the factor label library

## Input

* Image folders for query/retrieval datasets
* Trained FACTOR checkpoints
* Trained style classification checkpoint
* Label mapping files

## Output

* FACTOR predictions for each image
* Style predictions for each image
* Merged six-attribute CSV files
* Description text fields such as structured text or sentence-form text for retrieval use

## What the code is doing

### Part A. Load trained FACTOR model

This part restores the best attribute prediction model and prepares it for inference.

### Part B. Load trained style model

This section loads the trained style classifier used to predict the sixth attribute.

### Part C. Collect dataset images

The notebook gathers images from the recommendation and retrieval datasets and standardizes them into a unified inference format.

### Part D. Run FACTOR inference

The code predicts:

* category
* sub_category
* color
* material
* pattern

### Part E. Run style inference

The code predicts the final style class for each image.

### Part F. Merge predictions

The predicted five FACTOR attributes and the style label are merged into one row per image.

### Part G. Generate text representation

The notebook also adds a description column, such as:

* structured form:
  `category, subcategory, color, material, pattern, style`
* sentence form:
  `A {color} {material} {pattern} {subcategory} {category} in a {style} style.`

This is critical for your retrieval experiments because the text-only and multimodal models need standardized textual input.

## Small Summary from the Results

This notebook does not mainly introduce a new model; instead, it prepares the **semantic representation layer** for recommendation.
Its main contribution is that it transforms raw images into a consistent six-attribute representation that can be reused across:

* text-only retrieval
* fusion retrieval
* native multimodal retrieval

So in your thesis, this notebook can be described as the **semantic annotation generation stage for downstream recommendation**.

---

# 03_style_classify.ipynb

## Purpose

This notebook develops the **style classification stage**, which is the second main component of the framework.
Its goal is to classify each garment image into one of the defined style categories and compare different model families and adaptation strategies.

This notebook contains both:

* baseline experiments
* fine-tuning experiments

## Dataset Used

This notebook uses the **style classification dataset**, mainly built from:

* FashionStyle14K
* Fashion Product Images
* additional online images

The final task is a **17-class style classification problem**, based on your slide description. 

## Input

* Single garment image
* Style label
* Train / validation / test CSV files
* Image directories

## Output

* Trained style classification models
* Validation and test predictions
* Evaluation tables
* Comparison figures
* Confusion analysis outputs

## What the code is doing

### Part A. Baseline experiments

The notebook first builds baseline models using:

* ResNet50
* OpenFashionCLIP
* FashionCLIP

This gives a direct comparison between a CNN backbone and CLIP-style vision-language backbones.

### Part B. Dataset class and dataloaders

This section standardizes image reading, label mapping, and preprocessing for different backbones.

### Part C. Model definitions

The notebook defines:

* ResNet classifier
* OpenFashionCLIP classifier
* FashionCLIP classifier

### Part D. Training and inference

The code trains each model, saves the best checkpoints, and then performs inference on validation and test data.

### Part E. Fine-tuning experiments

This section expands the comparison to multiple adaptation methods such as:

* partial fine-tuning
* prompt-based classification
* LoRA
* full fine-tuning

This is important because style is more abstract than low-level attributes, so the notebook tests which transfer strategy best captures these high-level semantics.

### Part F. Evaluation and visualization

The notebook calculates:

* accuracy
* macro precision
* macro recall
* macro F1
* confusion-related summaries

## Small Summary from the Results

The overall finding is consistent with your slide:

* **Frozen baselines are not enough for fine-grained style recognition**
* **Controlled adaptation works better than full updating**
* **Partial fine-tuning gives the strongest and cleanest performance**

So for your thesis writing, the key message is:

> Style requires higher-level semantic adaptation, and partial fine-tuning gives the best trade-off between performance and stability.

---

# 03_robustness_style_classify.ipynb

## Purpose

This notebook extends the previous style classification notebook by testing whether the results remain stable across **five different data splits**.

Its role is to provide a **robustness study**, not just a single-split performance report.

## Dataset Used

* The same style classification data as the previous notebook
* Five train/validation/test split configurations

## Input

* Multi-split dataset CSV files
* Image directories
* Same model families and fine-tuning settings as the style classification notebook

## Output

* Per-split training and evaluation results
* Mean ± std summary tables across splits
* More reliable model comparison

## What the code is doing

### Part A. Baseline robustness

The notebook first repeats baseline models across five splits and aggregates the results.

### Part B. Fine-tuning robustness

It then repeats the fine-tuning experiments across all splits.

### Part C. Mean ± std reporting

The code summarizes each experiment with mean ± std, which is very useful for thesis tables.

## Small Summary from the Results

This notebook strengthens the validity of your style classification conclusions.
The most important takeaway is that the selected best method remains competitive across multiple splits, which means the chosen style model is not just a lucky result from a single partition.

In the thesis, this notebook can be presented as evidence that your style classification design is **robust and reproducible**.

---

# 04_recommendation.ipynb

## Purpose

This notebook implements the final **attribute-aware recommendation and retrieval stage**.
It is the notebook that actually turns predicted semantics into retrieval results.

This notebook compares four retrieval settings:

1. image-only retrieval
2. text-only retrieval
3. fusion retrieval
4. native multimodal retrieval

So this is the notebook that demonstrates the final value of the full framework.

## Dataset Used

According to your slide and notebook structure, the retrieval setup uses:

* **Query set**: Pinterest + Fashion Product Images
* **Gallery set**: Polyvore train split
* **Five seed-based galleries** for robustness evaluation

The final semantic fields used are:

* category
* sub_category
* color
* material
* pattern
* style 

## Input

Depending on retrieval setting, the input can be:

* query image only
* structured attribute text only
* image + text together
* fused image and text similarity scores

## Output

* Top-K retrieved items
* Retrieval CSV files
* Global ranking tables
* Attribute matching tables
* Semantic similarity tables
* Thesis-format summary tables

## What the code is doing

### Part A. Image-only retrieval

This part compares image encoders such as:

* SigLIP2
* FashionCLIP
* OpenFashionCLIP

The code encodes query and gallery images, computes cosine similarity, and retrieves the top-K nearest items.

### Part B. Text-only retrieval

This part uses the six predicted attributes as text and compares:

* BGE-M3
* Voyage

It tests both:

* structured text format
* full sentence template format

This is a very important design choice in your thesis because it directly examines whether **compact structured semantics** are already enough for high-quality retrieval.

### Part C. Fusion retrieval

This section combines:

* FashionCLIP image similarity
* Voyage text similarity

Two fusion methods are tested:

* score fusion
* rank fusion

This helps study whether late fusion gives better balanced retrieval quality.

### Part D. Native multimodal retrieval

This section evaluates models that natively encode image-text information together, such as:

* Voyage Multimodal
* SigLIP

### Part E. Global evaluation

The notebook computes multiple retrieval metrics, including:

* precision@k
* recall@k
* MRR@k
* MAP@k
* semantic similarity
* AMR@k
* EAM@k
* weighted attribute matching
* nDCG@k

These metrics are aligned with your thesis goal of evaluating not only visual similarity, but also **semantic faithfulness and attribute consistency**.

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
