# Data Science for Genomics and Precision Medicine

This repository serves as a collection of data science projects exploring complex challenges in bioinformatics, genomics, and computational biology. The notebooks within this collection apply advanced statistical modeling and machine learning techniques to real-world biological datasets, with a focus on uncovering insights relevant to drug discovery, cancer research, and precision medicine.

## Table of Contents

* [About This Repository](#about-this-repository)
* [Projects](#projects)
    * [1. Exploratory Data Analysis (EDA) of Gene Expression Data](#exploratory-data-analysis-(eda)-of-gene-expression-data)
    * [2. Prognostic Gene Signature in Breast Cancer](#project-1-prognostic-gene-signature-in-breast-cancer)
* [Setup & Installation](#setup--installation)


---

## About This Repository

The goal of this repository is to demonstrate the application of data science methodologies to biological data. Each project is contained within a self-explanatory Jupyter Notebook and tackles a specific question, from fundamental survival analysis to high-dimensional feature selection and model interpretation.

This repository is designed for data scientists, bioinformaticians, and researchers interested in the intersection of data science and genomics. It provides a practical framework for understanding how to apply statistical methods and machine learning techniques to biological datasets, particularly in the context of cancer research.

**Core topics explored include:**
* Survival Analysis (Kaplan-Meier, Cox Proportional-Hazards)
* High-Dimensional Data Analysis (p >> n problems)
* Unsupervised Learning (UMAP)
* Differential Gene Expression Analysis
* Feature Selection Techniques (Variance Filtering)
* Machine Learning for Genomics (LASSO)
* Feature Engineering with Clinical and Genomic Data
* Model Interpretation and Validation

---

## Projects

### 1. Exploratory Data Analysis (EDA) of Gene Expression Data

* **Notebook:** `notebooks/brca_gene_expression_eda.ipynb`

* **Description:** The objective of this notebook is to perform a comprehensive exploratory data analysis on the TCGA-BRCA RNA-sequencing dataset and define which genes are candidates for predictive modeling based on differential expression in tumor tissue vs. normal tissue.

* **Key Findings:**

  * **Gene Expression Data Reflects Known Biology**: Housekeeping genes included mitochondrial and riboxomal genes, validating the biological plausibility of the data.

  * **The Cancer Signal is a Dominant Source of Variation.**: A primary driver of variation in the dataset is the cancer phenotype itself, validated by UMAP plot of the high-variance genes between tumor samples and normal tissue samples.

  * **Specific Genes are Massively Dysregulated in Tumors**: A differential expression analysis pinpointed 73 statistically significant genes that are either up- (potential oncogenes) or down-regulated (potential tumor supressors) in cancer cells.

---

### 2. Prognostic Gene Signature in Breast Cancer

* **Notebook:** `notebooks/brca_gene_expression_survival.ipynb`

* **Description:** This project performs an end-to-end survival analysis on The Cancer Genome Atlas (TCGA) Breast Cancer (BRCA) cohort to identify a gene signature predictive of patient outcomes.

* **Key Findings:**

  * **Multivariate Survival Modeling**: A Cox Proportional-Hazards model is built using clinical and biological variables (Age, Race, and ESR1 gene expression) to assess their individual impact on survival.

  * **High-Dimensional Feature Selection**: To navigate the challenge of analyzing ~60,000 genes, feature selection is employed. This involves an initial unsupervised variance filter followed by a LASSO (L1) penalized Cox regression to select a sparse set of the most impactful genes.

  * **Model Validation and Interpretation**: The final set of selected genes is validated in a standard Cox model to derive unbiased hazard ratios and p-values, and the well-known ESR1 gene's exclusion by the LASSO model is investigated.

---

## Installation & Dependencies

To ensure reproducibility, all required packages are specified in the root directory.

Using `conda`:
For a complete, isolated environment that mirrors the development setup, use the `environment.yml` file:
```
conda env create -f environment.yml
```

Using `pip`:
Alternatively, you can install all necessary packages into your current environment using pip:
```
pip install -r requirements.txt
```
---
