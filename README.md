# Data Science for Genomics and Precision Medicine

This repository serves as a collection of data science projects exploring complex challenges in bioinformatics, genomics, and computational biology. The notebooks within this collection apply advanced statistical modeling and machine learning techniques to real-world biological datasets, with a focus on uncovering insights relevant to drug discovery, cancer research, and precision medicine.

## Table of Contents

* [About This Repository](#about-this-repository)
* [Projects](#projects)
    * [1. Prognostic Gene Signature in Breast Cancer](#project-1-prognostic-gene-signature-in-breast-cancer)
* [Setup & Installation](#setup--installation)
* [Contact](#contact)

---

## About This Repository

The goal of this repository is to demonstrate the application of data science methodologies to biological data. Each project is contained within a self-explanatory Jupyter Notebook and tackles a specific question, from fundamental survival analysis to high-dimensional feature selection and model interpretation.

**Core topics explored include:**
* Survival Analysis (Kaplan-Meier, Cox Proportional-Hazards)
* High-Dimensional Data Analysis (p >> n problems)
* Machine Learning for Genomics (LASSO)
* Feature Engineering with Clinical and Genomic Data
* Model Interpretation and Validation

---

## Projects

### 1. Prognostic Gene Signature in Breast Cancer

* **Notebook:** `notebooks/brca_gene_expression_survival.ipynb`
* **Description:** This project performs an end-to-end survival analysis on The Cancer Genome Atlas (TCGA) Breast Cancer (BRCA) cohort to identify a gene signature predictive of patient outcomes.

#### **Key Findings:**

This analysis successfully developed a pipeline to identify prognostic biomarkers from high-dimensional genomic data. 

* **Exploratory Survival Analysis**: A baseline survival curve for the entire cohort is established using the Kaplan-Meier estimator.

* **Multivariate Survival Modeling**: A Cox Proportional-Hazards model is built using clinical and biological variables (Age, Race, and ESR1 gene expression) to assess their individual impact on survival.

* **High-Dimensional Feature Selection**: To navigate the challenge of analyzing ~60,000 genes, feature selection is employed. This involves an initial unsupervised variance filter followed by a LASSO (L1) penalized Cox regression to select a sparse set of the most impactful genes.

* **Model Validation and Interpretation**: The final set of selected genes is validated in a standard Cox model to derive unbiased hazard ratios and p-values, and the well-known ESR1 gene's exclusion by the LASSO model is investigated.

The ultimate goal is to produce an interpretable model that uncovers a potential gene signature associated with patient survival in breast cancer.

---

## Setup & Installation

The packages required to run the notebooks in the repository can be found in the `requirements.txt` file in the root directory.

---

## Contact

For any questions or collaborations, please feel free to reach out.

**Daniel Trainer**
[LinkedIn](www.linkedin.com/in/daniel-trainer-800917181)
