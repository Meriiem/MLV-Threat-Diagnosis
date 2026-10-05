# Formally Guided Active Learning for Threat Diagnosis
This repository contains the supplementary data and code for the paper: 
**"Formally Guided Active Learning for Threat Diagnosis Applied to a Multi-Component Safety-Critical System"** 
*(M. Aoudia, K. El-Fakih, I. Zualkernan, D. Dghaym)*.

It includes the Event-B formal models, mutation-generated execution traces, and the Python machine learning pipeline used to evaluate active learning strategies on a Mechanical Lung Ventilator (MLV).

## Repository Structure

* **`EventB_Models/`** — The formally verified Event-B models of the MLV components.
* **`Datasets_Traces/`**
  * `inputs/` — Mutant catalogs, test suites, and raw ProB execution traces.
  * `features/` — Extracted sparse feature matrices and sample metadata.
  * `manifests/` — Candidate pools, test set splits, and experiment configurations.
  * `results/` — Output logs, curve-fitting parameters, and hyperparameter tuning data.
* **`ML_Code/`**
  * `active_learning.py` — Main pipeline for trace execution, candidate selection, and classifier training.
  * `pipeline/` — Modules for data preparation, feature building, and ProB interfacing.
  * `analysis/` — Scripts for curve fitting and performance evaluation.
* **`Figures/`** — Generated plots comparing classifiers and active learning query strategies.
* **`Supplementary_PDF/`** — Extended documentation and detailed tables (`Supplementary_Material.pdf`).
* **`requirements.txt`** — Python dependencies required to run the pipeline.
