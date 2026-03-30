# MCC Directory Structure

This directory contains the codebase for the User-Adaptive XAI (Explainable AI) Pipeline. It is structured into distinct stages of data processing, explanation generation, and evaluation, separated into interactive Jupyter notebooks and reusable Python modules.

## 📂 Directory Layout
- **`outputs/`**: This directory stores all intermediate checkpoint files generated at each stage of the pipeline (e.g., `lime_results.json`, `ontology_results.json`, `explanations.json`, `results.csv`).
- **`__pycache__/`**: Standard Python cache directory for compiled bytecode.

## 🐍 Python Modules (Helpers & Configuration)
- **`config.py`**: The central configuration file. It contains paths to models and directories, hyperparameter settings (LIME samples, LLM thresholds), classification label maps, and experiment parameters (like `USER_CATEGORY` and `ABLATION_MODE`).
- **`model_loaders.py`**: Responsible for loading the different AI models used in the pipeline. It handles loading the sequence classification model, the biomedical NER (Named Entity Recognition) pipeline, the instruction-tuned LLM (Qwen), and the OWL ontology object.
- **`ontology_helpers.py`**: Contains all logic for interacting with the biomedical ontology (`doid.owl`). It provides functions to search for concepts, extract ancestor hierarchies, and calculate adaptive suitability scores based on the `USER_CATEGORY` (e.g., matching general concepts for beginners and deep/specific concepts for experts).
- **`pipeline_helpers.py`**: A comprehensive utility module for stateless functions used across all notebook stages. It is grouped into sections like Entity merging, LIME predictor setup, Prompt building, LLM generation, Readability/Faithfulness metric calculation, and Checkpoint I/O.

## 📓 Jupyter Notebooks (Interactive Pipeline Stages)
- **`01_lime.ipynb`**: **Stage 1 (Feature Extraction)**. Extracts local feature attributions using LIME (Local Interpretable Model-Agnostic Explanations). It identifies the key tokens that were influential in the underlying text classification model's prediction.
- **`02_ontology.ipynb`**: **Stage 2 (Ontology Enrichment)**. Maps the influential LIME features to an ontology graph. It selects the most appropriate biomedical ancestors for those features using the adaptive scoring mechanisms.
- **`03_llm.ipynb`**: **Stage 3 (Explanation Generation)**. Takes the model's prediction, the top LIME features, and their ontology ancestor paths to construct a tailored instruction prompt. It then prompts the LLM to generate a natural language explanation adapted to the user's expertise level.
- **`04_analysis.ipynb`**: **Stage 4 (Evaluation & Metrics)**. Calculates objective metrics on the generated explanations, evaluating readability (Flesch Reading Ease, Flesch-Kincaid) and faithfulness (LIME coverage, ontology hit rate) to assess the pipeline's effectiveness.
