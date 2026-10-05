# MLV Threat Diagnosis: Supplementary Materials

Supplementary materials for the paper "Formally Guided Active Learning for Threat Diagnosis Applied to a Multi-Component Safety-Critical System" (M. Aoudia, K. El-Fakih, I. Zualkernan, D. Dghaym).

The repository contains the mutant catalog, the test suite, ProB-derived trace data, feature matrices, active-learning code, experiment results, and figures for the three diagnosis tasks (A: component, B: threat type, C: joint) on the mechanical lung ventilator (MLV).

## Directory structure

    EventB_Models/            Event-B models.
    Datasets_Traces/
        inputs/               Mutant catalog, test suite, expanded ProB state traces
        features/             Sparse feature matrices and per-sample metadata
        manifests/            Candidate pool, hidden test set, initial sets, experiment lists
        results/
            experiment_results/   Per-checkpoint results, selection logs, cost summary, repeated runs
            curve_fits/           Fitted saturating-exponential parameters and summary tables
            rf_tuning/            Random Forest hyperparameter experiments (xlsx and csv)
    ML_Code/
        active_learning.py        Active-learning pipeline (trace execution, selection, training)
        rf_grid_active_learning.py  Random Forest grid experiments
        pipeline/                 Candidate preparation, ProB execution, feature building, study runners
        analysis/                 Curve fitting and figure generation
    Figures/
        comparisons/          Classifier and strategy comparison plots (main paper figures)
        classifier_fits/      Fitted curves per classifier, task, initial set
        strategy_fits/        Fitted curves per query strategy, task, initial set
        summaries/            Plateau, rate k, and x95 summaries
        rf_tuning/            Random Forest tuning figures
    Supplementary_PDFs/       Supplementary_Material.pdf
    requirements.txt
