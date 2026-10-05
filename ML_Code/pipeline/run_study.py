from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder


def structural_features(row):
    result = {"fault_count": float(row.get("number_of_threats", row.get("fault_count", 0)) or 0)}
    for field in ["affected_components", "threat_types", "edit_targets", "source_single_mutants", "source_mutants"]:
        value = str(row.get(field, ""))
        for token in value.replace("+", " ").replace("/", " ").replace("::", " ").split():
            token = token.strip().lower()
            if token:
                result[f"token::{token}"] = 1.0
    return result


def task_label(row, task):
    if task == "Task A: Component":
        return str(row.get("affected_components", "UNKNOWN"))
    if task == "Task B: Threat Type":
        return str(row.get("threat_types", "UNKNOWN"))
    return f'{row.get("affected_components", "UNKNOWN")} || {row.get("threat_types", "UNKNOWN")}'


def token_set(row):
    return {key for key, value in structural_features(row).items() if key.startswith("token::") and value}


def train_selector(labeled_rows, task, seed):
    labels = [task_label(row, task) for row in labeled_rows]
    if len(set(labels)) < 2:
        return None
    vectorizer = DictVectorizer(sparse=True)
    matrix = vectorizer.fit_transform([structural_features(row) for row in labeled_rows])
    encoder = LabelEncoder()
    target = encoder.fit_transform(labels)
    model = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
    model.fit(matrix, target)
    return {"vectorizer": vectorizer, "model": model}


def choose_candidates(strategy, pool_rows, labeled_rows, count, task, seed):
    if count <= 0:
        return []
    count = min(count, len(pool_rows))
    rng = np.random.default_rng(seed)
    if strategy == "Random":
        indices = rng.choice(len(pool_rows), size=count, replace=False)
        return [pool_rows[int(index)] for index in indices]
    if strategy in {"Entropy", "Margin"}:
        selector = train_selector(labeled_rows, task, seed)
        if selector is None:
            indices = rng.choice(len(pool_rows), size=count, replace=False)
            return [pool_rows[int(index)] for index in indices]
        matrix = selector["vectorizer"].transform([structural_features(row) for row in pool_rows])
        probabilities = selector["model"].predict_proba(matrix)
        if strategy == "Entropy":
            scores = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum(axis=1)
            order = np.argsort(-scores)
        else:
            sorted_probabilities = np.sort(probabilities, axis=1)
            margins = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
            order = np.argsort(margins)
        return [pool_rows[int(index)] for index in order[:count]]
    remaining = list(pool_rows)
    chosen = []
    reference = [token_set(row) for row in labeled_rows]
    if not reference and remaining:
        first = int(rng.integers(0, len(remaining)))
        chosen.append(remaining.pop(first))
        reference.append(token_set(chosen[-1]))
    while remaining and len(chosen) < count:
        distances = []
        for row in remaining:
            current = token_set(row)
            distance = min(1.0 - len(current & item) / len(current | item) if current | item else 0.0 for item in reference)
            distances.append(distance)
        index = int(np.argmax(distances))
        chosen.append(remaining.pop(index))
        reference.append(token_set(chosen[-1]))
    return chosen


def make_model(name, seed):
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, class_weight="balanced", random_state=seed, n_jobs=-1, verbosity=-1)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, objective="multi:softprob", eval_metric="mlogloss", random_state=seed, n_jobs=-1)
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=5, random_state=seed)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", random_state=seed, n_jobs=-1)
    return LogisticRegression(max_iter=4000, class_weight="balanced", random_state=seed, n_jobs=-1)


def dense_if_needed(name, matrix):
    return matrix.toarray() if name == "Gradient Boosting" else matrix


def internal_validation(name, matrix, labels, groups, seed):
    if len(set(labels)) < 2:
        return math.nan, math.nan
    group_counts = pd.DataFrame({"label": labels, "group": groups}).drop_duplicates().groupby("label")["group"].nunique()
    splits = min(5, int(group_counts.min())) if not group_counts.empty else 0
    if splits < 2:
        return math.nan, math.nan
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    splitter = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=seed)
    scores = []
    for train_index, validation_index in splitter.split(np.zeros(len(labels)), encoded, groups):
        model = make_model(name, seed)
        x_train = dense_if_needed(name, matrix[train_index])
        x_validation = dense_if_needed(name, matrix[validation_index])
        model.fit(x_train, encoded[train_index])
        prediction = encoder.inverse_transform(model.predict(x_validation).astype(int))
        scores.append(f1_score(labels[validation_index], prediction, average="macro", zero_division=0) * 100)
    return float(np.mean(scores)), float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0


def fit_and_predict(name, x_train, y_train_text, x_test, seed):
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_text)
    model = make_model(name, seed)
    started = time.perf_counter()
    model.fit(dense_if_needed(name, x_train), y_train)
    elapsed = time.perf_counter() - started
    prediction = encoder.inverse_transform(model.predict(dense_if_needed(name, x_test)).astype(int))
    return prediction, elapsed


def selector_description(strategy, config):
    return config["selector_model"][strategy]


def stable_row(group, tolerance):
    ordered = group.sort_values("added_compositions").reset_index(drop=True)
    for index in range(len(ordered)):
        current = float(ordered.loc[index, "internal_validation_f1_mean"])
        later = ordered.loc[index:, "internal_validation_f1_mean"].astype(float)
        if float(later.max()) - current <= tolerance:
            return ordered.loc[index]
    return ordered.iloc[-1]


def choose_setup(internal_frame, tolerance, practical_tie):
    stable = []
    for (_, _), group in internal_frame.groupby(["diagnostic_classifier", "selection_strategy"], sort=False):
        stable.append(stable_row(group, tolerance))
    stable_frame = pd.DataFrame(stable)
    best_score = float(stable_frame["internal_validation_f1_mean"].max())
    eligible = stable_frame[stable_frame["internal_validation_f1_mean"] >= best_score - practical_tie]
    return eligible.sort_values(["new_prob_runs", "internal_validation_f1_mean"], ascending=[True, False]).iloc[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--candidate_combinations", required=True)
    parser.add_argument("--initial_sets", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--initial_set", default="all")
    parser.add_argument("--task", default="all")
    parser.add_argument("--repeats", type=int, default=0)
    parser.add_argument("--all_classifiers", action="store_true")
    args = parser.parse_args()

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    repeats = args.repeats or int(config["repeats"])
    matrix = sparse.load_npz(args.features).tocsr()
    metadata = pd.read_csv(args.metadata)
    candidates = pd.read_csv(args.candidate_combinations)
    initial_catalog = pd.read_csv(args.initial_sets)
    metadata_ids = set(metadata["candidate_id"].astype(str))
    hidden_rows = candidates[candidates["candidate_set"].eq("Hidden Test Set")].to_dict("records")
    available_rows = candidates[candidates["candidate_set"].eq("Available Candidate Set")].to_dict("records")
    hidden_ids = [str(row["candidate_id"]) for row in hidden_rows]
    requests = [{"candidate_id": item, "purpose": "Hidden Test Set"} for item in sorted(set(hidden_ids) - metadata_ids)]
    initial_names = config["initial_sets"] if args.initial_set == "all" else [args.initial_set]
    task_names = list(config["tasks"]) if args.task == "all" else [args.task]
    strategies = config["selection_strategies"]
    models = config["diagnostic_classifiers"] if args.all_classifiers else [config["primary_diagnostic_classifier"]]
    percentages = config["candidate_checkpoints_percent"]
    target_counts = [(f"{percent}%", int(math.ceil(len(available_rows) * percent / 100))) for percent in percentages]
    internal_rows = []
    selection_rows = []
    chosen_rows = []
    hidden_result_rows = []
    selected_map = {}

    for repeat in range(1, repeats + 1):
        for initial_name in initial_names:
            initial_rows = initial_catalog[initial_catalog["initial_set"].eq(initial_name)].to_dict("records")
            initial_ids = [str(row["system_id"]) for row in initial_rows]
            initial_candidate_ids = {str(row["candidate_id"]) for row in initial_rows if str(row.get("candidate_id", "")).strip() and str(row.get("candidate_id", "")).lower() != "nan"}
            for task in task_names:
                label_column = config["tasks"][task]
                for strategy in strategies:
                    selected = []
                    labeled_structural = [dict(row, source_mutants=row.get("system_id", "")) for row in initial_rows]
                    pool = [row for row in available_rows if str(row["candidate_id"]) not in initial_candidate_ids]
                    selected_map[(repeat, initial_name, task, strategy, "Initial set only")] = []
                    for checkpoint, target in [("Initial set only", 0), *target_counts]:
                        needed = target - len(selected)
                        if needed > 0:
                            batch = choose_candidates(strategy, pool, labeled_structural, needed, task, int(config["random_seed"]) + repeat * 1000 + target)
                            selected.extend(batch)
                            batch_ids = {str(row["candidate_id"]) for row in batch}
                            pool = [row for row in pool if str(row["candidate_id"]) not in batch_ids]
                            labeled_structural.extend(batch)
                            for rank, row in enumerate(batch, start=len(selected) - len(batch) + 1):
                                selection_rows.append({
                                    "repeat": repeat,
                                    "initial_set": initial_name,
                                    "task": task,
                                    "selection_strategy": strategy,
                                    "selector_model_or_rule": selector_description(strategy, config),
                                    "checkpoint": checkpoint,
                                    "selection_order": rank,
                                    "candidate_id": row["candidate_id"],
                                    "fault_count": row.get("number_of_threats", row.get("fault_count", "")),
                                    "affected_components": row["affected_components"],
                                    "threat_types": row["threat_types"],
                                })
                        selected_ids = [str(row["candidate_id"]) for row in selected]
                        selected_map[(repeat, initial_name, task, strategy, checkpoint)] = selected_ids.copy()
                        missing = sorted(set(selected_ids) - metadata_ids)
                        requests.extend({"candidate_id": item, "purpose": f"{initial_name} | {task} | {strategy} | {checkpoint}"} for item in missing)
                        if missing:
                            continue
                        training_ids = set(initial_ids + selected_ids)
                        training_mask = metadata["candidate_id"].astype(str).isin(training_ids).to_numpy()
                        x_train = matrix[training_mask]
                        y_train = metadata.loc[training_mask, label_column].astype(str).to_numpy()
                        groups = metadata.loc[training_mask, "candidate_id"].astype(str).to_numpy()
                        for model in models:
                            mean, std = internal_validation(model, x_train, y_train, groups, int(config["random_seed"]) + repeat)
                            internal_rows.append({
                                "result_status": "MEASURED INTERNAL VALIDATION",
                                "repeat": repeat,
                                "initial_set": initial_name,
                                "task": task,
                                "diagnostic_classifier": model,
                                "selection_strategy": "Initial set only" if checkpoint == "Initial set only" else strategy,
                                "selector_model_or_rule": "No selection yet" if checkpoint == "Initial set only" else selector_description(strategy, config),
                                "checkpoint": checkpoint,
                                "added_compositions": target,
                                "new_prob_runs": target * 30,
                                "internal_validation_f1_mean": mean,
                                "internal_validation_f1_std": std,
                            })

    pd.DataFrame(internal_rows).to_csv(output / "internal_comparison.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(output / "selection_history.csv", index=False)
    request_frame = pd.DataFrame(requests).drop_duplicates() if requests else pd.DataFrame(columns=["candidate_id", "purpose"])
    request_frame.to_csv(output / "execution_requests.csv", index=False)

    if not request_frame.empty:
        (output / "run_status.json").write_text(json.dumps({"status": "WAITING_FOR_REQUIRED_PROB_TRACES", "requested_candidates": int(request_frame["candidate_id"].nunique())}, indent=2), encoding="utf-8")
        pd.DataFrame(chosen_rows).to_csv(output / "chosen_setups.csv", index=False)
        pd.DataFrame(hidden_result_rows).to_csv(output / "final_hidden_results.csv", index=False)
        return

    internal_frame = pd.DataFrame(internal_rows)
    hidden_mask = metadata["candidate_id"].astype(str).isin(hidden_ids).to_numpy()
    x_hidden = matrix[hidden_mask]

    for repeat in range(1, repeats + 1):
        for initial_name in initial_names:
            initial_rows = initial_catalog[initial_catalog["initial_set"].eq(initial_name)].to_dict("records")
            initial_ids = [str(row["system_id"]) for row in initial_rows]
            for task in task_names:
                label_column = config["tasks"][task]
                subset = internal_frame[(internal_frame["repeat"] == repeat) & (internal_frame["initial_set"] == initial_name) & (internal_frame["task"] == task) & (internal_frame["selection_strategy"] != "Initial set only")].copy()
                chosen = choose_setup(subset, float(config["stabilization_tolerance_percentage_points"]), 0.25)
                strategy = str(chosen["selection_strategy"])
                checkpoint = str(chosen["checkpoint"])
                selected_ids = selected_map[(repeat, initial_name, task, strategy, checkpoint)]
                training_ids = set(initial_ids + selected_ids)
                training_mask = metadata["candidate_id"].astype(str).isin(training_ids).to_numpy()
                x_train = matrix[training_mask]
                y_train = metadata.loc[training_mask, label_column].astype(str).to_numpy()
                y_hidden = metadata.loc[hidden_mask, label_column].astype(str).to_numpy()
                prediction, training_seconds = fit_and_predict(str(chosen["diagnostic_classifier"]), x_train, y_train, x_hidden, int(config["random_seed"]) + repeat)
                chosen_rows.append({
                    "result_status": "MEASURED SETUP CHOSEN WITHOUT HIDDEN TEST",
                    "repeat": repeat,
                    "initial_set": initial_name,
                    "task": task,
                    "diagnostic_classifier": chosen["diagnostic_classifier"],
                    "selection_strategy": strategy,
                    "selector_model_or_rule": chosen["selector_model_or_rule"],
                    "selected_checkpoint": checkpoint,
                    "added_compositions": int(chosen["added_compositions"]),
                    "new_prob_runs": int(chosen["new_prob_runs"]),
                    "internal_validation_f1_mean": float(chosen["internal_validation_f1_mean"]),
                    "selection_reason": "Strongest stabilized internal-validation result; lower ProB cost breaks practical ties.",
                })
                hidden_result_rows.append({
                    **chosen_rows[-1],
                    "result_status": "MEASURED FINAL HIDDEN TEST",
                    "hidden_test_compositions": len(hidden_ids),
                    "hidden_test_f1_macro": f1_score(y_hidden, prediction, average="macro", zero_division=0) * 100,
                    "hidden_test_accuracy": accuracy_score(y_hidden, prediction) * 100,
                    "hidden_test_precision_macro": precision_score(y_hidden, prediction, average="macro", zero_division=0) * 100,
                    "hidden_test_recall_macro": recall_score(y_hidden, prediction, average="macro", zero_division=0) * 100,
                    "generalization_gap": float(chosen["internal_validation_f1_mean"]) - f1_score(y_hidden, prediction, average="macro", zero_division=0) * 100,
                    "model_training_seconds": training_seconds,
                })

    pd.DataFrame(chosen_rows).to_csv(output / "chosen_setups.csv", index=False)
    pd.DataFrame(hidden_result_rows).to_csv(output / "final_hidden_results.csv", index=False)
    (output / "run_status.json").write_text(json.dumps({"status": "COMPLETE", "hidden_test_opened_only_for_chosen_setups": True}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
