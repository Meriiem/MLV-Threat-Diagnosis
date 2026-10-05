import argparse
import json
import math
import random
import re
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

STATE_CATEGORICAL = ["alarm_state", "alarm_type", "controller_state", "gui_state", "is_alarm_displayed", "is_gui_frozen", "is_ventilating", "vent_mode"]
STATE_NUMERIC = ["battery_level", "ie_ratio", "insp_pressure", "pressure_reading", "resp_rate", "temperature_reading", "voltage_level"]
warnings.filterwarnings("ignore")

STRATEGIES = ["random", "uncertainty_entropy", "uncertainty_margin", "diversity_jaccard", "formal_hybrid"]


def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def split_plus(value):
    return [x.strip() for x in clean_text(value).split(" + ") if x.strip()]


def component_from_target(target):
    text = clean_text(target).upper()
    event = text.split("/")[0]
    if "GUI" in event or "DISPLAY" in event or "NAVIGATE" in event or "FREEZE" in event or "UNFREEZE" in event:
        return "GUI"
    if "ALARM" in event or "BATTERY" in event or "VOLTAGE" in event or "PRESSURE" in event or "TEMPERATURE" in event or "ACKNOWLEDGE" in event or "ESCALATE" in event:
        return "ALRM"
    return "CTRL"


def edit_signature(target, original, mutated):
    return "||".join([clean_text(target), clean_text(original), clean_text(mutated)])


def build_mutant_metadata(mutants):
    singles = mutants[mutants["Fault_Count"] == 1]
    signature_map = {}
    for _, row in singles.iterrows():
        for index in range(1, 4):
            target = row.get(f"Edit{index}_Target")
            original = row.get(f"Edit{index}_Original")
            mutated = row.get(f"Edit{index}_Mutated")
            if clean_text(target):
                signature_map[edit_signature(target, original, mutated)] = clean_text(row["Threat_Family"])
    records = []
    for _, row in mutants.iterrows():
        components = set()
        threats = set()
        pairs = set()
        tokens = set()
        edit_targets = []
        for index in range(1, 4):
            target = row.get(f"Edit{index}_Target")
            original = row.get(f"Edit{index}_Original")
            mutated = row.get(f"Edit{index}_Mutated")
            if not clean_text(target):
                continue
            component = component_from_target(target)
            threat = signature_map.get(edit_signature(target, original, mutated), "")
            components.add(component)
            if threat:
                threats.add(threat)
                pairs.add(f"{component}::{threat}")
            edit_targets.append(clean_text(target))
            tokens.update(x for x in re.split(r"[^A-Za-z0-9_]+", clean_text(target)) if x)
            tokens.update(x for x in re.split(r"[^A-Za-z0-9_]+", clean_text(original)) if x)
            tokens.update(x for x in re.split(r"[^A-Za-z0-9_]+", clean_text(mutated)) if x)
        if not threats:
            threats.update(split_plus(row["Threat_Family"]))
        if not components:
            components.update(x for x in clean_text(row["Fault_Components"]).split("+") if x)
        if not pairs:
            for component in components:
                for threat in threats:
                    pairs.add(f"{component}::{threat}")
        tokens.update(components)
        tokens.update(threats)
        tokens.add(f"fault_count_{int(row['Fault_Count'])}")
        records.append({
            "mutant_id": clean_text(row["ID"]),
            "fault_count": int(row["Fault_Count"]),
            "fault_order": {1: "single", 2: "double", 3: "triple"}[int(row["Fault_Count"])],
            "components": tuple(sorted(components)),
            "threats": tuple(sorted(threats)),
            "pairs": tuple(sorted(pairs)),
            "tokens": frozenset(tokens),
            "edit_targets": " | ".join(edit_targets),
            "threat_family": clean_text(row["Threat_Family"]),
            "fault_components": clean_text(row["Fault_Components"]),
        })
    return pd.DataFrame(records).set_index("mutant_id", drop=False)


def last_runtime_ms(text):
    values = re.findall(r"walltime:\s*(\d+)\s*ms", clean_text(text), flags=re.IGNORECASE)
    if not values:
        values = re.findall(r"Walltime:\s*(\d+)\s*ms", clean_text(text))
    return float(values[-1]) if values else 0.0


def trace_features(group):
    ordered = group.sort_values("step")
    features = {}
    for event in ordered["event"].dropna().astype(str):
        event = event.strip()
        if event and event.lower() != "nan":
            features[f"EV:{event}"] = 1.0
            features[f"EV_COUNT:{event}"] = features.get(f"EV_COUNT:{event}", 0.0) + 1.0
    for column in STATE_CATEGORICAL:
        if column in ordered.columns:
            for value in ordered[column].dropna().astype(str):
                value = value.strip()
                if value and value.lower() != "nan":
                    features[f"{column}={value}"] = 1.0
    for column in STATE_NUMERIC:
        if column in ordered.columns:
            values = pd.to_numeric(ordered[column], errors="coerce").dropna()
            if len(values):
                features[column] = float(values.iloc[-1])
    features["RUN_OK"] = float(clean_text(ordered.iloc[0].get("ok")) in {"1", "True", "true"})
    features["PARSE_OK"] = float(clean_text(ordered.iloc[0].get("parse_ok")) in {"1", "True", "true"})
    return features


def aggregate_traces(states):
    grouped_features = {}
    grouped_first = {}
    for (mutant_id, tc_id), group in states.groupby(["mutant_id", "tc_id"], sort=False):
        grouped_features[(str(mutant_id), str(tc_id))] = trace_features(group)
        grouped_first[(str(mutant_id), str(tc_id))] = group.iloc[0]
    base_by_test = {tc_id: features for (mutant_id, tc_id), features in grouped_features.items() if mutant_id.upper() == "BASE"}
    samples = []
    costs = []
    for (mutant_id, tc_id), absolute in grouped_features.items():
        if mutant_id.upper() == "BASE":
            continue
        reference = base_by_test.get(tc_id, {})
        combined = dict(absolute)
        for key in set(absolute) | set(reference):
            difference = float(absolute.get(key, 0.0)) - float(reference.get(key, 0.0))
            if difference != 0.0:
                combined[f"DELTA:{key}"] = difference
        first = grouped_first[(mutant_id, tc_id)]
        trace_ms = last_runtime_ms(first.get("stdout_trace"))
        replay_ms = last_runtime_ms(first.get("stdout_replay"))
        if trace_ms + replay_ms == 0 and clean_text(first.get("timed_out_trace")) == "1":
            trace_ms = float(first.get("timeout_used_ms", 0) or 0)
        if trace_ms + replay_ms == 0 and clean_text(first.get("timed_out_replay")) == "1":
            replay_ms = float(first.get("timeout_used_ms", 0) or 0)
        samples.append({"mutant_id": mutant_id, "tc_id": tc_id, "features": combined})
        costs.append({"mutant_id": mutant_id, "tc_id": tc_id, "prob_seconds": (trace_ms + replay_ms) / 1000.0})
    return pd.DataFrame(samples), pd.DataFrame(costs)

def prepare_data(states_path, mutants_path):
    states_path = str(states_path)
    states = pd.read_csv(states_path, low_memory=False) if states_path.lower().endswith(".csv") else pd.read_excel(states_path, sheet_name=0)
    mutants = pd.read_excel(mutants_path, sheet_name="mutants")
    metadata = build_mutant_metadata(mutants)
    samples, costs = aggregate_traces(states)
    samples = samples[samples["mutant_id"].isin(metadata.index)].reset_index(drop=True)
    vectorizer = DictVectorizer(sparse=True)
    matrix = vectorizer.fit_transform(samples["features"].tolist())
    sample_mutants = samples["mutant_id"].to_numpy()
    cost_by_mutant = costs.groupby("mutant_id")["prob_seconds"].sum().to_dict()
    return metadata, samples, matrix, sample_mutants, vectorizer, cost_by_mutant


def task_labels(metadata, task):
    if task == "TaskA_Component":
        return metadata["components"].to_dict()
    if task == "TaskB_Threat":
        return metadata["threats"].to_dict()
    return metadata["pairs"].to_dict()


def make_estimator(seed):
    base = LGBMClassifier(n_estimators=15, learning_rate=0.05, num_leaves=31, max_depth=-1, subsample=0.9, colsample_bytree=0.9, random_state=seed, n_jobs=1, verbosity=-1, force_col_wise=True)
    return OneVsRestClassifier(base, n_jobs=1)


def fit_predict(matrix, sample_mutants, labels_by_mutant, train_mutants, test_mutants, seed, label_classes=None):
    train_mask = np.isin(sample_mutants, list(train_mutants))
    test_mask = np.isin(sample_mutants, list(test_mutants))
    if label_classes is None:
        label_classes = sorted({label for mutant in labels_by_mutant for label in labels_by_mutant[mutant]})
    mlb = MultiLabelBinarizer(classes=label_classes)
    mlb.fit([label_classes])
    train_labels = mlb.transform([labels_by_mutant[mutant] for mutant in sample_mutants[train_mask]])
    test_labels = mlb.transform([labels_by_mutant[mutant] for mutant in sample_mutants[test_mask]])
    started = time.perf_counter()
    estimator = make_estimator(seed)
    estimator.fit(matrix[train_mask], train_labels)
    train_seconds = time.perf_counter() - started
    probabilities = estimator.predict_proba(matrix[test_mask])
    predictions = (probabilities >= 0.5).astype(int)
    macro = f1_score(test_labels, predictions, average="macro", zero_division=0)
    micro = f1_score(test_labels, predictions, average="micro", zero_division=0)
    exact = accuracy_score(test_labels, predictions)
    return estimator, mlb, macro, micro, exact, train_seconds


def metadata_dict(record):
    values = {}
    values[f"fault_count={record['fault_count']}"] = 1.0
    for component in record["components"]:
        values[f"component={component}"] = 1.0
    for threat in record["threats"]:
        values[f"threat={threat}"] = 1.0
    for pair in record["pairs"]:
        values[f"pair={pair}"] = 1.0
    for token in record["tokens"]:
        values[f"token={token}"] = 1.0
    return values


def selector_probabilities(metadata, labels_by_mutant, selected, candidates, seed, label_classes):
    vectorizer = DictVectorizer(sparse=True)
    train_matrix = vectorizer.fit_transform([metadata_dict(metadata.loc[mutant]) for mutant in selected])
    candidate_matrix = vectorizer.transform([metadata_dict(metadata.loc[mutant]) for mutant in candidates])
    mlb = MultiLabelBinarizer(classes=label_classes)
    mlb.fit([label_classes])
    train_labels = mlb.transform([labels_by_mutant[mutant] for mutant in selected])
    estimator = make_estimator(seed)
    estimator.fit(train_matrix, train_labels)
    return estimator.predict_proba(candidate_matrix)


def jaccard_distance(left, right):
    union = left | right
    return 1.0 - len(left & right) / len(union) if union else 0.0


def choose_batch(strategy, metadata, labels_by_mutant, selected, candidates, batch_size, seed, label_classes):
    started = time.perf_counter()
    rng = random.Random(seed)
    candidates = list(candidates)
    if strategy == "random":
        rng.shuffle(candidates)
        chosen = candidates[:batch_size]
    elif strategy in {"uncertainty_entropy", "uncertainty_margin"}:
        probabilities = selector_probabilities(metadata, labels_by_mutant, selected, candidates, seed, label_classes)
        if strategy == "uncertainty_entropy":
            clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
            scores = np.mean(-(clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped)), axis=1)
        else:
            scores = np.mean(0.5 - np.abs(probabilities - 0.5), axis=1)
        order = np.argsort(-scores)
        chosen = [candidates[index] for index in order[:batch_size]]
    else:
        selected_tokens = [metadata.loc[mutant, "tokens"] for mutant in selected]
        label_counts = {label: 0 for label in label_classes}
        for mutant in selected:
            for label in labels_by_mutant[mutant]:
                label_counts[label] += 1
        remaining = candidates[:]
        chosen = []
        while remaining and len(chosen) < batch_size:
            best_mutant = None
            best_score = -1.0
            for mutant in remaining:
                tokens = metadata.loc[mutant, "tokens"]
                references = selected_tokens + [metadata.loc[item, "tokens"] for item in chosen]
                diversity = min(jaccard_distance(tokens, reference) for reference in references) if references else 1.0
                deficits = [1.0 / (1.0 + label_counts[label]) for label in labels_by_mutant[mutant]]
                coverage = float(np.mean(deficits)) if deficits else 0.0
                score = diversity if strategy == "diversity_jaccard" else 0.5 * diversity + 0.5 * coverage
                if score > best_score:
                    best_score = score
                    best_mutant = mutant
            chosen.append(best_mutant)
            remaining.remove(best_mutant)
            for label in labels_by_mutant[best_mutant]:
                label_counts[label] += 1
    return chosen, time.perf_counter() - started


def balanced_split(metadata, repeat_seed, holdout_fraction):
    rng = random.Random(repeat_seed)
    holdout = []
    development = []
    for fault_count in [1, 2, 3]:
        group = metadata[metadata["fault_count"] == fault_count]["mutant_id"].tolist()
        rng.shuffle(group)
        count = max(1, int(round(len(group) * holdout_fraction)))
        holdout.extend(group[:count])
        development.extend(group[count:])
    return development, holdout


def balanced_seed(metadata, development, seed_per_level, repeat_seed):
    rng = random.Random(repeat_seed)
    selected = []
    for fault_count in [1, 2, 3]:
        group = [mutant for mutant in development if int(metadata.loc[mutant, "fault_count"]) == fault_count]
        rng.shuffle(group)
        selected.extend(group[:min(seed_per_level, len(group))])
    return selected


def first_stable_point(frame, ceiling, window, tolerance, ceiling_gap):
    values = frame["external_f1_macro"].to_numpy()
    for index in range(window - 1, len(values)):
        recent = values[index - window + 1:index + 1]
        if np.max(recent) - np.min(recent) <= tolerance and values[index] >= ceiling - ceiling_gap:
            return int(frame.iloc[index]["iteration"])
    return int(frame.iloc[-1]["iteration"])


def run_experiments(metadata, matrix, sample_mutants, cost_by_mutant, output_dir, repeats, seed_sizes, batch_size, holdout_fraction, window, tolerance, ceiling_gap, strategies, max_selected):
    trajectories = []
    selections = []
    manifests = []
    ceilings = []
    tasks = ["TaskA_Component", "TaskB_Threat", "TaskC_ComponentThreat"]
    for repeat in range(repeats):
        split_seed = 1000 + repeat
        development, holdout = balanced_split(metadata, split_seed, holdout_fraction)
        for mutant in development:
            manifests.append({"repeat": repeat, "mutant_id": mutant, "partition": "development", "fault_order": metadata.loc[mutant, "fault_order"]})
        for mutant in holdout:
            manifests.append({"repeat": repeat, "mutant_id": mutant, "partition": "external_holdout", "fault_order": metadata.loc[mutant, "fault_order"]})
        validation_cost = sum(cost_by_mutant.get(mutant, 0.0) for mutant in holdout)
        full_development_cost = sum(cost_by_mutant.get(mutant, 0.0) for mutant in development)
        for task in tasks:
            labels_by_mutant = task_labels(metadata, task)
            label_classes = sorted({label for mutant in labels_by_mutant for label in labels_by_mutant[mutant]})
            _, _, ceiling_macro, ceiling_micro, ceiling_exact, ceiling_train = fit_predict(matrix, sample_mutants, labels_by_mutant, development, holdout, split_seed, label_classes)
            ceilings.append({"repeat": repeat, "task": task, "development_compositions": len(development), "external_compositions": len(holdout), "ceiling_f1_macro": ceiling_macro, "ceiling_f1_micro": ceiling_micro, "ceiling_exact_match": ceiling_exact, "training_seconds": ceiling_train, "development_prob_seconds": full_development_cost, "external_validation_prob_seconds": validation_cost})
            for seed_per_level in seed_sizes:
                initial = balanced_seed(metadata, development, seed_per_level, split_seed + seed_per_level * 10)
                for strategy in strategies:
                    selected = list(initial)
                    pool = [mutant for mutant in development if mutant not in selected]
                    cumulative_training = 0.0
                    cumulative_query = 0.0
                    iteration = 0
                    while True:
                        _, _, macro, micro, exact, train_seconds = fit_predict(matrix, sample_mutants, labels_by_mutant, selected, holdout, split_seed + iteration, label_classes)
                        cumulative_training += train_seconds
                        selected_cost = sum(cost_by_mutant.get(mutant, 0.0) for mutant in selected)
                        trajectories.append({"repeat": repeat, "task": task, "strategy": strategy, "seed_per_level": seed_per_level, "iteration": iteration, "selected_compositions": len(selected), "added_compositions": len(selected) - len(initial), "selected_trace_executions": len(selected) * 30, "external_f1_macro": macro, "external_f1_micro": micro, "external_exact_match": exact, "ceiling_f1_macro": ceiling_macro, "gap_to_ceiling": ceiling_macro - macro, "training_seconds_this_iteration": train_seconds, "cumulative_training_seconds": cumulative_training, "cumulative_query_seconds": cumulative_query, "selected_prob_seconds": selected_cost, "full_development_prob_seconds": full_development_cost, "external_validation_prob_seconds": validation_cost, "formal_cost_saved_fraction": 1.0 - selected_cost / full_development_cost if full_development_cost else 0.0})
                        if not pool or len(selected) >= max_selected:
                            break
                        chosen, query_seconds = choose_batch(strategy, metadata, labels_by_mutant, selected, pool, min(batch_size, len(pool)), split_seed + iteration * 101 + seed_per_level, label_classes)
                        cumulative_query += query_seconds
                        for mutant in chosen:
                            selections.append({"repeat": repeat, "task": task, "strategy": strategy, "seed_per_level": seed_per_level, "iteration_selected": iteration + 1, "mutant_id": mutant, "fault_order": metadata.loc[mutant, "fault_order"], "fault_components": metadata.loc[mutant, "fault_components"], "threat_family": metadata.loc[mutant, "threat_family"], "edit_targets": metadata.loc[mutant, "edit_targets"], "prob_seconds": cost_by_mutant.get(mutant, 0.0)})
                        selected.extend(chosen)
                        pool = [mutant for mutant in pool if mutant not in chosen]
                        iteration += 1
    trajectory_frame = pd.DataFrame(trajectories)
    ceiling_frame = pd.DataFrame(ceilings)
    stable_rows = []
    for keys, group in trajectory_frame.groupby(["repeat", "task", "strategy", "seed_per_level"], sort=False):
        ordered = group.sort_values("iteration").reset_index(drop=True)
        ceiling = float(ordered["ceiling_f1_macro"].iloc[0])
        stable_iteration = first_stable_point(ordered, ceiling, window, tolerance, ceiling_gap)
        row = ordered[ordered["iteration"] == stable_iteration].iloc[0].to_dict()
        row["stabilization_iteration"] = stable_iteration
        row["reached_within_ceiling_gap"] = bool(row["external_f1_macro"] >= ceiling - ceiling_gap)
        stable_rows.append(row)
    stable_frame = pd.DataFrame(stable_rows)
    summary = stable_frame.groupby(["task", "strategy", "seed_per_level"], as_index=False).agg(stable_compositions_mean=("selected_compositions", "mean"), stable_compositions_std=("selected_compositions", "std"), stable_f1_mean=("external_f1_macro", "mean"), stable_f1_std=("external_f1_macro", "std"), ceiling_f1_mean=("ceiling_f1_macro", "mean"), gap_to_ceiling_mean=("gap_to_ceiling", "mean"), cost_saved_mean=("formal_cost_saved_fraction", "mean"), selected_prob_seconds_mean=("selected_prob_seconds", "mean"), training_seconds_mean=("cumulative_training_seconds", "mean"), query_seconds_mean=("cumulative_query_seconds", "mean"), reached_rate=("reached_within_ceiling_gap", "mean"))
    trajectory_frame.to_csv(output_dir / "trajectory.csv", index=False)
    pd.DataFrame(selections).to_csv(output_dir / "selected_compositions.csv", index=False)
    pd.DataFrame(manifests).to_csv(output_dir / "holdout_manifest.csv", index=False)
    ceiling_frame.to_csv(output_dir / "full_development_reference.csv", index=False)
    stable_frame.to_csv(output_dir / "stabilization_points.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    return trajectory_frame, ceiling_frame, stable_frame, summary


def plot_learning_curves(trajectory, output_dir):
    for task in trajectory["task"].unique():
        for seed_per_level in sorted(trajectory["seed_per_level"].unique()):
            subset = trajectory[(trajectory["task"] == task) & (trajectory["seed_per_level"] == seed_per_level)]
            plt.figure(figsize=(9, 5.5))
            for strategy, group in subset.groupby("strategy"):
                aggregated = group.groupby("selected_compositions")["external_f1_macro"].agg(["mean", "std"]).reset_index()
                plt.plot(aggregated["selected_compositions"], aggregated["mean"], marker="o", label=strategy)
                plt.fill_between(aggregated["selected_compositions"], aggregated["mean"] - aggregated["std"].fillna(0), aggregated["mean"] + aggregated["std"].fillna(0), alpha=0.15)
            ceiling = subset.groupby("repeat")["ceiling_f1_macro"].first().mean()
            plt.axhline(ceiling, linestyle="--", label="full-development reference")
            plt.xlabel("Selected faulty compositions")
            plt.ylabel("External holdout macro-F1")
            plt.title(f"{task}: learning curve from {seed_per_level}+{seed_per_level}+{seed_per_level} seed")
            plt.ylim(0, 1.02)
            plt.legend(loc="best")
            plt.tight_layout()
            plt.savefig(output_dir / f"learning_curve_{task}_seed_{seed_per_level}.png", dpi=220)
            plt.close()


def plot_cost_frontiers(trajectory, output_dir):
    for task in trajectory["task"].unique():
        subset = trajectory[trajectory["task"] == task]
        plt.figure(figsize=(9, 5.5))
        for (strategy, seed_per_level), group in subset.groupby(["strategy", "seed_per_level"]):
            aggregated = group.groupby("selected_prob_seconds")["external_f1_macro"].mean().reset_index().sort_values("selected_prob_seconds")
            plt.plot(aggregated["selected_prob_seconds"], aggregated["external_f1_macro"], marker="o", label=f"{strategy}, seed {seed_per_level}/level")
        plt.xlabel("Cumulative ProB-reported execution time for selected development compositions (s)")
        plt.ylabel("External holdout macro-F1")
        plt.title(f"{task}: performance versus formal execution cost")
        plt.ylim(0, 1.02)
        plt.legend(loc="best", fontsize=7)
        plt.tight_layout()
        plt.savefig(output_dir / f"cost_frontier_{task}.png", dpi=220)
        plt.close()


def plot_stabilization(summary, output_dir):
    for task in summary["task"].unique():
        subset = summary[summary["task"] == task].copy()
        labels = [f"{row.strategy}\nseed {int(row.seed_per_level)}/level" for row in subset.itertuples()]
        values = subset["stable_compositions_mean"].to_numpy()
        plt.figure(figsize=(10, 5.5))
        bars = plt.bar(labels, values)
        for bar, row in zip(bars, subset.itertuples()):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"F1={row.stable_f1_mean:.3f}\nsave={row.cost_saved_mean:.0%}", ha="center", va="bottom", fontsize=8)
        plt.ylabel("Compositions required at stabilization")
        plt.title(f"{task}: stabilization budget and formal cost saving")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / f"stabilization_{task}.png", dpi=220)
        plt.close()


def plot_protocol(output_dir):
    plt.figure(figsize=(12, 3.2))
    ax = plt.gca()
    ax.axis("off")
    boxes = [(0.02, "Candidate mutant\ncompositions"), (0.20, "Reserve 20%\nexternal holdout"), (0.40, "Balanced seed\n5/5/5 or 10/10/10"), (0.61, "Query metadata\nwithout candidate traces"), (0.80, "Compose + execute\nselected candidates only")]
    for x, text in boxes:
        ax.text(x + 0.075, 0.55, text, ha="center", va="center", bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "black"}, transform=ax.transAxes)
    for x in [0.17, 0.37, 0.58, 0.77]:
        ax.annotate("", xy=(x + 0.03, 0.55), xytext=(x, 0.55), arrowprops={"arrowstyle": "->"}, xycoords=ax.transAxes)
    ax.text(0.5, 0.14, "Retrain one fixed LightGBM learner → test only on untouched holdout → record F1, stabilization, ProB time, and compositions", ha="center", va="center", transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(output_dir / "experimental_protocol.png", dpi=220, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", default="expanded_states.xlsx")
    parser.add_argument("--mutants", default="mutants.xlsx")
    parser.add_argument("--out_dir", default="additional_experiment_results")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed_sizes", default="5,10")
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--holdout_fraction", type=float, default=0.20)
    parser.add_argument("--stability_window", type=int, default=4)
    parser.add_argument("--stability_tolerance", type=float, default=0.005)
    parser.add_argument("--ceiling_gap", type=float, default=0.01)
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    parser.add_argument("--max_selected", type=int, default=60)
    args = parser.parse_args()
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata, samples, matrix, sample_mutants, vectorizer, cost_by_mutant = prepare_data(args.states, args.mutants)
    metadata.reset_index(drop=True).to_csv(output_dir / "mutant_metadata.csv", index=False)
    pd.DataFrame({"feature": vectorizer.get_feature_names_out()}).to_csv(output_dir / "feature_catalog.csv", index=False)
    trajectory, ceiling, stable, summary = run_experiments(metadata, matrix, sample_mutants, cost_by_mutant, output_dir, args.repeats, [int(x) for x in args.seed_sizes.split(",") if x.strip()], args.batch_size, args.holdout_fraction, args.stability_window, args.stability_tolerance, args.ceiling_gap, [x.strip() for x in args.strategies.split(",") if x.strip()], args.max_selected)
    plot_protocol(output_dir)
    plot_learning_curves(trajectory, output_dir)
    plot_cost_frontiers(trajectory, output_dir)
    plot_stabilization(summary, output_dir)
    config = vars(args)
    config["n_mutants"] = int(len(metadata))
    config["n_trace_samples"] = int(len(samples))
    config["n_features"] = int(matrix.shape[1])
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
