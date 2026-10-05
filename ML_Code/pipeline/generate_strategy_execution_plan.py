from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


def structural_features(row: dict) -> dict[str, float]:
    result = {"fault_count": float(row.get("number_of_threats", row.get("fault_count", 0)) or 0)}
    for field in ["affected_components", "threat_types", "edit_targets", "source_single_mutants", "source_mutants"]:
        value = str(row.get(field, ""))
        for token in value.replace("+", " ").replace("/", " ").replace("::", " ").split():
            token = token.strip().lower()
            if token:
                result[f"token::{token}"] = 1.0
    return result


def task_label(row: dict, task: str) -> str:
    if task == "Task A: Component":
        return str(row.get("affected_components", "UNKNOWN"))
    if task == "Task B: Threat Type":
        return str(row.get("threat_types", "UNKNOWN"))
    return f'{row.get("affected_components", "UNKNOWN")} || {row.get("threat_types", "UNKNOWN")}'


def tokens(row: dict) -> set[str]:
    return {k for k, v in structural_features(row).items() if k.startswith("token::") and v}


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return 0.0 if not union else 1.0 - len(a & b) / len(union)


def choose_uncertainty(strategy: str, pool: list[dict], labeled: list[dict], count: int, task: str, seed: int) -> list[dict]:
    labels = [task_label(row, task) for row in labeled]
    rng = np.random.default_rng(seed)
    if len(set(labels)) < 2:
        idx = rng.choice(len(pool), size=min(count, len(pool)), replace=False)
        return [pool[int(i)] for i in idx]
    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform([structural_features(row) for row in labeled])
    x_pool = vectorizer.transform([structural_features(row) for row in pool])
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    model = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
    model.fit(x_train, y)
    probabilities = model.predict_proba(x_pool)
    if strategy == "Entropy":
        scores = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum(axis=1)
        order = np.argsort(-scores, kind="stable")
    else:
        sorted_p = np.sort(probabilities, axis=1)
        margins = sorted_p[:, -1] - sorted_p[:, -2]
        order = np.argsort(margins, kind="stable")
    return [pool[int(i)] for i in order[:count]]


def jaccard_sequence(pool: list[dict], labeled: list[dict], target: int) -> list[dict]:
    remaining = list(pool)
    selected: list[dict] = []
    reference_tokens = [tokens(row) for row in labeled]
    if not reference_tokens:
        raise ValueError("Initial set cannot be empty for Jaccard planning")
    # Current minimum distance to any labeled/selected item.
    min_distance = np.array([
        min(jaccard(tokens(row), ref) for ref in reference_tokens)
        for row in remaining
    ], dtype=float)
    while remaining and len(selected) < target:
        index = int(np.argmax(min_distance))
        chosen = remaining.pop(index)
        selected.append(chosen)
        min_distance = np.delete(min_distance, index)
        if not remaining:
            break
        chosen_tokens = tokens(chosen)
        new_distances = np.array([jaccard(tokens(row), chosen_tokens) for row in remaining], dtype=float)
        min_distance = np.minimum(min_distance, new_distances)
    return selected


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate_combinations", required=True)
    p.add_argument("--initial_sets", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    candidates = pd.read_csv(args.candidate_combinations)
    initial_catalog = pd.read_csv(args.initial_sets)
    metadata = pd.read_csv(args.metadata)
    existing_ids = set(metadata["candidate_id"].astype(str))
    hidden = candidates[candidates["candidate_set"].eq("Hidden Test Set")]
    available = candidates[candidates["candidate_set"].eq("Available Candidate Set")]
    available_rows_all = available.to_dict("records")
    percentages = config["candidate_checkpoints_percent"]
    targets = [(f"{p}%", int(math.ceil(len(available_rows_all) * p / 100))) for p in percentages]
    max_target = max(value for _, value in targets)

    selection_rows: list[dict] = []
    requested: dict[str, set[str]] = {}
    for cid in hidden["candidate_id"].astype(str):
        if cid not in existing_ids:
            requested.setdefault(cid, set()).add("Hidden Test Set")

    # Cache Jaccard sequences because they are independent of task and repeat.
    jaccard_cache: dict[str, list[dict]] = {}

    for initial_name in config["initial_sets"]:
        init_rows_raw = initial_catalog[initial_catalog["initial_set"].eq(initial_name)].to_dict("records")
        init_rows = [dict(row, source_mutants=row.get("system_id", "")) for row in init_rows_raw]
        initial_candidate_ids = {
            str(row.get("candidate_id", "")) for row in init_rows_raw
            if str(row.get("candidate_id", "")).strip() and str(row.get("candidate_id", "")).lower() != "nan"
        }
        base_pool = [row for row in available_rows_all if str(row["candidate_id"]) not in initial_candidate_ids]
        jaccard_cache[initial_name] = jaccard_sequence(base_pool, init_rows, max_target)

        for repeat in range(1, int(config["repeats"]) + 1):
            for task in config["tasks"]:
                for strategy in config["selection_strategies"]:
                    selected: list[dict] = []
                    pool = list(base_pool)
                    labeled = list(init_rows)
                    for checkpoint, target in targets:
                        needed = target - len(selected)
                        if needed <= 0:
                            continue
                        seed = int(config["random_seed"]) + repeat * 1000 + target
                        if strategy == "Random":
                            rng = np.random.default_rng(seed)
                            indices = rng.choice(len(pool), size=min(needed, len(pool)), replace=False)
                            batch = [pool[int(i)] for i in indices]
                        elif strategy in {"Entropy", "Margin"}:
                            batch = choose_uncertainty(strategy, pool, labeled, needed, task, seed)
                        else:
                            # Use the cached prefix; remove anything already selected.
                            chosen_ids = {str(row["candidate_id"]) for row in selected}
                            batch = [row for row in jaccard_cache[initial_name] if str(row["candidate_id"]) not in chosen_ids][:needed]
                        batch_ids = {str(row["candidate_id"]) for row in batch}
                        pool = [row for row in pool if str(row["candidate_id"]) not in batch_ids]
                        selected.extend(batch)
                        labeled.extend(batch)
                        start_rank = len(selected) - len(batch) + 1
                        for offset, row in enumerate(batch):
                            cid = str(row["candidate_id"])
                            rank = start_rank + offset
                            purpose = f"{initial_name} | repeat {repeat} | {task} | {strategy} | through {checkpoint}"
                            selection_rows.append({
                                "repeat": repeat,
                                "initial_set": initial_name,
                                "task": task,
                                "selection_strategy": strategy,
                                "checkpoint_added": checkpoint,
                                "selection_order": rank,
                                "candidate_id": cid,
                                "fault_count": row.get("fault_count", ""),
                                "affected_components": row.get("affected_components", ""),
                                "threat_types": row.get("threat_types", ""),
                            })
                            if cid not in existing_ids:
                                requested.setdefault(cid, set()).add(purpose)

    pd.DataFrame(selection_rows).to_csv(out / "planned_selection_history.csv", index=False)
    request_rows = [
        {"candidate_id": cid, "purpose": " ; ".join(sorted(purposes))}
        for cid, purposes in sorted(requested.items())
    ]
    pd.DataFrame(request_rows).to_csv(out / "execution_requests.csv", index=False)
    summary = {
        "hidden_test_candidates": int(len(hidden)),
        "available_candidates": int(len(available)),
        "existing_trace_systems": int(len(existing_ids)),
        "unique_candidates_requiring_ProB": int(len(request_rows)),
        "planned_selection_rows": int(len(selection_rows)),
        "note": "Each requested composition is executed once on all 30 test cases; the resulting trace cache is shared by all five diagnostic classifiers."
    }
    (out / "execution_plan_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
