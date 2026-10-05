from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generalization_core import (
    COMPONENTS,
    build_trace_table,
    clean,
    evaluate_models,
    fit_component_threat_models,
    internal_group_cv,
    label_matrix,
    select_candidates,
    stabilization_budget,
    vectorize_train_test,
    write_json,
)


def load_pipeline(path: Path):
    spec = importlib.util.spec_from_file_location("original_active_learning_pipeline", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pipeline from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_seed_states(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    sheets = pd.ExcelFile(path).sheet_names
    sheet = "states_expanded" if "states_expanded" in sheets else sheets[0]
    return pd.read_excel(path, sheet_name=sheet)


def normalize_single_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["candidate_id"] = output["source_id"].map(clean)
    output["fault_count"] = 1
    output["fault_order"] = "single"
    output["component_set"] = output["component"].map(clean)
    output["threat_family_set"] = output["threat_family"].map(clean)
    output["edit_targets"] = output["target"].map(clean)
    output["events"] = output["event"].map(clean)
    output["mutation_kinds"] = output["mutation_kind"].map(clean)
    return output[
        [
            "candidate_id",
            "fault_count",
            "fault_order",
            "component_set",
            "threat_family_set",
            "edit_targets",
            "events",
            "mutation_kinds",
            "seed30_rank",
            "seed5_member",
            "seed30_member",
        ]
    ]


def candidate_lookup(manifest: pd.DataFrame, singles: pd.DataFrame) -> dict[str, dict[str, str]]:
    lookup = {}
    for _, row in manifest.iterrows():
        lookup[clean(row["candidate_id"])] = {
            "component_set": clean(row["component_set"]),
            "threat_family_set": clean(row["threat_family_set"]),
            "fault_order": clean(row["fault_order"]),
        }
    for _, row in singles.iterrows():
        lookup[clean(row["candidate_id"])] = {
            "component_set": clean(row["component_set"]),
            "threat_family_set": clean(row["threat_family_set"]),
            "fault_order": "single",
        }
    return lookup


def load_source_mutants(pipeline, mutants_xlsx: Path):
    mutants, _ = pipeline.read_mutants_xlsx(mutants_xlsx, pipeline.MUTANT_XLSX_MAPPING)
    return {mutant.mutant_id: mutant for mutant in mutants if len(mutant.edits) == 1 and mutant.mutant_id.startswith("M") and int(mutant.mutant_id[1:]) <= 30}


def proposed_candidate(pipeline, row: pd.Series, source_mutants: dict[str, object]):
    source_ids = [part.strip() for part in clean(row["source_ids"]).split(",") if part.strip()]
    edits = []
    for source_id in source_ids:
        if source_id not in source_mutants:
            raise KeyError(f"Missing source mutant {source_id}")
        source_edit = source_mutants[source_id].edits[0]
        edits.append(
            pipeline.Edit(
                location=source_edit.location,
                original=source_edit.original,
                mutated=source_edit.mutated,
                mutation_type="ACTIVE_LEARNING",
            )
        )
    return pipeline.ProposedMutant(
        mutant_id=clean(row["candidate_id"]),
        edits=edits,
        fault_count=int(row["fault_count"]),
        fault_components=clean(row["component_combo"]),
        mutation_type="ACTIVE_LEARNING",
        source_edit_ids=clean(row["source_ids"]),
    )


def ensure_candidate(
    pipeline,
    row: pd.Series,
    source_mutants: dict[str, object],
    tests,
    probcli: Path,
    baseline_export_dir: Path,
    baseline_eventb_name: str,
    cache_dir: Path,
    timeout_ms: int,
    logger,
) -> tuple[pd.DataFrame, dict[str, object]]:
    candidate_id = clean(row["candidate_id"])
    candidate_dir = cache_dir / candidate_id
    states_path = candidate_dir / "states_expanded.csv"
    log_path = candidate_dir / "execution.json"
    if states_path.exists() and log_path.exists():
        return pd.read_csv(states_path, low_memory=False), json.loads(log_path.read_text(encoding="utf-8"))
    candidate_dir.mkdir(parents=True, exist_ok=True)
    model_dir = candidate_dir / "model"
    runs_dir = candidate_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    mutant = proposed_candidate(pipeline, row, source_mutants)
    started = time.perf_counter()
    model_eventb, applied, failed, mutation_log = pipeline.apply_mutant_to_model(
        baseline_export_dir,
        baseline_eventb_name,
        mutant,
        model_dir,
    )
    runs, states, parse_failures = pipeline.harvest_mutant_traces(
        probcli=probcli,
        model_eventb=model_eventb,
        tests=tests,
        mutant=mutant,
        runs_dir=runs_dir,
        timeout_ms=timeout_ms,
        clpfd=True,
        logger=logger,
    )
    elapsed = float(time.perf_counter() - started)
    states.to_csv(states_path, index=False)
    runs.to_csv(candidate_dir / "runs.csv", index=False)
    pd.DataFrame(mutation_log).to_csv(candidate_dir / "mutation_log.csv", index=False)
    parse_failures.to_csv(candidate_dir / "parse_failures.csv", index=False)
    record = {
        "candidate_id": candidate_id,
        "source_ids": clean(row["source_ids"]),
        "fault_order": clean(row["fault_order"]),
        "applied_edits": int(applied),
        "failed_edits": int(failed),
        "test_cases": int(len(tests)),
        "successful_test_rate": float(runs["ok"].mean()) if not runs.empty and "ok" in runs.columns else math.nan,
        "parse_failures": int(len(parse_failures)),
        "composition_and_trace_seconds": elapsed,
        "states_rows": int(len(states)),
    }
    write_json(log_path, record)
    return states, record


def states_for_ids(
    ids: list[str],
    manifest_index: pd.DataFrame,
    single_states: pd.DataFrame,
    pipeline,
    source_mutants,
    tests,
    probcli,
    baseline_export_dir,
    baseline_eventb_name,
    cache_dir,
    timeout_ms,
    logger,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    frames = []
    records = []
    single_ids = set(single_states["mutant_id"].map(clean).unique())
    for candidate_id in ids:
        if candidate_id in single_ids:
            frames.append(single_states[single_states["mutant_id"].map(clean) == candidate_id].copy())
            continue
        row = manifest_index.loc[candidate_id]
        states, record = ensure_candidate(
            pipeline,
            row,
            source_mutants,
            tests,
            probcli,
            baseline_export_dir,
            baseline_eventb_name,
            cache_dir,
            timeout_ms,
            logger,
        )
        frames.append(states)
        records.append(record)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), records


def train_evaluate(
    train_table: pd.DataFrame,
    hidden_table: pd.DataFrame,
    component_classes: list[str],
    threat_classes: list[str],
    task: str,
    seed: int,
) -> dict[str, float]:
    _, x_train, x_hidden = vectorize_train_test(train_table["features"].tolist(), hidden_table["features"].tolist())
    component_train = label_matrix(train_table["component_set"].tolist(), component_classes)
    component_hidden = label_matrix(hidden_table["component_set"].tolist(), component_classes)
    threat_train = label_matrix(train_table["threat_family_set"].tolist(), threat_classes)
    threat_hidden = label_matrix(hidden_table["threat_family_set"].tolist(), threat_classes)
    models = fit_component_threat_models(
        x_train,
        component_train,
        threat_train,
        train_table["mutant_id"].to_numpy(),
        seed,
        task,
    )
    metrics = evaluate_models(models, x_hidden, component_hidden, threat_hidden)
    metrics["train_seconds"] = float(models["train_seconds"])
    return metrics


def plot_results(results: pd.DataFrame, out_dir: Path) -> None:
    if results.empty:
        return
    for task in sorted(results["task"].unique()):
        for seed_size in sorted(results["seed_per_order"].unique()):
            subset = results[(results["task"] == task) & (results["seed_per_order"] == seed_size)]
            fig, axis = plt.subplots(figsize=(10, 6))
            for strategy, group in subset.groupby("strategy"):
                summary = group.groupby("budget_pct", as_index=False)["external_primary_f1"].agg(["mean", "std"]).reset_index()
                x = summary["budget_pct"].to_numpy()
                y = summary["mean"].to_numpy() * 100
                spread = summary["std"].fillna(0).to_numpy() * 100
                axis.plot(x, y, marker="o", linewidth=2, label=strategy)
                axis.fill_between(x, y - spread, y + spread, alpha=0.12)
            axis.set_title(f"External Generalization | {task} | Seed {seed_size}+{seed_size}+{seed_size}")
            axis.set_xlabel("Additional queried budget (% of development pool)")
            axis.set_ylabel("F1-Macro (%)")
            axis.set_xticks(sorted(subset["budget_pct"].unique()))
            axis.grid(True, alpha=0.25)
            axis.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(out_dir / f"external_curve_{task}_seed_{seed_size}.png", dpi=220)
            plt.close(fig)
    fig, axis = plt.subplots(figsize=(11, 6.5))
    grouped = results.groupby(["task", "seed_per_order", "strategy", "training_candidate_compositions"], as_index=False)["external_primary_f1"].mean()
    for (task, seed_size, strategy), group in grouped.groupby(["task", "seed_per_order", "strategy"]):
        group = group.sort_values("training_candidate_compositions")
        axis.plot(
            group["training_candidate_compositions"],
            group["external_primary_f1"] * 100,
            marker="o",
            linewidth=1.5,
            label=f"{task} | {strategy} | {seed_size}+{seed_size}+{seed_size}",
        )
    axis.set_title("Composition Cost versus External Generalization")
    axis.set_xlabel("Development compositions executed for training")
    axis.set_ylabel("External F1-Macro (%)")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "cost_generalization_frontier.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_script", required=True)
    parser.add_argument("--probcli", required=True)
    parser.add_argument("--baseline_export_dir", required=True)
    parser.add_argument("--baseline_eventb_name", default="Ventilator_Composed_mch.eventb")
    parser.add_argument("--tests_xlsx", required=True)
    parser.add_argument("--mutants_xlsx", required=True)
    parser.add_argument("--seed_states", required=True)
    parser.add_argument("--candidate_manifest_csv", required=True)
    parser.add_argument("--single_seed_manifest_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed_sizes", default="5,30")
    parser.add_argument("--budgets", default="1,2,4,8,16")
    parser.add_argument("--strategies", default="random,uncertainty_entropy,uncertainty_margin,diversity_jaccard")
    parser.add_argument("--tasks", default="component,threat,joint")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout_ms", type=int, default=8000)
    parser.add_argument("--holdout_limit", type=int, default=0)
    parser.add_argument("--run_ceiling", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "trace_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pipeline = load_pipeline(Path(args.pipeline_script).resolve())
    logger = pipeline.setup_logger(out_dir / "logs")
    probcli = Path(args.probcli).resolve()
    baseline_export_dir = Path(args.baseline_export_dir).resolve()
    tests = pipeline.read_tests_xlsx(Path(args.tests_xlsx).resolve(), pipeline.TEST_XLSX_MAPPING)
    source_mutants = load_source_mutants(pipeline, Path(args.mutants_xlsx).resolve())
    manifest = pd.read_csv(Path(args.candidate_manifest_csv).resolve())
    manifest["candidate_id"] = manifest["candidate_id"].map(clean)
    manifest_index = manifest.set_index("candidate_id", drop=False)
    singles_raw = pd.read_csv(Path(args.single_seed_manifest_csv).resolve())
    singles = normalize_single_metadata(singles_raw)
    seed_states = read_seed_states(Path(args.seed_states).resolve())
    seed_states["mutant_id"] = seed_states["mutant_id"].map(clean)
    seed_states = seed_states[seed_states["mutant_id"].isin(singles["candidate_id"])].copy()
    lookup = candidate_lookup(manifest, singles)
    threat_classes = sorted(
        {
            clean(value)
            for value in singles["threat_family_set"].tolist()
            if clean(value)
        }
    )
    hidden_manifest = manifest[manifest["split"] == "hidden_test"].copy().sort_values("holdout_rank")
    if args.holdout_limit > 0:
        hidden_manifest = hidden_manifest.head(args.holdout_limit)
    hidden_ids = hidden_manifest["candidate_id"].tolist()
    hidden_states, hidden_execution_records = states_for_ids(
        hidden_ids,
        manifest_index,
        seed_states,
        pipeline,
        source_mutants,
        tests,
        probcli,
        baseline_export_dir,
        args.baseline_eventb_name,
        cache_dir,
        args.timeout_ms,
        logger,
    )
    hidden_table = build_trace_table(hidden_states, lookup)
    development = manifest[manifest["split"] == "development"].copy()
    budgets = [0.0] + [float(value) for value in args.budgets.split(",") if value.strip()]
    strategies = [value.strip() for value in args.strategies.split(",") if value.strip()]
    tasks = [value.strip() for value in args.tasks.split(",") if value.strip()]
    seed_sizes = [int(value) for value in args.seed_sizes.split(",") if value.strip()]
    result_rows = []
    selection_rows = []
    execution_records = list(hidden_execution_records)
    for seed_size in seed_sizes:
        single_ids = singles[singles["seed30_rank"] <= seed_size]["candidate_id"].tolist()
        seed_candidates = development[development["seed30_rank"] <= seed_size].copy()
        seed_candidate_ids = seed_candidates["candidate_id"].tolist()
        seed_candidate_states, seed_records = states_for_ids(
            seed_candidate_ids,
            manifest_index,
            seed_states,
            pipeline,
            source_mutants,
            tests,
            probcli,
            baseline_export_dir,
            args.baseline_eventb_name,
            cache_dir,
            args.timeout_ms,
            logger,
        )
        execution_records.extend(seed_records)
        single_seed_states = seed_states[seed_states["mutant_id"].isin(single_ids)].copy()
        base_states = pd.concat([single_seed_states, seed_candidate_states], ignore_index=True)
        for task in tasks:
            for repeat in range(args.repeats):
                for strategy in strategies:
                    selected_ids = []
                    previous_count = 0
                    run_history = []
                    for checkpoint, budget_pct in enumerate(budgets):
                        target_count = int(math.ceil((budget_pct / 100.0) * len(development)))
                        target_count = min(target_count, len(development) - len(seed_candidate_ids))
                        additional = max(0, target_count - previous_count)
                        if additional:
                            labeled_metadata = pd.concat(
                                [
                                    singles[singles["candidate_id"].isin(single_ids)],
                                    development[development["candidate_id"].isin(seed_candidate_ids + selected_ids)],
                                ],
                                ignore_index=True,
                            )
                            remaining = development[
                                ~development["candidate_id"].isin(seed_candidate_ids + selected_ids)
                            ].copy()
                            chosen = select_candidates(
                                labeled_metadata,
                                remaining,
                                additional,
                                strategy,
                                task,
                                args.seed + repeat * 10000 + seed_size * 1000 + checkpoint,
                                COMPONENTS,
                                threat_classes,
                            )
                            selected_ids.extend(chosen)
                            for candidate_id in chosen:
                                selection_rows.append(
                                    {
                                        "task": task,
                                        "seed_per_order": seed_size,
                                        "repeat": repeat + 1,
                                        "strategy": strategy,
                                        "budget_pct": budget_pct,
                                        "selection_order": len(selected_ids),
                                        "candidate_id": candidate_id,
                                    }
                                )
                        previous_count = len(selected_ids)
                        selected_states, selected_records = states_for_ids(
                            selected_ids,
                            manifest_index,
                            seed_states,
                            pipeline,
                            source_mutants,
                            tests,
                            probcli,
                            baseline_export_dir,
                            args.baseline_eventb_name,
                            cache_dir,
                            args.timeout_ms,
                            logger,
                        )
                        execution_records.extend(selected_records)
                        train_states = pd.concat([base_states, selected_states], ignore_index=True)
                        train_table = build_trace_table(train_states, lookup)
                        internal_started = time.perf_counter()
                        internal_f1 = internal_group_cv(
                            train_table,
                            COMPONENTS,
                            threat_classes,
                            task,
                            args.seed + repeat + checkpoint,
                        )
                        internal_seconds = float(time.perf_counter() - internal_started)
                        external_metrics = train_evaluate(
                            train_table,
                            hidden_table,
                            COMPONENTS,
                            threat_classes,
                            task,
                            args.seed + repeat + checkpoint,
                        )
                        row = {
                            "task": task,
                            "seed_per_order": seed_size,
                            "repeat": repeat + 1,
                            "strategy": strategy,
                            "budget_pct": budget_pct,
                            "single_seed_compositions": len(single_ids),
                            "multi_seed_compositions": len(seed_candidate_ids),
                            "queried_compositions": len(selected_ids),
                            "training_candidate_compositions": len(seed_candidate_ids) + len(selected_ids),
                            "total_training_compositions": len(single_ids) + len(seed_candidate_ids) + len(selected_ids),
                            "hidden_test_compositions": len(hidden_ids),
                            "training_prob_executions": (len(seed_candidate_ids) + len(selected_ids)) * len(tests),
                            "hidden_prob_executions": len(hidden_ids) * len(tests),
                            "cost_saved_vs_full_development": 1.0 - (len(seed_candidate_ids) + len(selected_ids)) / len(development),
                            "internal_cv_f1": internal_f1,
                            "internal_cv_seconds": internal_seconds,
                            "external_primary_f1": external_metrics["primary_f1"],
                            "external_component_f1": external_metrics["component_f1_macro"],
                            "external_threat_f1": external_metrics["threat_f1_macro"],
                            "external_component_exact_match": external_metrics["component_exact_match"],
                            "external_threat_exact_match": external_metrics["threat_exact_match"],
                            "train_seconds": external_metrics["train_seconds"],
                            "infer_seconds": external_metrics["infer_seconds"],
                        }
                        result_rows.append(row)
                        run_history.append(row)
                    history_frame = pd.DataFrame(run_history)
                    stable = stabilization_budget(history_frame)
                    for row in result_rows[-len(run_history):]:
                        row["stabilization_budget_pct"] = stable
    if args.run_ceiling:
        all_single_ids = singles["candidate_id"].tolist()
        all_development_ids = development["candidate_id"].tolist()
        ceiling_states, ceiling_records = states_for_ids(
            all_development_ids,
            manifest_index,
            seed_states,
            pipeline,
            source_mutants,
            tests,
            probcli,
            baseline_export_dir,
            args.baseline_eventb_name,
            cache_dir,
            args.timeout_ms,
            logger,
        )
        execution_records.extend(ceiling_records)
        ceiling_seed_states = seed_states[seed_states["mutant_id"].isin(all_single_ids)].copy()
        ceiling_table = build_trace_table(pd.concat([ceiling_seed_states, ceiling_states], ignore_index=True), lookup)
        for task in tasks:
            ceiling_internal_started = time.perf_counter()
            ceiling_internal_f1 = internal_group_cv(
                ceiling_table,
                COMPONENTS,
                threat_classes,
                task,
                args.seed + 900000,
            )
            ceiling_internal_seconds = float(time.perf_counter() - ceiling_internal_started)
            ceiling_metrics = train_evaluate(
                ceiling_table,
                hidden_table,
                COMPONENTS,
                threat_classes,
                task,
                args.seed + 900000,
            )
            result_rows.append(
                {
                    "task": task,
                    "seed_per_order": 30,
                    "repeat": 1,
                    "strategy": "full_development_ceiling",
                    "budget_pct": 100.0,
                    "single_seed_compositions": len(all_single_ids),
                    "multi_seed_compositions": len(development),
                    "queried_compositions": 0,
                    "training_candidate_compositions": len(development),
                    "total_training_compositions": len(all_single_ids) + len(development),
                    "hidden_test_compositions": len(hidden_ids),
                    "training_prob_executions": len(development) * len(tests),
                    "hidden_prob_executions": len(hidden_ids) * len(tests),
                    "cost_saved_vs_full_development": 0.0,
                    "internal_cv_f1": ceiling_internal_f1,
                    "internal_cv_seconds": ceiling_internal_seconds,
                    "external_primary_f1": ceiling_metrics["primary_f1"],
                    "external_component_f1": ceiling_metrics["component_f1_macro"],
                    "external_threat_f1": ceiling_metrics["threat_f1_macro"],
                    "external_component_exact_match": ceiling_metrics["component_exact_match"],
                    "external_threat_exact_match": ceiling_metrics["threat_exact_match"],
                    "train_seconds": ceiling_metrics["train_seconds"],
                    "infer_seconds": ceiling_metrics["infer_seconds"],
                    "stabilization_budget_pct": 100.0,
                }
            )
    results = pd.DataFrame(result_rows)
    selections = pd.DataFrame(selection_rows)
    execution_log = pd.DataFrame(execution_records).drop_duplicates("candidate_id", keep="last") if execution_records else pd.DataFrame()
    results.to_csv(out_dir / "heldout_generalization_results.csv", index=False)
    selections.to_csv(out_dir / "heldout_generalization_selections.csv", index=False)
    execution_log.to_csv(out_dir / "composition_execution_log.csv", index=False)
    hidden_manifest.to_csv(out_dir / "hidden_test_manifest.csv", index=False)
    development.to_csv(out_dir / "development_pool_manifest.csv", index=False)
    if not results.empty:
        stabilization = (
            results.groupby(["task", "seed_per_order", "strategy", "repeat"], as_index=False)
            .agg(
                stabilization_budget_pct=("stabilization_budget_pct", "first"),
                best_external_f1=("external_primary_f1", "max"),
                final_external_f1=("external_primary_f1", "last"),
                max_training_candidate_compositions=("training_candidate_compositions", "max"),
            )
        )
        stabilization.to_csv(out_dir / "stabilization_summary.csv", index=False)
    plot_results(results, out_dir)
    config = {
        "holdout_compositions": len(hidden_ids),
        "development_compositions": len(development),
        "holdout_fraction": len(hidden_ids) / len(manifest),
        "seed_sizes": seed_sizes,
        "budgets": budgets,
        "strategies": strategies,
        "tasks": tasks,
        "repeats": args.repeats,
        "test_cases": len(tests),
        "stopping_rule": "first predefined checkpoint whose internal group-CV F1 is within 0.005 of the previous two checkpoints",
        "external_test_usage": "reported at predefined checkpoints only and never used for selection or stopping",
        "run_ceiling": bool(args.run_ceiling),
    }
    write_json(out_dir / "run_config.json", config)
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
