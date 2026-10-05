from __future__ import annotations

import argparse
import importlib.util
import logging
import shutil
import sys
import time
from pathlib import Path

import pandas as pd


def load_module(path):
    spec = importlib.util.spec_from_file_location("mlv_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy_pipeline", required=True)
    parser.add_argument("--candidate_combinations", required=True)
    parser.add_argument("--execution_requests", required=True)
    parser.add_argument("--tests", required=True)
    parser.add_argument("--baseline_export", required=True)
    parser.add_argument("--baseline_eventb_name", default="Ventilator_Composed_mch.eventb")
    parser.add_argument("--probcli", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--timeout_ms", type=int, default=8000)
    parser.add_argument("--no_clpfd", action="store_true")
    args = parser.parse_args()
    pipeline = load_module(Path(args.legacy_pipeline))
    manifest = pd.read_csv(args.candidate_combinations).set_index("candidate_id")
    requests = pd.read_csv(args.execution_requests)
    ids = requests["candidate_id"].dropna().astype(str).drop_duplicates().tolist()
    tests = pipeline.read_tests_xlsx(Path(args.tests), pipeline.TEST_XLSX_MAPPING)
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("candidate_execution")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())
    timing = []
    for candidate_id in ids:
        target = cache / candidate_id
        states_path = target / "states_expanded.csv"
        if states_path.exists():
            timing.append({"candidate_id": candidate_id, "status": "CACHE_HIT", "total_seconds": 0.0})
            continue
        row = manifest.loc[candidate_id]
        sources = str(row["source_mutants"]).split("+")
        targets = str(row["edit_targets"]).split("+")
        originals = str(row["edit_originals"]).split(" || ")
        mutated = str(row["edit_mutated_values"]).split(" || ")
        edits = [
            pipeline.Edit(location=location, original=original, mutated=new_value, mutation_type="ACTIVE_LEARNING")
            for location, original, new_value in zip(targets, originals, mutated)
        ]
        mutant = pipeline.ProposedMutant(
            mutant_id=candidate_id,
            edits=edits,
            fault_count=int(row["fault_count"]),
            fault_components=str(row["affected_components"]),
            mutation_type="ACTIVE_LEARNING",
            source_edit_ids="+".join(sources),
        )
        work = target / "work"
        if work.exists():
            shutil.rmtree(work)
        baseline_copy = work / "baseline"
        pipeline.copy_baseline_export(Path(args.baseline_export), baseline_copy)
        started = time.perf_counter()
        model_eventb, applied, failed, edit_log = pipeline.apply_mutant_to_model(
            baseline_copy,
            args.baseline_eventb_name,
            mutant,
            work / "model",
        )
        runs, states, parse_fails = pipeline.harvest_mutant_traces(
            Path(args.probcli),
            model_eventb,
            tests,
            mutant,
            target / "runs",
            args.timeout_ms,
            not args.no_clpfd,
            logger,
        )
        elapsed = time.perf_counter() - started
        states["threat_class"] = str(row["threat_types"])
        states["edit_count"] = int(row["fault_count"])
        states["edit_targets"] = str(row["edit_targets"])
        states["mutate_applied_edits"] = applied
        states["mutate_failed_edits"] = failed
        runs["threat_class"] = str(row["threat_types"])
        target.mkdir(parents=True, exist_ok=True)
        states.to_csv(states_path, index=False)
        runs.to_csv(target / "execution_log.csv", index=False)
        parse_fails.to_csv(target / "parse_failures.csv", index=False)
        pd.DataFrame(edit_log).to_csv(target / "edit_application_log.csv", index=False)
        timing.append({
            "candidate_id": candidate_id,
            "status": "EXECUTED",
            "total_seconds": elapsed,
            "successful_test_runs": int(runs["ok"].sum()) if "ok" in runs else 0,
            "failed_test_runs": int((~runs["ok"].astype(bool)).sum()) if "ok" in runs else len(tests),
        })
    timing_frame = pd.DataFrame(timing)
    timing_path = cache / "execution_timing.csv"
    if timing_path.exists():
        timing_frame = pd.concat([pd.read_csv(timing_path), timing_frame], ignore_index=True)
    timing_frame.to_csv(timing_path, index=False)


if __name__ == "__main__":
    main()
