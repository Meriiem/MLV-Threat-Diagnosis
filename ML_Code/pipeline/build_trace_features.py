from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer


def load_module(path):
    spec = importlib.util.spec_from_file_location("mlv_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_states(path):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    return pd.read_excel(path, sheet_name="states_expanded")


def normalize_plus(value):
    parts = [part.strip() for part in str(value or "").split("+") if part.strip()]
    return "+".join(sorted(dict.fromkeys(parts)))


def normalize_threats(value):
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return "UNKNOWN"
    parts = [part.strip() for part in text.split(" + ") if part.strip()]
    return " + ".join(sorted(dict.fromkeys(parts)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_states", required=True)
    parser.add_argument("--execution_cache", default="")
    parser.add_argument("--legacy_pipeline", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    frames = [read_states(Path(args.base_states))]
    if args.execution_cache:
        for path in sorted(Path(args.execution_cache).glob("*/states_expanded.csv")):
            frames.append(pd.read_csv(path, low_memory=False))
    states = pd.concat(frames, ignore_index=True, sort=False)
    states = states.drop_duplicates(subset=["mutant_id", "tc_id", "step", "event"], keep="last")
    pipeline = load_module(Path(args.legacy_pipeline))
    sample_keys, feature_dicts, metadata_rows = pipeline.build_samples_from_states(states)
    labels = {}
    for (mutant_id, tc_id), group in states.groupby(["mutant_id", "tc_id"], sort=False):
        components = normalize_plus(group["fault_components"].iloc[0] if "fault_components" in group else "") or "UNKNOWN"
        threats = normalize_threats(group["threat_class"].iloc[0] if "threat_class" in group else "")
        labels[f"{mutant_id}|{tc_id}"] = {
            "task_a_label": components,
            "task_b_label": threats,
            "task_c_label": f"{components} || {threats}",
            "fault_count": group["fault_count"].iloc[0] if "fault_count" in group else "",
            "fault_order": group["fault_order"].iloc[0] if "fault_order" in group else "",
        }
    vectorizer = DictVectorizer(sparse=True)
    matrix = vectorizer.fit_transform(feature_dicts)
    metadata = []
    for key, row in zip(sample_keys, metadata_rows):
        label = labels[key]
        metadata.append({
            "sample_key": key,
            "candidate_id": row["mutant_id"],
            "test_case_id": row["tc_id"],
            **label,
        })
    sparse.save_npz(output / "trace_features.npz", matrix.tocsr())
    pd.DataFrame(metadata).to_csv(output / "trace_metadata.csv", index=False)
    joblib.dump(vectorizer, output / "feature_vectorizer.joblib")
    (output / "feature_schema.json").write_text(json.dumps({
        "sample_rows": int(matrix.shape[0]),
        "feature_columns": int(matrix.shape[1]),
        "feature_names": vectorizer.get_feature_names_out().tolist(),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
