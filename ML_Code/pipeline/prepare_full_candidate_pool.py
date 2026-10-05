from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

import pandas as pd


def clean(value):
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def signature(target, original, mutated):
    return "||".join([clean(target), clean(original), clean(mutated)])


def component(target):
    text = str(target).lower()
    if "alarm" in text or "acknowledge" in text or "deactivatealarm" in text:
        return "ALRM"
    if "gui" in text or "freeze" in text or "unfreeze" in text or "navigate" in text:
        return "GUI"
    return "CTRL"


def source_catalog(frame):
    singles = frame[frame["Fault_Count"] == 1].copy().sort_values("ID")
    source = []
    signature_map = {}
    for _, row in singles.iterrows():
        mutant_id = str(row["ID"])
        edit_signature = signature(row["Edit1_Target"], row["Edit1_Original"], row["Edit1_Mutated"])
        signature_map[edit_signature] = mutant_id
        source.append({
            "mutant_id": mutant_id,
            "component": str(row["Fault_Components"]),
            "threat": str(row["Threat_Family"]),
            "pair": f"{str(row['Fault_Components'])}::{str(row['Threat_Family'])}",
            "target": clean(row["Edit1_Target"]),
        })
    return source, signature_map


def historical_compositions(frame, signature_map):
    mapped = {}
    audit = []
    for _, row in frame[frame["Fault_Count"] > 1].iterrows():
        source_ids = []
        for index in [1, 2, 3]:
            target = row.get(f"Edit{index}_Target")
            if pd.isna(target) or not str(target).strip():
                continue
            edit_signature = signature(target, row.get(f"Edit{index}_Original"), row.get(f"Edit{index}_Mutated"))
            source_id = signature_map.get(edit_signature)
            if source_id:
                source_ids.append(source_id)
            else:
                audit.append({
                    "mutant_id": str(row["ID"]),
                    "fault_count": int(row["Fault_Count"]),
                    "edit_index": index,
                    "target": clean(target),
                    "original": clean(row.get(f"Edit{index}_Original")),
                    "mutated": clean(row.get(f"Edit{index}_Mutated")),
                    "issue": "No exact source edit among M001-M030",
                })
        if len(source_ids) == int(row["Fault_Count"]):
            mapped[tuple(sorted(source_ids))] = str(row["ID"])
    return mapped, pd.DataFrame(audit)


def build_manifest(source, historical, holdout_fraction, seed):
    rows = []
    rng = random.Random(seed)
    for fault_count, prefix in [(2, "D"), (3, "T")]:
        combinations = list(itertools.combinations(source, fault_count))
        for combination in combinations:
            source_ids = tuple(sorted(item["mutant_id"] for item in combination))
            rows.append({
                "candidate_id": f"{prefix}_" + "_".join(source_ids),
                "fault_count": fault_count,
                "source_mutants": "+".join(source_ids),
                "affected_components": "+".join(sorted({item["component"] for item in combination})),
                "threat_families": "+".join(sorted({item["threat"] for item in combination})),
                "component_threat_pairs": "+".join(sorted({item["pair"] for item in combination})),
                "edit_targets": "+".join(sorted({item["target"] for item in combination})),
                "historical_mutant_id": historical.get(source_ids, ""),
                "historical_seed_eligible": source_ids in historical,
                "split": "development_pool",
                "test_cases_per_candidate": 30,
                "planned_prob_executions": 30,
                "trace_status": "not_generated",
            })
    manifest = pd.DataFrame(rows)
    for fault_count in [2, 3]:
        indexes = manifest[(manifest["fault_count"] == fault_count) & (~manifest["historical_seed_eligible"])].index.tolist()
        rng.shuffle(indexes)
        count = round(len(indexes) * holdout_fraction)
        manifest.loc[indexes[:count], "split"] = "external_holdout"
    manifest.loc[manifest["historical_seed_eligible"], "split"] = "historical_seed_pool"
    return manifest


def plan(manifest, audit):
    rows = []
    for fault_count in [2, 3]:
        subset = manifest[manifest["fault_count"] == fault_count]
        rows.append({"section": "Pool split", "item": f"{fault_count}-threat theoretical total", "value": len(subset), "unit": "compositions"})
        for split in ["historical_seed_pool", "development_pool", "external_holdout"]:
            rows.append({"section": "Pool split", "item": f"{fault_count}-threat {split}", "value": int((subset["split"] == split).sum()), "unit": "compositions"})
    development = int((manifest["split"] == "development_pool").sum())
    external = int((manifest["split"] == "external_holdout").sum())
    rows.extend([
        {"section": "Pool split", "item": "Theoretical candidate total", "value": len(manifest), "unit": "compositions"},
        {"section": "Pool split", "item": "Mapped historical multi-threat seeds", "value": int(manifest["historical_seed_eligible"].sum()), "unit": "compositions"},
        {"section": "Pool split", "item": "Development pool total", "value": development, "unit": "compositions"},
        {"section": "Pool split", "item": "External holdout reserve", "value": external, "unit": "compositions"},
        {"section": "Pool split", "item": "Recommended external execution subset", "value": min(600, external), "unit": "compositions"},
        {"section": "Catalog audit", "item": "Existing composed mutants with unmatched source edits", "value": int(audit["mutant_id"].nunique()) if not audit.empty else 0, "unit": "mutants"},
        {"section": "Catalog audit", "item": "Unmatched edit rows", "value": len(audit), "unit": "edits"},
        {"section": "Cost", "item": "Exhaustive theoretical executions", "value": len(manifest) * 30, "unit": "ProB test executions"},
        {"section": "Cost", "item": "600-composition external validation", "value": min(600, external) * 30, "unit": "ProB test executions"},
    ])
    for percentage in [1, 2, 4, 8, 16, 32, 50, 100]:
        compositions = max(1, round(development * percentage / 100))
        rows.append({"section": "AL budget", "item": f"{percentage}% of development pool", "value": compositions, "unit": "compositions"})
        rows.append({"section": "AL budget", "item": f"{percentage}% execution cost", "value": compositions * 30, "unit": "ProB test executions"})
    for label, compositions in [("5+5+5", 15), ("10+10+10", 30), ("30+30+30", 90)]:
        rows.append({"section": "Initial seed", "item": label, "value": compositions, "unit": "compositions"})
        rows.append({"section": "Initial seed", "item": label + " execution cost", "value": compositions * 30, "unit": "ProB test executions"})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutants", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--holdout_fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_excel(args.mutants, sheet_name="mutants")
    source, signature_map = source_catalog(frame)
    historical, audit = historical_compositions(frame, signature_map)
    manifest = build_manifest(source, historical, args.holdout_fraction, args.seed)
    manifest.to_csv(output / "full_candidate_pool_manifest.csv", index=False)
    audit.to_csv(output / "catalog_consistency_audit.csv", index=False)
    plan(manifest, audit).to_csv(output / "full_candidate_pool_plan.csv", index=False)


if __name__ == "__main__":
    main()
