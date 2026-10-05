from __future__ import annotations

import argparse
import itertools
import math
import random
import re
from pathlib import Path

import pandas as pd


def clean(value):
    return "" if pd.isna(value) else str(value).strip()


def edit_signature(row, index):
    target = clean(row.get(f"Edit{index}_Target"))
    original = clean(row.get(f"Edit{index}_Original"))
    mutated = clean(row.get(f"Edit{index}_Mutated"))
    return None if not target or not original or not mutated else (target, original, mutated)


def normalize(values):
    return "+".join(sorted({clean(value) for value in values if clean(value)}))


def tokens(*values):
    result = set()
    for value in values:
        result.update(token.lower() for token in re.split(r"[^A-Za-z0-9_]+", clean(value)) if token)
    return result


def jaccard_distance(left, right):
    union = left | right
    return 0.0 if not union else 1.0 - len(left & right) / len(union)


def diverse_extend(rows, chosen, target, seed):
    rng = random.Random(seed)
    selected = list(chosen)
    selected_ids = {row["item_id"] for row in selected}
    remaining = [row for row in rows if row["item_id"] not in selected_ids]
    if not selected and remaining:
        selected.append(remaining.pop(rng.randrange(len(remaining))))
    while remaining and len(selected) < target:
        selected_tokens = [row["tokens"] for row in selected]
        best_index = max(
            range(len(remaining)),
            key=lambda index: (
                min(jaccard_distance(remaining[index]["tokens"], value) for value in selected_tokens),
                -int(remaining[index]["item_id"][1:]),
            ),
        )
        selected.append(remaining.pop(best_index))
    return selected


def source_mutants(frame):
    records = []
    for row in frame[pd.to_numeric(frame["Fault_Count"], errors="coerce").eq(1)].to_dict("records"):
        signature = edit_signature(row, 1)
        if signature is None:
            continue
        record = {
            "source_mutant_id": clean(row["ID"]),
            "affected_components": clean(row["Fault_Components"]),
            "threat_types": clean(row["Threat_Family"]),
            "threat_domain": clean(row["Threat_Domain"]),
            "edit_target": signature[0],
            "edit_original": signature[1],
            "edit_mutated": signature[2],
        }
        record["tokens"] = tokens(*record.values())
        records.append(record)
    if len(records) != 30:
        raise ValueError(f"Expected 30 single-threat mutants, found {len(records)}")
    return records


def map_existing_systems(frame, signature_to_source):
    exact = {}
    audit = []
    for row in frame[pd.to_numeric(frame["Fault_Count"], errors="coerce").gt(1)].to_dict("records"):
        declared = int(row["Fault_Count"])
        mapped = []
        unmatched = []
        complete = 0
        for index in range(1, 4):
            signature = edit_signature(row, index)
            if signature is None:
                continue
            complete += 1
            if signature in signature_to_source:
                mapped.append(signature_to_source[signature])
            else:
                unmatched.append(" || ".join(signature))
        if complete < declared:
            status = "malformed"
        elif unmatched:
            status = "partial_mapping"
        elif len(mapped) == declared:
            status = "exact_mapping"
        else:
            status = "mapping_error"
        if status == "exact_mapping":
            exact[tuple(sorted(mapped))] = clean(row["ID"])
        audit.append({
            "system_id": clean(row["ID"]),
            "declared_threats": declared,
            "complete_edits_found": complete,
            "mapped_source_mutants": "+".join(sorted(mapped)),
            "unmatched_edits": len(unmatched),
            "unmatched_edit_details": " | ".join(unmatched),
            "mapping_status": status,
        })
    return exact, pd.DataFrame(audit)


def build_candidate(combo, lookup, exact):
    rows = [lookup[item] for item in combo]
    count = len(rows)
    return {
        "candidate_id": ("D_" if count == 2 else "T_") + "_".join(combo),
        "fault_count": count,
        "fault_order": "double" if count == 2 else "triple",
        "source_mutants": "+".join(combo),
        "affected_components": normalize(row["affected_components"] for row in rows),
        "threat_types": normalize(row["threat_types"] for row in rows),
        "component_threat_pairs": normalize(f'{row["affected_components"]}::{row["threat_types"]}' for row in rows),
        "threat_domains": normalize(row["threat_domain"] for row in rows),
        "edit_targets": "+".join(row["edit_target"] for row in rows),
        "edit_originals": " || ".join(row["edit_original"] for row in rows),
        "edit_mutated_values": " || ".join(row["edit_mutated"] for row in rows),
        "existing_system_id": exact.get(tuple(sorted(combo)), ""),
        "test_cases": 30,
    }


def assign_hidden_test(frame, fraction, seed):
    rng = random.Random(seed)
    frame = frame.copy()
    frame["candidate_set"] = "Available Candidate Set"
    target = round(len(frame) * fraction)
    double_target = round(435 * fraction)
    triple_target = target - double_target
    protected = frame["existing_system_id"].astype(str).ne("")
    for fault_count, count in [(2, double_target), (3, triple_target)]:
        candidates = frame[frame["fault_count"].eq(fault_count) & ~protected].index.tolist()
        rng.shuffle(candidates)
        frame.loc[candidates[:count], "candidate_set"] = "Hidden Test Set"
    if int(frame["candidate_set"].eq("Hidden Test Set").sum()) != target:
        raise RuntimeError("Hidden test set has the wrong size")
    return frame


def current_system_rows(frame, exact_candidate_by_system, audit_by_system):
    rows = []
    for row in frame.to_dict("records"):
        count = int(row["Fault_Count"])
        targets = [edit_signature(row, index)[0] for index in range(1, 4) if edit_signature(row, index)]
        item = {
            "item_id": clean(row["ID"]),
            "candidate_id": exact_candidate_by_system.get(clean(row["ID"]), ""),
            "fault_count": count,
            "fault_order": {1: "single", 2: "double", 3: "triple"}[count],
            "affected_components": clean(row["Fault_Components"]),
            "threat_types": clean(row["Threat_Family"]),
            "edit_targets": "+".join(targets),
            "mapping_status": "single_base_mutant" if count == 1 else audit_by_system[clean(row["ID"])]["mapping_status"],
        }
        item["tokens"] = tokens(item["affected_components"], item["threat_types"], item["edit_targets"], f'count_{count}')
        rows.append(item)
    return rows


def build_initial_sets(current_rows, seed):
    selected5 = []
    selected10 = []
    for count in [1, 2, 3]:
        group = [row for row in current_rows if row["fault_count"] == count]
        if count > 1:
            group = [row for row in group if row["mapping_status"] == "exact_mapping"]
        five = diverse_extend(group, [], 5, seed + count)
        ten = diverse_extend(group, five, 10, seed + 100 + count)
        selected5.extend(five)
        selected10.extend(ten)
    plans = [("5-5-5", selected5), ("10-10-10", selected10), ("30-30-30", current_rows)]
    output = []
    for plan, rows in plans:
        for order, row in enumerate(rows, start=1):
            output.append({
                "initial_set": plan,
                "system_id": row["item_id"],
                "candidate_id": row["candidate_id"],
                "fault_count": row["fault_count"],
                "fault_order": row["fault_order"],
                "affected_components": row["affected_components"],
                "threat_types": row["threat_types"],
                "edit_targets": row["edit_targets"],
                "selection_order": order,
                "mapping_status": row["mapping_status"],
            })
    return pd.DataFrame(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutants", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--hidden_fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    mutants = pd.read_excel(args.mutants)
    sources = source_mutants(mutants)
    source_lookup = {row["source_mutant_id"]: row for row in sources}
    signature_to_source = {
        edit_signature(row, 1): clean(row["ID"])
        for row in mutants[pd.to_numeric(mutants["Fault_Count"], errors="coerce").eq(1)].to_dict("records")
        if edit_signature(row, 1)
    }
    exact, audit = map_existing_systems(mutants, signature_to_source)
    candidates = [
        build_candidate(combo, source_lookup, exact)
        for size in [2, 3]
        for combo in itertools.combinations(sorted(source_lookup), size)
    ]
    candidate_frame = assign_hidden_test(pd.DataFrame(candidates), args.hidden_fraction, args.seed)
    exact_candidate_by_system = {
        row["existing_system_id"]: row["candidate_id"]
        for row in candidates
        if row["existing_system_id"]
    }
    audit_by_system = {row["system_id"]: row for row in audit.to_dict("records")}
    current_rows = current_system_rows(mutants, exact_candidate_by_system, audit_by_system)
    initial_sets = build_initial_sets(current_rows, args.seed)
    available_count = int(candidate_frame["candidate_set"].eq("Available Candidate Set").sum())
    budgets = []
    for plan, initial_count in [("5-5-5", 15), ("10-10-10", 30), ("30-30-30", 90)]:
        for percent in [0, 1, 2, 4, 8, 16]:
            added = 0 if percent == 0 else math.ceil(available_count * percent / 100)
            budgets.append({
                "initial_set": plan,
                "checkpoint": "Initial set only" if percent == 0 else f"{percent}%",
                "added_candidate_percent": percent,
                "initial_systems": initial_count,
                "added_compositions": added,
                "new_prob_runs": added * 30,
            })
    candidate_frame.to_csv(output / "candidate_combinations.csv", index=False)
    candidate_frame[candidate_frame["candidate_set"].eq("Hidden Test Set")].to_csv(output / "hidden_test_set.csv", index=False)
    candidate_frame[candidate_frame["candidate_set"].eq("Available Candidate Set")].to_csv(output / "available_candidate_set.csv", index=False)
    initial_sets.to_csv(output / "initial_sets.csv", index=False)
    pd.DataFrame(budgets).to_csv(output / "budget_checkpoints.csv", index=False)
    audit.to_csv(output / "catalog_audit.csv", index=False)


if __name__ == "__main__":
    main()
