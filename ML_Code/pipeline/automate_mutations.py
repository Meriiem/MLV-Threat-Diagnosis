#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import dataclasses
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

try:
    import pandas as pd
except ImportError:
    print("Missing dependency: pandas. Install with: pip install pandas openpyxl", file=sys.stderr)
    raise


DEFAULT_TIMEOUT_MS = 8000

VAR_COLUMNS = [
    "alarm_state", "alarm_type", "battery_level", "voltage_level", "pressure_reading", "temperature_reading",
    "controller_state", "vent_mode", "resp_rate", "insp_pressure", "ie_ratio", "is_ventilating",
    "gui_state", "is_alarm_displayed", "is_gui_frozen",
]

WILDCARD_EVENTS = {
    "Activate*Alarm": [
        "ActivateBatteryAlarm",
        "ActivateVoltageAlarm",
        "ActivatePressureAlarm",
        "ActivateTemperatureAlarm",
    ]
}

TEST_XLSX_MAPPING = {
    "tc_id_col": ["TC_ID", "TCID", "TestID"],
    "sequence_col": ["Sequence", "Seq", "Trace"],
}

MUTANT_XLSX_MAPPING = {
    "mutant_id_col": ["ID", "MutantID", "Mutant_Id"],
    "mutation_type_col": ["Mutation_Type", "MutationType", "Mutation"],
    "edits": [
        (["Edit1_Target", "Change1_Target"], ["Edit1_Original", "Change1_Original"], ["Edit1_Mutated", "Change1_Mutated"]),
        (["Edit2_Target", "Change2_Target"], ["Edit2_Original", "Change2_Original"], ["Edit2_Mutated", "Change2_Mutated"]),
        (["Edit3_Target", "Change3_Target"], ["Edit3_Original", "Change3_Original"], ["Edit3_Mutated", "Change3_Mutated"]),
    ],
}

_ILLEGAL_XLSX_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def clean_xlsx(v):
    if isinstance(v, str):
        return _ILLEGAL_XLSX_RE.sub("", v)
    return v


def _pick_col(df: "pd.DataFrame", candidates: List[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Missing required column for {label}. Tried: {candidates}. Found: {list(df.columns)}")




@dataclass
class TestCase:
    tc_id: str
    sequence: str  # "Ev1; Ev2; ..."


@dataclass
class Edit:
    location: str
    original: str
    mutated: str
    mutation_type: str = ""  # optional (from XLSX), helps heuristics


@dataclass
class Mutant:
    mutant_id: str
    edits: List[Edit]



def read_tests_xlsx(path: Path, mapping: Dict[str, Any]) -> List[TestCase]:
    df = pd.read_excel(path)

    tc_col = _pick_col(df, mapping["tc_id_col"], "testcase id")
    seq_col = _pick_col(df, mapping["sequence_col"], "sequence")

    tests: List[TestCase] = []
    for _, row in df.iterrows():
        tc = str(row.get(tc_col, "")).strip()
        seq = str(row.get(seq_col, "")).strip()
        if not tc or tc.lower() == "nan":
            continue
        if not seq or seq.lower() == "nan":
            continue
        tests.append(TestCase(tc_id=tc, sequence=seq))
    return tests


def read_mutants_xlsx(path: Path, mapping: Dict[str, Any]) -> List[Mutant]:
    df = pd.read_excel(path)

    mid_col = _pick_col(df, mapping["mutant_id_col"], "mutant id")

    mt_col = None
    try:
        mt_col = _pick_col(df, mapping.get("mutation_type_col", []), "mutation type")
    except Exception:
        mt_col = None

    mutants: List[Mutant] = []
    for _, row in df.iterrows():
        mid = str(row.get(mid_col, "")).strip()
        if not mid or mid.lower() == "nan":
            continue

        mutation_type = str(row.get(mt_col, "")).strip() if mt_col else ""

        edits: List[Edit] = []
        for loc_cols, orig_cols, mut_cols in mapping["edits"]:
            try:
                loc_col = _pick_col(df, loc_cols, "edit target")
                orig_col = _pick_col(df, orig_cols, "edit original")
                mut_col = _pick_col(df, mut_cols, "edit mutated")
            except ValueError:
                continue

            loc = str(row.get(loc_col, "")).strip()
            orig = str(row.get(orig_col, "")).strip()
            mut = str(row.get(mut_col, "")).strip()

            if loc and loc.lower() != "nan" and orig and orig.lower() != "nan" and mut and mut.lower() != "nan":
                edits.append(Edit(location=loc, original=orig, mutated=mut, mutation_type=mutation_type))

        # Keep mutant even if edits are missing; you said "do not discard mutants".
        mutants.append(Mutant(mutant_id=mid, edits=edits))

    return mutants



_LOC_EVENT_LABEL_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*/\s*@\s*(grd\d+|act\d+)\s*$")
_LOC_EVENT_ACTION_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*/\s*action\s*$", re.IGNORECASE)
_LOC_INV_RE = re.compile(r"inv(\d+)", re.IGNORECASE)

_ACT_IN_SNIPPET_RE = re.compile(r"@act(\d+)", re.IGNORECASE)


def normalize_eventb_expr(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("≔", ":=")
    s = s.replace("≤", "<=")
    s = s.replace("≥", ">=")
    s = s.replace("≠", "!=")
    s = s.replace("∧", "&&")
    s = s.replace("∨", "||")
    s = s.replace("⇔", "<->")
    s = s.replace("⇒", "->")
    s = re.sub(r"\s+", " ", s)
    return s


def detect_machine_name(eventb_text: str) -> str:
    m = re.search(r"rodinpos\('([^']+)'\s*,", eventb_text)
    if not m:
        raise ValueError("Cannot detect machine name (no rodinpos('...') found) in .eventb")
    return m.group(1)


def parse_location(loc: str, original: str, mutated: str) -> Tuple[str, str]:
    """
    Returns:
      - ("invariant", "inv16")
      - ("EventName", "grd2") / ("EventName", "act1")
    """
    loc = (loc or "").strip()

    # invariant formats: inv16, @inv16, Invariant/inv16, etc.
    inv_m = _LOC_INV_RE.search(loc)
    if inv_m and ("inv" in loc.lower()):
        return ("invariant", f"inv{inv_m.group(1)}")

    m = _LOC_EVENT_LABEL_RE.match(loc)
    if m:
        return (m.group(1), m.group(2))

    m2 = _LOC_EVENT_ACTION_RE.match(loc)
    if m2:
        ev = m2.group(1)
        s = normalize_eventb_expr(original) + " " + normalize_eventb_expr(mutated)
        am = _ACT_IN_SNIPPET_RE.search(s)
        if not am:
            raise ValueError(f"Location '{loc}' has '/action' but no @actK in snippet to infer which action to patch.")
        return (ev, f"act{am.group(1)}")

    m3 = re.match(r"^\s*([A-Za-z0-9_]+)\s*/\s*@\s*(grd\d+|act\d+)", loc)
    if m3:
        return (m3.group(1), m3.group(2))

    # bare label like "inv16"
    if re.fullmatch(r"@?inv\d+", loc, flags=re.IGNORECASE):
        nm = re.search(r"inv(\d+)", loc, flags=re.IGNORECASE)
        return ("invariant", f"inv{nm.group(1)}")

    raise ValueError(f"Unrecognized location format: '{loc}'")


def find_event_segment(text: str, event_name: str, machine_name: str) -> Tuple[int, int]:
    marker1 = f"event(rodinpos('{machine_name}','{event_name}'"
    start = text.find(marker1)
    if start == -1:
        marker2 = f"event(rodinpos('{machine_name}','{event_name}','"
        start = text.find(marker2)
    if start == -1:
        m = re.search(rf"event\(rodinpos\('{re.escape(machine_name)}','{re.escape(event_name)}'[^)]*\)", text)
        if not m:
            raise ValueError(f"Event '{event_name}' not found in .eventb for machine '{machine_name}'")
        start = m.start()

    next_marker = f"event(rodinpos('{machine_name}','"
    end = text.find(next_marker, start + 10)
    if end == -1:
        end = len(text)
    return start, end


def _find_label_pos(segment: str, label: str, machine_name: str) -> int:
    patterns = [
        f"rodinpos('{machine_name}',{label},",
        f"rodinpos('{machine_name}','{label}',",
    ]
    for pat in patterns:
        pos = segment.find(pat)
        if pos != -1:
            return pos
    return -1


def extract_labeled_term(segment: str, label: str, machine_name: str) -> Tuple[int, int, str]:
    pos = _find_label_pos(segment, label, machine_name)
    if pos == -1:
        raise ValueError(f"Label '{label}' not found in segment (machine='{machine_name}')")

    paren = segment.rfind("(", 0, pos)
    if paren == -1:
        raise ValueError("Malformed segment: no '(' before label")

    i = paren - 1
    while i >= 0 and (segment[i].isalnum() or segment[i] == "_"):
        i -= 1
    functor_start = i + 1

    depth = 0
    j = paren
    while j < len(segment):
        ch = segment[j]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                term_end = j + 1
                return functor_start, term_end, segment[functor_start:term_end]
        j += 1

    raise ValueError("Malformed segment: unbalanced parentheses while extracting term")


def replace_functor_name(term: str, new_functor: str) -> str:
    k = term.find("(")
    if k == -1:
        raise ValueError("Bad term (no '('): cannot replace functor")
    return new_functor + term[k:]


def prolog_identifier_candidates(token: str) -> List[str]:
    t = (token or "").strip()
    out: List[str] = []

    if t.upper() == "TRUE":
        return ["boolean_true(none)"]
    if t.upper() == "FALSE":
        return ["boolean_false(none)"]

    if re.fullmatch(r"-?\d+", t):
        return [f"integer(none,{t})", t]

    t_plain = t.strip("'\"")
    out.append(f"identifier(none,{t_plain})")
    out.append(f"identifier(none,'{t_plain}')")
    out.append(f"identifier(none,\"{t_plain}\")")
    return out


def to_prolog_identifier(token: str) -> str:
    t = (token or "").strip()
    if t.upper() == "TRUE":
        return "boolean_true(none)"
    if t.upper() == "FALSE":
        return "boolean_false(none)"
    if re.fullmatch(r"-?\d+", t):
        return f"integer(none,{t})"
    if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
        t2 = t.strip("'\"")
        return f"identifier(none,'{t2}')"
    if re.match(r"^[a-z_][a-z0-9_]*$", t):
        return f"identifier(none,{t})"
    return f"identifier(none,'{t}')"


def _top_level_args(term: str) -> List[str]:
    lp = term.find("(")
    rp = term.rfind(")")
    if lp == -1 or rp == -1 or rp <= lp:
        return []
    body = term[lp + 1:rp]

    args: List[str] = []
    depth = 0
    cur = []
    for ch in body:
        if ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            args.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        args.append("".join(cur).strip())
    return args


def _extract_balanced_from(text: str, start_idx: int) -> Tuple[int, int, str]:
    lp = text.find("(", start_idx)
    if lp == -1:
        raise ValueError("No '(' after functor name")
    depth = 0
    j = lp
    while j < len(text):
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0:
                return start_idx, j + 1, text[start_idx:j + 1]
        j += 1
    raise ValueError("Unbalanced parentheses while extracting subterm")


def _find_first_subterm(term: str, functor: str, predicate: Callable[[str], bool]) -> Optional[Tuple[int, int, str]]:
    needle = functor + "("
    idx = 0
    while True:
        i = term.find(needle, idx)
        if i == -1:
            return None
        s, e, sub = _extract_balanced_from(term, i)
        if predicate(sub):
            return (s, e, sub)
        idx = e


def _parse_set_literal(s: str) -> List[str]:
    m = re.search(r"\{([^}]+)\}", s)
    if not m:
        return []
    inside = m.group(1)
    toks = []
    for part in inside.split(","):
        t = part.strip()
        if t:
            toks.append(t)
    return toks


def _parse_membership_expr(s: str) -> Optional[Tuple[str, List[str], bool]]:
    if not s:
        return None
    mm = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*([∈∉])\s*\{", s)
    if not mm:
        return None
    lhs = mm.group(1).strip()
    sym = mm.group(2)
    elems = _parse_set_literal(s)
    neg = (sym == "∉")
    return (lhs, elems, neg)


def _parse_equality_expr(s: str) -> Optional[Tuple[str, str]]:
    s2 = normalize_eventb_expr(s)
    if ":=" in s2:
        return None
    m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z0-9_'\-]+)", s2)
    if not m:
        return None
    return (m.group(1).strip(), m.group(2).strip())


def detect_simple_mutation(original: str, mutated: str) -> Dict[str, str]:
    o = normalize_eventb_expr(original)
    m = normalize_eventb_expr(mutated)

    if "delete" in m.lower() or "omit" in m.lower() or "/* delete" in m.lower():
        return {"kind": "delete"}

    if "<->" in o and "->" in m and "<->" not in m:
        return {"kind": "iff_to_impl"}

    if "||" in o and "&&" in m:
        return {"kind": "or_to_and"}
    if "&&" in o and "||" in m:
        return {"kind": "and_to_or"}

    if ("∈" in original or "∉" in original) and ("∈" in mutated or "∉" in mutated):
        return {"kind": "membership_set_change"}
    if ("∈" in original or "∉" in original) and ("=" in m) and ("∈" not in mutated and "∉" not in mutated):
        return {"kind": "membership_to_equality"}

    if ("=" in o) and ("=" in m) and (":=" not in o) and (":=" not in m) and (o != m):
        return {"kind": "equality_rhs_change"}

    ro = re.search(r"(<=|>=|!=|=|<|>)", o)
    rm = re.search(r"(<=|>=|!=|=|<|>)", m)
    if ro and rm and ro.group(1) != rm.group(1):
        return {"kind": "relop_change", "from": ro.group(1), "to": rm.group(1)}

    if ":=" in o and ":=" in m:
        return {"kind": "action_value"}

    return {"kind": "unknown", "o": o, "m": m}


def replace_last_arg(term: str, new_last: str) -> str:
    functor = term[:term.find("(")]
    args = _top_level_args(term)
    if not args:
        raise ValueError("Bad term for replace_last_arg")
    args[-1] = new_last
    return f"{functor}(" + ",".join(args) + ")"


def patch_relop_in_term(term: str, original: str, mutated: str) -> str:
    o = normalize_eventb_expr(original)
    m = normalize_eventb_expr(mutated)

    ro = re.search(r"(<=|>=|!=|=|<|>)", o)
    rm = re.search(r"(<=|>=|!=|=|<|>)", m)
    if not ro or not rm:
        raise ValueError(f"Cannot detect relop change in: '{original}' -> '{mutated}'")

    op_from = ro.group(1)
    op_to = rm.group(1)
    if op_from == op_to:
        return term

    rel_map = {
        "<=": "less_equal",
        "<": "less",
        ">": "greater",
        ">=": "greater_equal",
        "=": "equal",
        "!=": "not_equal",
    }
    f_from = rel_map.get(op_from)
    f_to = rel_map.get(op_to)
    if not f_from or not f_to:
        raise ValueError(f"Unsupported relop mapping: {op_from} -> {op_to}")

    if (f_from + "(") in term:
        return term.replace(f_from + "(", f_to + "(", 1)

    raise ValueError("Could not apply relop patch (no target functor found)")


def patch_or_to_and_in_term(term: str) -> str:
    if term.startswith("disjunct("):
        return replace_functor_name(term, "conjunct")
    return term.replace("disjunct(", "conjunct(", 1)


def patch_and_to_or_in_term(term: str) -> str:
    if term.startswith("conjunct("):
        return replace_functor_name(term, "disjunct")
    return term.replace("conjunct(", "disjunct(", 1)


def patch_iff_to_impl_in_term(term: str) -> str:
    if term.startswith("equivalence(") or term.startswith("equiv("):
        return replace_functor_name(term, "implication")
    if "equivalence(" in term:
        return term.replace("equivalence(", "implication(", 1)
    if "equiv(" in term:
        return term.replace("equiv(", "implication(", 1)
    return term


def _find_first_match_any(term: str, needles: List[str]) -> Optional[str]:
    for n in needles:
        if n and n in term:
            return n
    return None


def patch_action_value_in_assign_term(assign_term: str, original: str, mutated: str) -> str:
    o = normalize_eventb_expr(original)
    m = normalize_eventb_expr(mutated)

    o2 = re.sub(r"^@act\d+\s+", "", o).strip()
    m2 = re.sub(r"^@act\d+\s+", "", m).strip()

    mo = re.search(r":=\s*([A-Za-z0-9_'\-]+)", o2)
    mm = re.search(r":=\s*([A-Za-z0-9_'\-]+)", m2)
    if not mo or not mm:
        raise ValueError(f"Cannot parse action value: '{original}' -> '{mutated}'")

    tok_from = mo.group(1)
    tok_to = mm.group(1)

    from_cands = prolog_identifier_candidates(tok_from)
    to_primary = to_prolog_identifier(tok_to)
    to_cands = [to_primary] + [c for c in prolog_identifier_candidates(tok_to) if c != to_primary]

    hit = _find_first_match_any(assign_term, from_cands)
    if hit:
        repl = to_primary
        if hit in to_cands:
            repl = hit
        return assign_term.replace(hit, repl, 1)

    return assign_term


def _extract_balanced_brackets(s: str, start: int) -> Tuple[int, int, str]:
    if start < 0 or start >= len(s) or s[start] != "[":
        raise ValueError("Bracket extraction must start at '['")
    depth = 0
    i = start
    while i < len(s):
        if s[i] == "[":
            depth += 1
        elif s[i] == "]":
            depth -= 1
            if depth == 0:
                return start, i + 1, s[start:i + 1]
        i += 1
    raise ValueError("Unbalanced brackets while extracting list")


def patch_action_omission_in_assign_term(assign_term: str) -> str:
    lb1 = assign_term.find("[")
    if lb1 == -1:
        raise ValueError("Cannot find first '[' in assignment term")
    s1, e1, list1 = _extract_balanced_brackets(assign_term, lb1)

    lb2 = assign_term.find("[", e1)
    if lb2 == -1:
        raise ValueError("Cannot find second '[' in assignment term")
    s2, e2, _list2 = _extract_balanced_brackets(assign_term, lb2)

    return assign_term[:s2] + list1 + assign_term[e2:]


def _replace_set_extension_list(member_subterm: str, new_elems: List[str]) -> str:
    k = member_subterm.find("set_extension(")
    if k == -1:
        raise ValueError("No set_extension(...) in membership subterm")
    lb = member_subterm.find("[", k)
    if lb == -1:
        raise ValueError("No '[' in set_extension(...)")
    depth = 0
    j = lb
    while j < len(member_subterm):
        if member_subterm[j] == "[":
            depth += 1
        elif member_subterm[j] == "]":
            depth -= 1
            if depth == 0:
                rb = j
                new_list = ",".join(to_prolog_identifier(x) for x in new_elems)
                return member_subterm[:lb + 1] + new_list + member_subterm[rb:]
        j += 1
    raise ValueError("Unbalanced set_extension list brackets")


def _patch_membership_set_change(any_term: str, original: str, mutated: str) -> str:
    parsed_o = _parse_membership_expr(original)
    parsed_m = _parse_membership_expr(mutated)
    if not parsed_o or not parsed_m:
        raise ValueError("Cannot parse membership expressions for set change")
    lhs, _old_elems, _neg_o = parsed_o
    _lhs2, new_elems, _neg_m = parsed_m

    lhs_id = to_prolog_identifier(lhs)

    found = _find_first_subterm(
        any_term,
        "member",
        lambda sub: (lhs_id in sub) and ("set_extension(" in sub),
    )
    if not found:
        raise ValueError(f"Could not find member(...) subterm containing {lhs_id}")

    s, e, sub = found
    patched_sub = _replace_set_extension_list(sub, new_elems)
    return any_term[:s] + patched_sub + any_term[e:]


def _patch_membership_to_equality(any_term: str, original: str, mutated: str) -> str:
    parsed_o = _parse_membership_expr(original)
    parsed_eq = _parse_equality_expr(mutated)
    if not parsed_o or not parsed_eq:
        raise ValueError("Cannot parse membership->equality expressions")

    lhs, _elems, _neg = parsed_o
    _lhs2, rhs = parsed_eq

    lhs_id = to_prolog_identifier(lhs)
    rhs_id = to_prolog_identifier(rhs)

    found = _find_first_subterm(
        any_term,
        "member",
        lambda sub: (lhs_id in sub) and ("set_extension(" in sub or "natural_set(" in sub),
    )
    if not found:
        raise ValueError(f"Could not find member(...) subterm for {lhs}")

    s, e, sub = found
    args = _top_level_args(sub)
    if len(args) < 3:
        raise ValueError("Malformed member(...) term (cannot split args)")

    a1 = args[0]
    a2 = args[1]
    new_sub = f"equal({a1},{a2},{rhs_id})"
    return any_term[:s] + new_sub + any_term[e:]


def _patch_equality_rhs_change(any_term: str, original: str, mutated: str) -> str:
    o = _parse_equality_expr(original)
    m = _parse_equality_expr(mutated)
    if not o or not m:
        raise ValueError("Cannot parse equality RHS change")
    lhs, rhs_from = o
    _lhs2, rhs_to = m

    lhs_id = to_prolog_identifier(lhs)
    rhs_from_id = to_prolog_identifier(rhs_from)
    rhs_to_id = to_prolog_identifier(rhs_to)

    found = _find_first_subterm(
        any_term,
        "equal",
        lambda sub: (lhs_id in sub) and (rhs_from_id in sub),
    )
    if found:
        s, e, sub = found
        return any_term[:s] + sub.replace(rhs_from_id, rhs_to_id, 1) + any_term[e:]

    found2 = _find_first_subterm(any_term, "equal", lambda sub: (lhs_id in sub))
    if found2:
        s, e, sub = found2
        args = _top_level_args(sub)
        if len(args) >= 3:
            args[-1] = rhs_to_id
            new_sub = "equal(" + ",".join(args) + ")"
            return any_term[:s] + new_sub + any_term[e:]

    raise ValueError(f"Could not find equal(...) subterm for {lhs}")


def apply_edit_to_eventb_prolog(full_text: str, edit: Edit) -> str:
    machine_name = detect_machine_name(full_text)

    if "Activate*Alarm" in edit.location:
        out = full_text
        for ev in WILDCARD_EVENTS.get("Activate*Alarm", []):
            concrete = dataclasses.replace(edit, location=edit.location.replace("Activate*Alarm", ev))
            try:
                out = apply_edit_to_eventb_prolog(out, concrete)
            except Exception:
                pass
        return out

    target, label = parse_location(edit.location, edit.original, edit.mutated)
    kind = detect_simple_mutation(edit.original, edit.mutated)["kind"]

    if target == "invariant":
        start, end, term = extract_labeled_term(full_text, label, machine_name)

        if kind == "delete":
            new_term = replace_last_arg(term, "boolean_true(none)")
            return full_text[:start] + new_term + full_text[end:]

        if kind == "iff_to_impl":
            new_term = patch_iff_to_impl_in_term(term)
        elif kind == "or_to_and":
            new_term = patch_or_to_and_in_term(term)
        elif kind == "and_to_or":
            new_term = patch_and_to_or_in_term(term)
        elif kind == "membership_set_change":
            new_term = _patch_membership_set_change(term, edit.original, edit.mutated)
        elif kind == "membership_to_equality":
            new_term = _patch_membership_to_equality(term, edit.original, edit.mutated)
        elif kind == "equality_rhs_change":
            new_term = _patch_equality_rhs_change(term, edit.original, edit.mutated)
        elif kind == "relop_change":
            new_term = patch_relop_in_term(term, edit.original, edit.mutated)
        else:
            new_term = term

        return full_text[:start] + new_term + full_text[end:]

    event_name = target
    ev_s, ev_e = find_event_segment(full_text, event_name, machine_name)
    ev_seg = full_text[ev_s:ev_e]

    lt_s, lt_e, term = extract_labeled_term(ev_seg, label, machine_name)

    if label.startswith("grd"):
        if kind == "delete":
            new_term = replace_last_arg(term, "boolean_true(none)")
        elif kind == "or_to_and":
            new_term = patch_or_to_and_in_term(term)
        elif kind == "and_to_or":
            new_term = patch_and_to_or_in_term(term)
        elif kind == "membership_set_change":
            new_term = _patch_membership_set_change(term, edit.original, edit.mutated)
        elif kind == "membership_to_equality":
            new_term = _patch_membership_to_equality(term, edit.original, edit.mutated)
        elif kind == "equality_rhs_change":
            new_term = _patch_equality_rhs_change(term, edit.original, edit.mutated)
        elif kind == "relop_change":
            new_term = patch_relop_in_term(term, edit.original, edit.mutated)
        else:
            new_term = term

        patched_ev = ev_seg[:lt_s] + new_term + ev_seg[lt_e:]
        return full_text[:ev_s] + patched_ev + full_text[ev_e:]

    if label.startswith("act"):
        if kind == "delete":
            new_term = patch_action_omission_in_assign_term(term)
        elif kind == "action_value":
            new_term = patch_action_value_in_assign_term(term, edit.original, edit.mutated)
        else:
            try:
                new_term = patch_action_value_in_assign_term(term, edit.original, edit.mutated)
            except Exception:
                new_term = term

        patched_ev = ev_seg[:lt_s] + new_term + ev_seg[lt_e:]
        return full_text[:ev_s] + patched_ev + full_text[ev_e:]

    raise ValueError(f"Unhandled label type in location: {edit.location}")

def run_cmd(args: List[str], cwd: Optional[Path] = None, timeout_s: Optional[float] = None) -> Tuple[int, str, str, bool]:
    p = subprocess.Popen(
        args,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
    )
    try:
        out, err = p.communicate(timeout=timeout_s)
        return p.returncode, out, err, False
    except subprocess.TimeoutExpired:
        try:
            p.kill()
        except Exception:
            pass
        out, err = p.communicate()
        return 124, out, err, True


def _write_placeholder(path: Path, title: str, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = payload if isinstance(payload, str) else repr(payload)
    safe = safe.replace("\r\n", "\n")
    path.write_text(f"{title}\n{safe}\n", encoding="utf-8", errors="replace")


def run_testcase(probcli: Path, model_eventb: Path, tc: TestCase, out_dir: Path,
                 timeout_ms: int, clpfd: bool) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    trace_path = (out_dir / f"{tc.tc_id}.trace").resolve()
    states_path = (out_dir / f"{tc.tc_id}_states.txt").resolve()

    seq = tc.sequence.strip()
    model_cwd = model_eventb.parent.resolve()
    model_arg = model_eventb.name

    retry_timeouts = [timeout_ms, max(timeout_ms * 3, timeout_ms + 5000), max(timeout_ms * 10, timeout_ms + 20000)]

    last = {
        "rc_trace": 999, "rc_replay": 999, "ok": False,
        "stdout_trace": "", "stderr_trace": "", "stdout_replay": "", "stderr_replay": "",
        "timed_out_trace": False, "timed_out_replay": False,
    }

    for attempt, tmo in enumerate(retry_timeouts, start=1):
        cmd1 = [
            str(probcli),
            model_arg,
            "-init",
            "-p", "CLPFD", "TRUE" if clpfd else "FALSE",
            "-p", "TIME_OUT", str(tmo),
            "-cbc_sequence", seq,
            "-his", str(trace_path),
            "-his_option", "trace_file",
        ]
        rc1, out1, err1, to1 = run_cmd(cmd1, cwd=model_cwd, timeout_s=(tmo / 1000.0) + 10.0)

        load_failed = ("error occurred" in (err1 or "").lower()) or ("error occurred" in (out1 or "").lower())
        ok1 = (not load_failed) and trace_path.exists() and trace_path.stat().st_size > 0

        rc2, out2, err2, to2 = (999, "", "", False)
        ok2 = False
        if ok1:
            cmd2 = [
                str(probcli),
                model_arg,
                "-trace_replay", "prolog",
                str(trace_path),
                "-his", str(states_path),
                "-his_option", "show_states",
            ]
            rc2, out2, err2, to2 = run_cmd(cmd2, cwd=model_cwd, timeout_s=(tmo / 1000.0) + 10.0)
            load_failed2 = ("error occurred" in (err2 or "").lower()) or ("error occurred" in (out2 or "").lower())
            ok2 = (not load_failed2) and states_path.exists() and states_path.stat().st_size > 0

        last = {
            "rc_trace": rc1, "rc_replay": rc2,
            "ok": bool(ok1 and ok2),
            "stdout_trace": (out1 or ""),
            "stderr_trace": (err1 or ""),
            "stdout_replay": (out2 or ""),
            "stderr_replay": (err2 or ""),
            "timed_out_trace": bool(to1),
            "timed_out_replay": bool(to2),
            "attempt": attempt,
            "timeout_used_ms": tmo,
        }

        if last["ok"]:
            break

    if not trace_path.exists() or trace_path.stat().st_size == 0:
        _write_placeholder(trace_path, "NO_TRACE", f"tc={tc.tc_id}\nsequence={tc.sequence}\n{last['stderr_trace']}\n{last['stdout_trace']}")
    if not states_path.exists() or states_path.stat().st_size == 0:
        _write_placeholder(states_path, "NO_STATES", f"tc={tc.tc_id}\nsequence={tc.sequence}\n{last['stderr_replay']}\n{last['stdout_replay']}")

    def tr(s: str, n: int = 6000) -> str:
        s = s or ""
        if len(s) <= n:
            return s
        return s[:n] + "\n...[truncated]..."

    return {
        "tc_id": tc.tc_id,
        "sequence": tc.sequence,
        "trace_path": str(trace_path),
        "states_path": str(states_path),
        "rc_trace": last["rc_trace"],
        "rc_replay": last["rc_replay"],
        "ok": bool(last["ok"]),
        "attempt": int(last.get("attempt", 1)),
        "timeout_used_ms": int(last.get("timeout_used_ms", timeout_ms)),
        "timed_out_trace": bool(last.get("timed_out_trace", False)),
        "timed_out_replay": bool(last.get("timed_out_replay", False)),
        "stderr_trace": tr(last["stderr_trace"]),
        "stderr_replay": tr(last["stderr_replay"]),
        "stdout_trace": tr(last["stdout_trace"]),
        "stdout_replay": tr(last["stdout_replay"]),
    }



_STATE_SPLIT_RE = re.compile(r"^\s*State\s+(\d+)\s*[:\-]?\s*$", re.IGNORECASE | re.MULTILINE)
_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$")


def parse_show_states(states_txt: Path, sequence: str) -> List[Dict[str, Any]]:
    raw = states_txt.read_text(encoding="utf-8", errors="replace")

    matches = list(_STATE_SPLIT_RE.finditer(raw))
    chunks: List[Tuple[int, str]] = []

    if matches:
        for i, mm in enumerate(matches):
            st_idx = int(mm.group(1))
            start = mm.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
            chunks.append((st_idx, raw[start:end]))
    else:
        chunks = [(0, raw)]

    events = [x.strip() for x in sequence.split(";") if x.strip()]

    rows: List[Dict[str, Any]] = []
    for (st_idx, body) in chunks:
        snap: Dict[str, Any] = {k: "" for k in VAR_COLUMNS}
        for line in body.splitlines():
            mm = _ASSIGN_RE.match(line)
            if not mm:
                continue
            var = mm.group(1).strip()
            val = mm.group(2).strip()
            if var in snap:
                snap[var] = val

        step = st_idx
        ev = "" if step == 0 else (events[step - 1] if (step - 1) < len(events) else "")
        row = {"step": step, "event": ev}
        row.update(snap)
        rows.append(row)

    rows.sort(key=lambda r: int(r.get("step", 0)))
    return rows



def copy_baseline_export(baseline_dir: Path, dest_dir: Path) -> None:
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(baseline_dir, dest_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probcli", required=True, help="Path to probcli executable (or probcli.exe)")
    ap.add_argument("--baseline_export_dir", required=True, help="Folder containing baseline ProB export (includes *.eventb)")
    ap.add_argument("--baseline_eventb_name", default="Ventilator_Composed_mch.eventb",
                    help="Name of the .eventb file inside baseline_export_dir")
    ap.add_argument("--tests_xlsx", required=True, help="XLSX containing the 30 test cases (TC_ID, Sequence)")
    ap.add_argument("--mutants_xlsx", required=True, help="XLSX containing mutants (your catalog)")
    ap.add_argument("--out_dir", required=True, help="Output folder")
    ap.add_argument("--timeout_ms", type=int, default=DEFAULT_TIMEOUT_MS)
    ap.add_argument("--no_clpfd", action="store_true")
    ap.add_argument("--max_mutants", type=int, default=0, help="If >0, run only first N mutants")
    ap.add_argument("--include_baseline", action="store_true", help="Also run BASE (unmutated) as mutant_id=BASE")
    args = ap.parse_args()

    probcli = Path(args.probcli).resolve()
    baseline_export_dir = Path(args.baseline_export_dir).resolve()
    baseline_eventb = baseline_export_dir / args.baseline_eventb_name
    tests_xlsx = Path(args.tests_xlsx).resolve()
    mutants_xlsx = Path(args.mutants_xlsx).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not probcli.exists():
        raise FileNotFoundError(f"probcli not found: {probcli}")
    if not baseline_export_dir.exists():
        raise FileNotFoundError(f"baseline_export_dir not found: {baseline_export_dir}")
    if not baseline_eventb.exists():
        raise FileNotFoundError(f"baseline .eventb not found: {baseline_eventb}")
    if not tests_xlsx.exists():
        raise FileNotFoundError(f"tests_xlsx not found: {tests_xlsx}")
    if not mutants_xlsx.exists():
        raise FileNotFoundError(f"mutants_xlsx not found: {mutants_xlsx}")

    clpfd = not args.no_clpfd

    tests = read_tests_xlsx(tests_xlsx, TEST_XLSX_MAPPING)
    mutants = read_mutants_xlsx(mutants_xlsx, MUTANT_XLSX_MAPPING)
    if args.max_mutants and args.max_mutants > 0:
        mutants = mutants[:args.max_mutants]

    all_runs: List[Dict[str, Any]] = []
    all_states: List[Dict[str, Any]] = []
    mutate_log: List[Dict[str, Any]] = []

    worklist: List[Mutant] = []
    if args.include_baseline:
        worklist.append(Mutant(mutant_id="BASE", edits=[]))
    worklist.extend(mutants)

    for mi, mutant in enumerate(worklist, start=1):
        print(f"[{mi}/{len(worklist)}] Mutant: {mutant.mutant_id}")

        mutant_root = out_dir / mutant.mutant_id
        model_dir = mutant_root / "model"
        runs_dir = mutant_root / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        copy_baseline_export(baseline_export_dir, model_dir)
        model_eventb = model_dir / args.baseline_eventb_name

        applied = 0
        failed = 0
        if mutant.edits:
            text = model_eventb.read_text(encoding="utf-8", errors="replace")
            for ei, edit in enumerate(mutant.edits, start=1):
                try:
                    new_text = apply_edit_to_eventb_prolog(text, edit)
                    if new_text != text:
                        applied += 1
                    text = new_text
                    mutate_log.append({
                        "mutant_id": mutant.mutant_id,
                        "edit_index": ei,
                        "location": edit.location,
                        "status": "ok",
                        "error": "",
                    })
                except Exception as e:
                    failed += 1
                    mutate_log.append({
                        "mutant_id": mutant.mutant_id,
                        "edit_index": ei,
                        "location": edit.location,
                        "status": "fail",
                        "error": f"{e}",
                    })
                    print(f"  [MUTATE-FAIL] {mutant.mutant_id} edit#{ei}: {edit.location} | {e}", file=sys.stderr)
            model_eventb.write_text(text, encoding="utf-8")

        for tc in tests:
            tc_out = runs_dir / tc.tc_id
            res = run_testcase(
                probcli=probcli,
                model_eventb=model_eventb,
                tc=tc,
                out_dir=tc_out,
                timeout_ms=args.timeout_ms,
                clpfd=clpfd,
            )
            res_row = {"mutant_id": mutant.mutant_id, "mutate_applied_edits": applied, "mutate_failed_edits": failed}
            res_row.update(res)
            all_runs.append(res_row)

            try:
                state_rows = parse_show_states(Path(res["states_path"]), tc.sequence)
                for r in state_rows:
                    all_states.append({"mutant_id": mutant.mutant_id, "tc_id": tc.tc_id, **r})
            except Exception as e:
                all_runs.append({
                    "mutant_id": mutant.mutant_id,
                    "tc_id": tc.tc_id,
                    "ok": False,
                    "stage": "parse_states",
                    "error": str(e),
                    "sequence": tc.sequence,
                    "trace_path": res["trace_path"],
                    "states_path": res["states_path"],
                    "rc_trace": res["rc_trace"],
                    "rc_replay": res["rc_replay"],
                })
                print(f"  [PARSE-FAIL] {mutant.mutant_id}/{tc.tc_id}: {e}", file=sys.stderr)

    runs_df = pd.DataFrame(all_runs)
    states_df = pd.DataFrame(all_states)
    mutate_df = pd.DataFrame(mutate_log)

    for df in (runs_df, states_df, mutate_df):
        for c in df.columns:
            if df[c].dtype == "object":
                df[c] = df[c].map(clean_xlsx)

    dataset_xlsx = out_dir / "dataset.xlsx"
    with pd.ExcelWriter(dataset_xlsx, engine="openpyxl") as w:
        runs_df.to_excel(w, index=False, sheet_name="runs")
        states_df.to_excel(w, index=False, sheet_name="states")
        mutate_df.to_excel(w, index=False, sheet_name="mutate_log")

    print(f"[DONE] Wrote: {dataset_xlsx}")


if __name__ == "__main__":
    main()