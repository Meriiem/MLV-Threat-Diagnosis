#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler

# ----------------------------
# CONFIG
# ----------------------------

DEFAULT_TIMEOUT_MS = 8000

DEFAULT_VAR_COLUMNS = [
    "alarm_state", "alarm_type", "battery_level", "voltage_level", "pressure_reading", "temperature_reading",
    "controller_state", "vent_mode", "resp_rate", "insp_pressure", "ie_ratio", "is_ventilating",
    "gui_state", "is_alarm_displayed", "is_gui_frozen",
]

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

WILDCARD_EVENTS = {
    "Activate*Alarm": [
        "ActivateBatteryAlarm",
        "ActivateVoltageAlarm",
        "ActivatePressureAlarm",
        "ActivateTemperatureAlarm",
    ]
}

AL_STRATEGIES = [
    "random",
    "uncertainty_entropy",
    "uncertainty_margin",
    "diversity_jaccard",
]

_ILLEGAL_XLSX_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

def clean_xlsx(v: Any) -> Any:
    if isinstance(v, str):
        return _ILLEGAL_XLSX_RE.sub("", v)
    return v


def _pick_col(df: pd.DataFrame, candidates: List[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Missing required column for {label}. Tried: {candidates}. Found: {list(df.columns)}")


# ----------------------------
# DATA STRUCTURES
# ----------------------------

@dataclass
class TestCase:
    tc_id: str
    sequence: str

@dataclass
class Edit:
    location: str
    original: str
    mutated: str
    mutation_type: str = ""

@dataclass
class Mutant:
    mutant_id: str
    edits: List[Edit]

@dataclass
class ProposedMutant:
    mutant_id: str
    edits: List[Edit]
    fault_count: int
    fault_components: str
    mutation_type: str = "ACTIVE_LEARNING"
    source_edit_ids: str = ""


# ----------------------------
# XLSX READERS
# ----------------------------

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


def read_mutants_xlsx(path: Path, mapping: Dict[str, Any]) -> Tuple[List[Mutant], pd.DataFrame]:
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

        mutants.append(Mutant(mutant_id=mid, edits=edits))

    return mutants, df


# ----------------------------
# COMPONENT HEURISTICS
# ----------------------------

GUI_HINTS = ("GUI", "Navigate", "Select", "Freeze", "Unfreeze", "Display", "Screen")
ALRM_HINTS = ("Alarm", "Voltage", "Pressure", "Temperature", "Battery", "ALRM")
CTRL_HINTS = ("RespRate", "InspPressure", "IERatio", "Ventilation", "PCV", "Controller", "CTRL")

def infer_component_from_location(loc: str) -> str:
    s = (loc or "").strip()
    ev = s.split("/")[0].strip() if "/" in s else s
    ev_upper = ev.upper()
    if any(h.upper() in ev_upper for h in GUI_HINTS):
        return "GUI"
    if any(h.upper() in ev_upper for h in ALRM_HINTS):
        return "ALRM"
    if any(h.upper() in ev_upper for h in CTRL_HINTS):
        return "CTRL"
    s_upper = s.upper()
    if "ALARM" in s_upper:
        return "ALRM"
    if "GUI" in s_upper:
        return "GUI"
    if "CTRL" in s_upper or "CONTROLLER" in s_upper:
        return "CTRL"
    return "UNKNOWN"

def normalize_combo(comps: List[str]) -> str:
    comps2 = [c for c in comps if c and c != "UNKNOWN"]
    if not comps2:
        return "UNKNOWN"
    comps2 = sorted(comps2)
    return "+".join(comps2)

def fault_order_label(n: int) -> str:
    return {1: "single", 2: "double", 3: "triple"}.get(int(n), str(n))


# ----------------------------
# PROB .eventb PATCHING
# ----------------------------

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
    loc = (loc or "").strip()

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

def replace_last_arg(term: str, new_last: str) -> str:
    functor = term[:term.find("(")]
    args = _top_level_args(term)
    if not args:
        raise ValueError("Bad term for replace_last_arg")
    args[-1] = new_last
    return f"{functor}(" + ",".join(args) + ")"

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

    ro = re.search(r"(<=|>=|!=|=|<|>)", o)
    rm = re.search(r"(<=|>=|!=|=|<|>)", m)
    if ro and rm and ro.group(1) != rm.group(1):
        return {"kind": "relop_change", "from": ro.group(1), "to": rm.group(1)}

    if ":=" in o and ":=" in m:
        return {"kind": "action_value"}

    return {"kind": "unknown", "o": o, "m": m}

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

    hit = _find_first_match_any(assign_term, from_cands)
    if hit:
        return assign_term.replace(hit, to_primary, 1)
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
        elif kind == "iff_to_impl":
            new_term = patch_iff_to_impl_in_term(term)
        elif kind == "or_to_and":
            new_term = patch_or_to_and_in_term(term)
        elif kind == "and_to_or":
            new_term = patch_and_to_or_in_term(term)
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
        elif kind == "relop_change":
            new_term = patch_relop_in_term(term, edit.original, edit.mutated)
        else:
            new_term = term
        patched_ev = ev_seg[:lt_s] + new_term + ev_seg[lt_e:]
        return full_text[:ev_s] + patched_ev + full_text[ev_e:]

    if label.startswith("act"):
        if kind == "delete":
            new_term = patch_action_omission_in_assign_term(term)
        else:
            try:
                new_term = patch_action_value_in_assign_term(term, edit.original, edit.mutated)
            except Exception:
                new_term = term
        patched_ev = ev_seg[:lt_s] + new_term + ev_seg[lt_e:]
        return full_text[:ev_s] + patched_ev + full_text[ev_e:]

    return full_text


# ----------------------------
# PROBCLI RUNNERS
# ----------------------------
def run_cmd_stream(
    args: List[str],
    logger: logging.Logger,
    cwd: Optional[Path] = None,
    timeout_s: Optional[float] = None,
) -> Tuple[int, str, str, bool]:
    p = subprocess.Popen(
        args,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
        shell=False,
    )

    out_buf: List[str] = []
    err_buf: List[str] = []
    out_lock = threading.Lock()
    err_lock = threading.Lock()

    MAX_CHARS = 200_000

    def _append(buf: List[str], lock: threading.Lock, s: str):
        with lock:
            buf.append(s)
            joined_len = sum(len(x) for x in buf)
            if joined_len > MAX_CHARS:
                while buf and joined_len > MAX_CHARS:
                    joined_len -= len(buf[0])
                    buf.pop(0)

    def _reader(pipe, prefix: str, buf: List[str], lock: threading.Lock):
        try:
            for line in iter(pipe.readline, ""):
                line = line.rstrip("\n")
                logger.info("%s%s", prefix, line)
                _append(buf, lock, line + "\n")
        except Exception:
            pass
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    t_out = threading.Thread(target=_reader, args=(p.stdout, "[STDOUT] ", out_buf, out_lock), daemon=True)
    t_err = threading.Thread(target=_reader, args=(p.stderr, "[STDERR] ", err_buf, err_lock), daemon=True)
    t_out.start()
    t_err.start()

    timed_out = False
    try:
        p.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            p.kill()
        except Exception:
            pass
        try:
            p.wait(timeout=5)
        except Exception:
            pass

    t_out.join(timeout=2)
    t_err.join(timeout=2)

    with out_lock:
        out_text = "".join(out_buf)
    with err_lock:
        err_text = "".join(err_buf)

    rc = p.returncode if p.returncode is not None else 124
    return rc, out_text, err_text, timed_out


def _write_placeholder(path: Path, title: str, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = payload if isinstance(payload, str) else repr(payload)
    safe = safe.replace("\r\n", "\n")
    path.write_text(f"{title}\n{safe}\n", encoding="utf-8", errors="replace")

def run_testcase(probcli: Path, model_eventb: Path, tc: TestCase, out_dir: Path,
                 timeout_ms: int, clpfd: bool, logger: logging.Logger) -> Dict[str, Any]:

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
        rc1, out1, err1, to1 = run_cmd_stream(cmd1, logger=logger, cwd=model_cwd, timeout_s=(tmo / 1000.0) + 10.0)

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
            rc2, out2, err2, to2 = run_cmd_stream(cmd2, logger=logger, cwd=model_cwd, timeout_s=(tmo / 1000.0) + 10.0)
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


# ----------------------------
# show_states PARSER
# ----------------------------

_STATE_SPLIT_RE = re.compile(r"^\s*State\s+(\d+)\s*[:\-]?\s*$", re.IGNORECASE | re.MULTILINE)
_EVENT_LINE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*(?:\([^)]*\))?)\s*;\s*$", re.MULTILINE)
_VARS_BLOCK_RE = re.compile(r"/\*\s*Variables\s*\n.*?\n\s*\*/", re.DOTALL | re.IGNORECASE)
_TOKEN_RE = re.compile(
    r"(?P<vars>/\*\s*Variables\s*\n.*?\n\s*\*/)|(?P<event>^\s*[A-Za-z_][A-Za-z0-9_]*(?:\([^)]*\))?\s*;\s*$)",
    re.MULTILINE | re.DOTALL | re.IGNORECASE
)
_ASSIGN_LINE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$")
_ASSIGN_INLINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$")

def _dedup_last_per_step(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keep: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        keep[int(r.get("step", 0))] = r
    return [keep[k] for k in sorted(keep.keys())]

def parse_states_state_headers(raw: str, sequence: str) -> List[Dict[str, Any]]:
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
        assigns: Dict[str, str] = {}
        for line in body.splitlines():
            mm = _ASSIGN_LINE_RE.match(line)
            if not mm:
                continue
            assigns[mm.group(1).strip()] = mm.group(2).strip()

        step = st_idx
        ev = "" if step == 0 else (events[step - 1] if (step - 1) < len(events) else "")
        row: Dict[str, Any] = {"step": step, "event": ev}
        row.update(assigns)
        rows.append(row)

    rows.sort(key=lambda r: int(r.get("step", 0)))
    return rows

def parse_states_event_variable_blocks(raw: str) -> List[Dict[str, Any]]:
    step = 0
    last_event = ""
    rows: List[Dict[str, Any]] = []

    for m in _TOKEN_RE.finditer(raw):
        if m.group("event"):
            last_event = m.group("event").strip().rstrip(";").strip()
            step += 1
            continue

        vb = m.group("vars")
        inner = re.sub(r"^/\*\s*Variables\s*\n", "", vb, flags=re.IGNORECASE)
        inner = re.sub(r"\n\s*\*/\s*$", "", inner)

        parts = re.split(r"\s*&\s*", inner.strip())
        assigns: Dict[str, str] = {}
        for a in parts:
            a = a.strip()
            if not a:
                continue
            mm = _ASSIGN_INLINE_RE.match(a)
            if not mm:
                continue
            assigns[mm.group(1).strip()] = mm.group(2).strip()

        ev = "" if step == 0 else last_event
        row: Dict[str, Any] = {"step": step if step > 0 else 0, "event": ev}
        row.update(assigns)
        rows.append(row)

    return _dedup_last_per_step(rows)

def parse_states_file(path: Path, sequence: str) -> Tuple[List[Dict[str, Any]], bool, str]:
    if not path.exists():
        return ([], False, f"states file not found: {path}")
    raw = path.read_text(encoding="utf-8", errors="replace")
    if raw.lstrip().startswith("NO_STATES"):
        return ([], False, "NO_STATES placeholder")

    if _STATE_SPLIT_RE.search(raw):
        try:
            return (parse_states_state_headers(raw, sequence), True, "")
        except Exception as e:
            return ([], False, f"parse State-N format failed: {e}")

    if _VARS_BLOCK_RE.search(raw) and _EVENT_LINE_RE.search(raw):
        try:
            return (parse_states_event_variable_blocks(raw), True, "")
        except Exception as e:
            return ([], False, f"parse Variables-block format failed: {e}")

    try:
        return (parse_states_state_headers(raw, sequence), True, "")
    except Exception as e:
        return ([], False, f"unrecognized format: {e}")


# ----------------------------
# FEATURE EXTRACTION + MODELS
# ----------------------------

def _safe_str(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x)

def build_samples_from_states(df_states: pd.DataFrame) -> Tuple[List[str], List[Dict[str, float]], List[Dict[str, str]]]:
    required = {"mutant_id", "tc_id", "step", "event"}
    miss = required - set(df_states.columns)
    if miss:
        raise ValueError(f"expanded states missing columns: {sorted(miss)}")

    df_states = df_states.copy()
    df_states["step"] = pd.to_numeric(df_states["step"], errors="coerce").fillna(-1).astype(int)

    non_var = set(["mutant_id", "tc_id", "step", "event", "parse_ok", "parse_error", "states_path",
                   "mutation_type", "fault_count", "fault_order", "fault_components", "threat_class",
                   "edit_count", "edit_targets"])
    var_cols = [c for c in df_states.columns if c not in non_var]

    sample_keys: List[str] = []
    sparse_dicts: List[Dict[str, float]] = []
    meta_rows: List[Dict[str, str]] = []

    for (mid, tc), g in df_states.groupby(["mutant_id", "tc_id"], sort=False):
        g = g.sort_values("step", ascending=True)
        feats: Dict[str, float] = {}

        if "parse_ok" in g.columns and (g["parse_ok"].astype(str).iloc[0].lower() == "false"):
            feats["RUN_PARSE_FAIL"] = 1.0
        if "no_trace" in g.columns and int(pd.to_numeric(g["no_trace"], errors="coerce").fillna(0).iloc[0]) == 1:
            feats["RUN_NO_TRACE"] = 1.0

        for ev in g["event"].astype(str).tolist():
            ev = ev.strip()
            if not ev:
                continue
            feats[f"EV={ev}"] = feats.get(f"EV={ev}", 0.0) + 1.0

        for _, row in g.iterrows():
            for c in var_cols:
                v = _safe_str(row.get(c, "")).strip()
                if not v:
                    continue
                key = f"{c}={v}"
                feats[key] = feats.get(key, 0.0) + 1.0

        prev = None
        for _, row in g.iterrows():
            if prev is None:
                prev = row
                continue
            for c in var_cols:
                a = _safe_str(prev.get(c, "")).strip()
                b = _safe_str(row.get(c, "")).strip()
                if a and b and a != b:
                    dk = f"DELTA:{c}:{a}->{b}"
                    feats[dk] = feats.get(dk, 0.0) + 1.0
            prev = row

        sample_keys.append(f"{mid}|{tc}")
        sparse_dicts.append(feats)
        meta_rows.append({"mutant_id": str(mid), "tc_id": str(tc)})

    return sample_keys, sparse_dicts, meta_rows

def make_labels(df_states: pd.DataFrame, task: str) -> Dict[str, str]:
    if "fault_components" not in df_states.columns:
        raise ValueError("expanded dataset must contain 'fault_components' to build labels")
    if "fault_count" not in df_states.columns:
        df_states = df_states.copy()
        if "fault_order" in df_states.columns:
            df_states["fault_count"] = df_states["fault_order"].map(lambda s: {"single":1,"double":2,"triple":3}.get(str(s), np.nan))

    lab: Dict[str, str] = {}
    for (mid, tc), g in df_states.groupby(["mutant_id", "tc_id"], sort=False):
        comps = _safe_str(g["fault_components"].iloc[0]).strip()
        cnt = g["fault_count"].iloc[0] if "fault_count" in g.columns else np.nan
        cnti = int(cnt) if (isinstance(cnt, (int, np.integer)) or (isinstance(cnt, float) and not np.isnan(cnt))) else None

        if task == "single":
            if "+" in comps:
                continue
            lab[f"{mid}|{tc}"] = comps if comps else "UNKNOWN"
        elif task == "double":
            if cnti != 2:
                continue
            lab[f"{mid}|{tc}"] = comps if comps else "UNKNOWN"
        elif task == "triple":
            if cnti != 3:
                continue
            lab[f"{mid}|{tc}"] = comps if comps else "UNKNOWN"
        elif task == "any":
            lab[f"{mid}|{tc}"] = comps if comps else "UNKNOWN"
        else:
            raise ValueError(f"Unknown task: {task}")
    return lab

def train_sklearn_models(X, y, seed: int = 42) -> Dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import StratifiedKFold

    models: Dict[str, Any] = {}

    lr = LogisticRegression(max_iter=500, n_jobs=None, class_weight="balanced", random_state=seed)
    models["logreg"] = lr.fit(X, y)

    nb = MultinomialNB()
    models["nb"] = nb.fit(X, y)

    svm = LinearSVC(class_weight="balanced", random_state=seed)

    try:
        classes, counts = np.unique(y, return_counts=True)
        min_count = int(counts.min()) if len(counts) else 0
        n_splits = min(3, min_count)
        if n_splits < 2:
            raise ValueError("too_few_samples_per_class_for_calibration")

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        svm_cal = CalibratedClassifierCV(svm, cv=cv, method="sigmoid")
        models["svm_cal"] = svm_cal.fit(X, y)

    except Exception:
        models["svm"] = svm.fit(X, y)

    rf = RandomForestClassifier(
        n_estimators=400, random_state=seed, class_weight="balanced_subsample", n_jobs=-1, max_depth=None
    )
    models["rf"] = rf.fit(X, y)

    gb = GradientBoostingClassifier(n_estimators=200, random_state=seed, max_depth=5)
    models["gb"] = gb.fit(X, y)

    try:
        import xgboost as xgb
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=seed,
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_jobs=-1
        )
        models["xgboost"] = xgb_model.fit(X, y)
    except ImportError:
        pass

    return models

def eval_model(est, X_test, y_test) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

    y_pred = est.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    rep = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    return {
        "acc": acc, 
        "f1_macro": f1m, 
        "precision": prec, 
        "recall": rec,
        "report": rep, 
        "cm": cm, 
        "y_pred": y_pred
    }

def plot_confusion_matrix(cm: np.ndarray, labels: List[str], out_png: Path, title: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def plot_metrics_over_rounds(history_df: pd.DataFrame, out_png: Path, title: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title)

    metrics = ["acc", "precision", "recall", "f1_macro"]
    titles = ["Accuracy", "Precision", "Recall", "F1 (Macro)"]

    for idx, (metric, mtitle) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        if metric in history_df.columns:
            ax.plot(history_df["round"], history_df[metric], marker='o', linewidth=2)
            ax.set_xlabel("Round")
            ax.set_ylabel(mtitle)
            ax.set_title(mtitle)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"No data for {mtitle}", ha='center', va='center')
            ax.set_title(mtitle)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# ----------------------------
# EXPLAINABILITY (SHAP & LIME)
# ----------------------------

def explain_with_shap(model, vectorizer, X_train, X_test, y_test, out_dir: Path, model_name: str, logger: logging.Logger):
    try:
        import shap
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        logger.info(f"[SHAP] Computing SHAP values for {model_name}...")

        if hasattr(model, "predict_proba"):
            explainer = shap.Explainer(model.predict_proba, X_train)
            shap_values = explainer(X_test)
        else:
            explainer = shap.Explainer(model.predict, X_train)
            shap_values = explainer(X_test)

        feature_names = vectorizer.get_feature_names_out()

        plt.figure()
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_summary_{model_name}.png", dpi=160, bbox_inches='tight')
        plt.close()

        logger.info(f"[SHAP] Saved SHAP summary plot: shap_summary_{model_name}.png")

    except Exception as e:
        logger.info(f"[SHAP] Failed for {model_name}: {e}")


def explain_with_lime(model, vectorizer, X_test, y_test, out_dir: Path, model_name: str, logger: logging.Logger, num_samples: int = 5):
    try:
        from lime.lime_tabular import LimeTabularExplainer

        logger.info(f"[LIME] Generating LIME explanations for {model_name}...")

        X_test_dense = X_test.toarray() if hasattr(X_test, 'toarray') else X_test
        feature_names = vectorizer.get_feature_names_out()

        explainer = LimeTabularExplainer(
            X_test_dense,
            feature_names=feature_names,
            class_names=np.unique(y_test),
            mode='classification'
        )

        lime_reports = []
        for i in range(min(num_samples, X_test_dense.shape[0])):
            exp = explainer.explain_instance(X_test_dense[i], model.predict_proba, num_features=10)
            
            exp_html = exp.as_html()
            html_path = out_dir / f"lime_{model_name}_sample_{i}.html"
            html_path.write_text(exp_html, encoding='utf-8')

            lime_reports.append({
                "sample_idx": i,
                "true_label": y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i],
                "prediction": model.predict(X_test[i:i+1])[0],
                "html_file": f"lime_{model_name}_sample_{i}.html"
            })

        pd.DataFrame(lime_reports).to_csv(out_dir / f"lime_{model_name}_summary.csv", index=False)
        logger.info(f"[LIME] Saved {num_samples} LIME explanations for {model_name}")

    except Exception as e:
        logger.info(f"[LIME] Failed for {model_name}: {e}")


# ----------------------------
# MUTANT PROPOSAL
# ----------------------------

def _edit_signature(e: Edit) -> str:
    s = f"{e.location}||{normalize_eventb_expr(e.original)}||{normalize_eventb_expr(e.mutated)}"
    return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()

def build_edit_pool(mutants: List[Mutant]) -> List[Tuple[str, Edit, str]]:
    pool: List[Tuple[str, Edit, str]] = []
    seen = set()
    for m in mutants:
        for e in m.edits:
            sig = _edit_signature(e)
            if sig in seen:
                continue
            seen.add(sig)
            comp = infer_component_from_location(e.location)
            pool.append((m.mutant_id, e, comp))
    return pool

def pick_underrepresented_targets(task: str, counts: Dict[str, int]) -> List[str]:
    if not counts:
        return ["GUI", "ALRM", "CTRL"]

    if task == "single":
        keys = ["GUI", "ALRM", "CTRL"]
        keys = [k for k in keys if k in counts] or list(counts.keys())
        keys_sorted = sorted(keys, key=lambda k: counts.get(k, 0))
        return keys_sorted

    keys_sorted = sorted(counts.keys(), key=lambda k: counts.get(k, 0))
    return keys_sorted

def propose_mutants_from_pool(
    pool: List[Tuple[str, Edit, str]],
    existing_signatures: set,
    round_idx: int,
    n_new: int,
    fault_orders: List[int],
    task: str,
    class_counts: Dict[str, int],
    seed: int,
) -> List[ProposedMutant]:
    rng = random.Random(seed + 1000 * round_idx)
    targets_ranked = pick_underrepresented_targets(task, class_counts)

    by_comp: Dict[str, List[Tuple[str, Edit]]] = {}
    for src_id, e, comp in pool:
        by_comp.setdefault(comp, []).append((src_id, e))

    out: List[ProposedMutant] = []
    used_in_round = set()

    def sample_edit_for(comp: str) -> Optional[Tuple[str, Edit]]:
        cand = by_comp.get(comp, [])
        if not cand:
            return None
        return rng.choice(cand)

    def combo_to_components(combo: str, k: int) -> List[str]:
        if "+" in combo:
            comps = combo.split("+")
            return [c.strip() for c in comps if c.strip()]
        if k == 1:
            return [combo]
        base = combo
        comps = [base]
        while len(comps) < k:
            comps.append(rng.choice(["GUI", "ALRM", "CTRL"]))
        return comps

    i = 0
    tries = 0
    while i < n_new and tries < n_new * 80:
        tries += 1

        k = rng.choice(fault_orders)
        if task == "single":
            k = 1
        elif task == "double":
            k = 2
        elif task == "triple":
            k = 3

        tgt = rng.choice(targets_ranked) if targets_ranked else rng.choice(["GUI", "ALRM", "CTRL"])

        comps = combo_to_components(tgt, k)
        comps = [c if c in ("GUI", "ALRM", "CTRL") else rng.choice(["GUI", "ALRM", "CTRL"]) for c in comps]

        chosen: List[Tuple[str, Edit, str]] = []
        for c in comps:
            s = sample_edit_for(c)
            if s is None:
                break
            src_id, ed = s
            chosen.append((src_id, ed, c))
        if len(chosen) != k:
            continue

        sigs = [_edit_signature(ed) for _, ed, _ in chosen]
        sig_key = "|".join(sorted(sigs))
        if sig_key in existing_signatures or sig_key in used_in_round:
            continue

        new_id = f"AL_R{round_idx:02d}_{i+1:04d}"
        edits = [dataclasses.replace(ed, mutation_type="ACTIVE_LEARNING") for _, ed, _ in chosen]
        fault_components = normalize_combo([c for _, _, c in chosen])
        source_edit_ids = ",".join([src for src, _, _ in chosen])

        pm = ProposedMutant(
            mutant_id=new_id,
            edits=edits,
            fault_count=k,
            fault_components=fault_components,
            mutation_type="ACTIVE_LEARNING",
            source_edit_ids=source_edit_ids,
        )
        out.append(pm)
        used_in_round.add(sig_key)
        existing_signatures.add(sig_key)
        i += 1

    return out


def mutant_meta_features(pm: ProposedMutant) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    feats[f"k={pm.fault_count}"] = 1.0
    feats[f"combo={pm.fault_components}"] = 1.0

    for e in pm.edits:
        loc = (e.location or "").strip()
        feats[f"loc={loc}"] = feats.get(f"loc={loc}", 0.0) + 1.0

        ev = loc.split("/")[0].strip() if "/" in loc else loc
        if ev:
            feats[f"event={ev}"] = feats.get(f"event={ev}", 0.0) + 1.0

        kind = detect_simple_mutation(e.original, e.mutated).get("kind", "unknown")
        feats[f"kind={kind}"] = feats.get(f"kind={kind}", 0.0) + 1.0

        comp = infer_component_from_location(loc)
        feats[f"comp_hint={comp}"] = feats.get(f"comp_hint={comp}", 0.0) + 1.0

        for tok in re.split(r"[^A-Za-z0-9_]+", loc):
            tok = tok.strip()
            if tok:
                feats[f"tok={tok}"] = feats.get(f"tok={tok}", 0.0) + 1.0

    return feats


def train_selector_model(executed_meta: pd.DataFrame, seed: int = 42):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression

    if executed_meta.empty:
        return None

    need = {"label", "feat_json"}
    if not need.issubset(set(executed_meta.columns)):
        return None

    y = executed_meta["label"].astype(str).tolist()
    if len(set(y)) < 2:
        return None

    feat_dicts = []
    for s in executed_meta["feat_json"].astype(str).tolist():
        try:
            feat_dicts.append(json.loads(s))
        except Exception:
            feat_dicts.append({})

    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(feat_dicts)

    clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)
    clf.fit(X, y)
    return {"vectorizer": vec, "model": clf}


def entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def margin(p: np.ndarray) -> float:
    if p.size == 0:
        return 0.0
    s = np.sort(p)[::-1]
    if len(s) == 1:
        return 1.0
    return float(s[0] - s[1])


def score_uncertainty(selector, cand_feats: Dict[str, float], mode: str) -> float:
    if selector is None:
        return 0.0
    vec = selector["vectorizer"]
    model = selector["model"]
    X = vec.transform([cand_feats])
    proba = model.predict_proba(X)[0]
    if mode == "entropy":
        return entropy(proba)
    if mode == "margin":
        return -margin(proba)
    return 0.0


def jaccard_distance(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return 1.0 - (inter / uni) if uni else 0.0


def cand_token_set(pm: ProposedMutant) -> set:
    toks = set()
    for e in pm.edits:
        loc = (e.location or "")
        for tok in re.split(r"[^A-Za-z0-9_]+", loc):
            tok = tok.strip()
            if tok:
                toks.add(tok)
    toks.add(pm.fault_components)
    toks.add(f"k{pm.fault_count}")
    return toks


def select_mutants(
    candidates: List[ProposedMutant],
    n_select: int,
    strategy: str,
    class_counts: Dict[str, int],
    selector_model,
    seed: int,
) -> List[ProposedMutant]:
    rng = random.Random(seed)
    if not candidates:
        return []

    cands = candidates[:]
    rng.shuffle(cands)

    def label_of(pm: ProposedMutant) -> str:
        return pm.fault_components if pm.fault_components else "UNKNOWN"

    if strategy == "random":
        return cands[:n_select]

    if strategy in ("uncertainty_entropy", "uncertainty_margin"):
        mode = "entropy" if strategy.endswith("entropy") else "margin"
        scored = []
        for pm in cands:
            feats = mutant_meta_features(pm)
            s = score_uncertainty(selector_model, feats, mode=mode)
            scored.append((s, pm))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [pm for _, pm in scored[:n_select]]

    if strategy == "diversity_jaccard":
        picked = []
        if not cands:
            return picked
        picked.append(cands.pop(0))
        picked_sets = [cand_token_set(picked[0])]
        while cands and len(picked) < n_select:
            best_i, best_d = 0, -1.0
            for i, pm in enumerate(cands):
                s = cand_token_set(pm)
                d = min(jaccard_distance(s, ps) for ps in picked_sets)
                if d > best_d:
                    best_d = d
                    best_i = i
            pm = cands.pop(best_i)
            picked.append(pm)
            picked_sets.append(cand_token_set(pm))
        return picked

    return cands[:n_select]


# ----------------------------
# LOGGING SETUP
# ----------------------------

def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("AL")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = RotatingFileHandler(str(log_dir / "run.log"), maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    return logger

def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")


# ----------------------------
# DATASET IO
# ----------------------------

def load_or_init_dataset(seed_dataset_xlsx: Optional[Path], out_dir: Path) -> pd.DataFrame:
    if seed_dataset_xlsx and seed_dataset_xlsx.exists():
        xl = pd.ExcelFile(seed_dataset_xlsx)
        sheet = "states_expanded" if "states_expanded" in xl.sheet_names else xl.sheet_names[0]
        df = pd.read_excel(seed_dataset_xlsx, sheet_name=sheet)
        return df
    return pd.DataFrame()

def safe_write_excel_multi(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for name, df in sheets.items():
            for c in df.columns:
                if df[c].dtype == "object":
                    df[c] = df[c].map(clean_xlsx)
            df.to_excel(w, index=False, sheet_name=name)


# ----------------------------
# MUTANT EXECUTION
# ----------------------------

def copy_baseline_export(baseline_dir: Path, dest_dir: Path) -> None:
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(baseline_dir, dest_dir)

def apply_mutant_to_model(baseline_export_dir: Path, baseline_eventb_name: str, mutant: ProposedMutant, out_model_dir: Path) -> Tuple[Path, int, int, List[Dict[str, Any]]]:

    copy_baseline_export(baseline_export_dir, out_model_dir)
    model_eventb = out_model_dir / baseline_eventb_name
    text = model_eventb.read_text(encoding="utf-8", errors="replace")

    applied, failed = 0, 0
    log_rows: List[Dict[str, Any]] = []
    for i, e in enumerate(mutant.edits, start=1):
        try:
            new_text = apply_edit_to_eventb_prolog(text, e)
            if new_text != text:
                applied += 1
            text = new_text
            log_rows.append({"mutant_id": mutant.mutant_id, "edit_index": i, "location": e.location, "status": "ok", "error": ""})
        except Exception as ex:
            failed += 1
            log_rows.append({"mutant_id": mutant.mutant_id, "edit_index": i, "location": e.location, "status": "fail", "error": str(ex)})
    model_eventb.write_text(text, encoding="utf-8")
    return model_eventb, applied, failed, log_rows

def harvest_mutant_traces(
    probcli: Path,
    model_eventb: Path,
    tests: List[TestCase],
    mutant: ProposedMutant,
    runs_dir: Path,
    timeout_ms: int,
    clpfd: bool, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:


    all_runs: List[Dict[str, Any]] = []
    all_states: List[Dict[str, Any]] = []
    parse_fails: List[Dict[str, Any]] = []

    n_total = len(tests)
    for idx, tc in enumerate(tests, start=1):
        tc_out = runs_dir / tc.tc_id
        res = run_testcase(probcli, model_eventb, tc, tc_out, timeout_ms, clpfd, logger=logger)

        logger.info("[TC] %d/%d %s ok=%d", idx, n_total, tc.tc_id, 1 if res.get("ok") else 0)
        res_row = {
            "mutant_id": mutant.mutant_id,
            "tc_id": tc.tc_id,
            "sequence": tc.sequence,
            "fault_count": mutant.fault_count,
            "fault_order": fault_order_label(mutant.fault_count),
            "fault_components": mutant.fault_components,
            "mutation_type": mutant.mutation_type,
            "source_edit_ids": mutant.source_edit_ids,
        }
        res_row.update(res)
        all_runs.append(res_row)

        rows, ok, err = parse_states_file(Path(res["states_path"]), tc.sequence)
        if not rows:
            parse_fails.append({
                "mutant_id": mutant.mutant_id,
                "tc_id": tc.tc_id,
                "states_path": res["states_path"],
                "parse_ok": ok,
                "parse_error": err,
            })

            trace_path = Path(res.get("trace_path", "")) if res.get("trace_path") else None
            no_states = 0
            no_trace = 0
            ev_tok = "__PARSE_FAIL__"

            try:
                if Path(res["states_path"]).exists():
                    first = Path(res["states_path"]).read_text(encoding="utf-8", errors="replace").lstrip()[:40].upper()
                    if first.startswith("NO_STATES"):
                        no_states = 1
                        ev_tok = "__NO_STATES__"
            except Exception:
                pass

            try:
                if trace_path and trace_path.exists():
                    first = trace_path.read_text(encoding="utf-8", errors="replace").lstrip()[:40].upper()
                    if first.startswith("NO_TRACE"):
                        no_trace = 1
                        ev_tok = "__NO_TRACE__"
            except Exception:
                pass

            out = {
                "mutant_id": mutant.mutant_id,
                "tc_id": tc.tc_id,
                "step": -1,
                "event": ev_tok,
                "fault_count": mutant.fault_count,
                "fault_order": fault_order_label(mutant.fault_count),
                "fault_components": mutant.fault_components,
                "mutation_type": mutant.mutation_type,
                "source_edit_ids": mutant.source_edit_ids,
                "parse_ok": False,
                "parse_error": err if err else "parse_failed",
                "states_path": res["states_path"],
                "no_states": int(no_states),
                "no_trace": int(no_trace),

                "ok": bool(res.get("ok", False)),
                "rc_trace": res.get("rc_trace", ""),
                "rc_replay": res.get("rc_replay", ""),
                "timed_out_trace": bool(res.get("timed_out_trace", False)),
                "timed_out_replay": bool(res.get("timed_out_replay", False)),
            }
            all_states.append(out)
            continue

        for r in rows:
            out = {
                "mutant_id": mutant.mutant_id,
                "tc_id": tc.tc_id,
                "step": int(r.get("step", 0)),
                "event": str(r.get("event", "")),
                "fault_count": mutant.fault_count,
                "fault_order": fault_order_label(mutant.fault_count),
                "fault_components": mutant.fault_components,
                "mutation_type": mutant.mutation_type,
                "source_edit_ids": mutant.source_edit_ids,
                "parse_ok": True,
                "parse_error": "",
                "states_path": res["states_path"],
            }
            for k, v in r.items():
                if k in ("step", "event"):
                    continue
                out[k] = v
            all_states.append(out)

    try:
        timeouts = 0
        for r in all_runs:
            if r.get("timed_out_trace") or r.get("timed_out_replay"):
                timeouts += 1
        ok_rate = float(pd.DataFrame(all_runs)["ok"].mean()) if all_runs else float("nan")
        logger.info(
            "[MUTANT_SUMMARY] mutant=%s ok_rate=%.4f parse_fails=%d timeouts=%d",
            mutant.mutant_id,
            ok_rate,
            int(len(parse_fails)),
            int(timeouts),
        )
    except Exception:
        pass

    return pd.DataFrame(all_runs), pd.DataFrame(all_states), pd.DataFrame(parse_fails)


# ----------------------------
# ACTIVE LEARNING LOOP
# ----------------------------

def compute_class_counts(labels: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for y in labels:
        out[y] = out.get(y, 0) + 1
    return out


def check_convergence(history_df: pd.DataFrame, patience: int = 3, threshold: float = 0.01) -> bool:
    if len(history_df) < patience:
        return False
    
    recent = history_df.tail(patience)
    if "f1_macro" not in recent.columns:
        return False
    
    f1_vals = recent["f1_macro"].values
    if len(f1_vals) < patience:
        return False
    
    diffs = np.diff(f1_vals)
    if np.all(np.abs(diffs) < threshold):
        return True
    
    return False


def run_active_learning(args: argparse.Namespace) -> None:

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_dir)

    logger.info("Starting active learning")
    logger.info("out_dir=%s", str(out_dir))

    tests = read_tests_xlsx(Path(args.tests_xlsx).resolve(), TEST_XLSX_MAPPING)
    logger.info("[INFO] loaded_tests=%d", len(tests))

    mutants, mutants_raw_df = read_mutants_xlsx(Path(args.mutants_xlsx).resolve(), MUTANT_XLSX_MAPPING)

    pool = build_edit_pool(mutants)

    seed_dataset = Path(args.seed_dataset_xlsx).resolve() if args.seed_dataset_xlsx else None
    df_states = load_or_init_dataset(seed_dataset, out_dir)

    if not df_states.empty and "fault_components" not in df_states.columns:
        if "ID" in mutants_raw_df.columns:
            meta = mutants_raw_df.copy()
            meta["mutant_id"] = meta["ID"].astype(str)

            keep_cols = ["mutant_id"]
            for c in ["Fault_Count", "Fault_Components", "Mutation_Type", "Threat_Class"]:
                if c in meta.columns:
                    keep_cols.append(c)

            meta = meta[keep_cols].drop_duplicates("mutant_id")
            df_states = df_states.merge(meta, on="mutant_id", how="left")

            if "Fault_Count" in df_states.columns and "fault_count" not in df_states.columns:
                df_states["fault_count"] = pd.to_numeric(df_states["Fault_Count"], errors="coerce")
            if "Fault_Components" in df_states.columns and "fault_components" not in df_states.columns:
                df_states["fault_components"] = df_states["Fault_Components"].astype(str)

    if not df_states.empty and not args.include_base:
        df_states = df_states[df_states["mutant_id"].astype(str).str.upper() != "BASE"].copy()

    master_xlsx = out_dir / "active_learning_dataset.xlsx"

    history_rows: List[Dict[str, Any]] = []
    proposed_rows: List[Dict[str, Any]] = []
    mutate_log_rows: List[Dict[str, Any]] = []
    parse_fail_rows: List[Dict[str, Any]] = []
    run_rows_accum: List[pd.DataFrame] = []
    new_states_accum: List[pd.DataFrame] = []

    existing_sig_set: set = set()

    executed_meta_path = out_dir / "executed_mutants_meta.csv"

    strategy = getattr(args, "strategy_name", "random")
    cand_mult = int(getattr(args, "candidate_multiplier", 10))

    probcli = Path(args.probcli).resolve()
    baseline_export_dir = Path(args.baseline_export_dir).resolve()
    baseline_eventb_name = args.baseline_eventb_name
    clpfd = not args.no_clpfd

    fault_orders = [int(x) for x in str(args.fault_orders).split(",") if x.strip()]

    for r in range(1, args.rounds + 1):
        round_dir = out_dir / f"round_{r:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        best_model_name = "none"
        best_metrics: Dict[str, Any] = {"acc": np.nan, "f1_macro": np.nan, "precision": np.nan, "recall": np.nan}
        class_counts: Dict[str, int] = {}
        best_est = None
        vectorizer = None

        if df_states.empty:
            logger.info(f"[ROUND {r}] No seed dataset; skipping training.")
        else:
            try:
                keys, feat_dicts, _meta = build_samples_from_states(df_states)
                label_map = make_labels(df_states, args.task)

                kept_idx = [i for i, k in enumerate(keys) if k in label_map]

                if len(kept_idx) < 20:
                    logger.info(f"[ROUND {r}] Not enough labeled samples for task={args.task}. Have {len(kept_idx)}.")
                else:
                    keys2 = [keys[i] for i in kept_idx]
                    feats2 = [feat_dicts[i] for i in kept_idx]
                    y = [label_map[k] for k in keys2]

                    class_counts = compute_class_counts(y)

                    from sklearn.feature_extraction import DictVectorizer
                    from sklearn.model_selection import train_test_split

                    vec = DictVectorizer(sparse=True)
                    X = vec.fit_transform(feats2)
                    vectorizer = vec

                    strat = y if len(set(y)) > 1 else None
                    Xtr, Xte, ytr, yte = train_test_split(
                        X, y, test_size=0.25, random_state=args.seed + r, stratify=strat
                    )

                    models = train_sklearn_models(Xtr, ytr, seed=args.seed + r)

                    best_est = None
                    best_model_name = None
                    best_metrics = None

                    for name, est in models.items():
                        m = eval_model(est, Xte, yte)
                        if best_metrics is None or m["f1_macro"] > best_metrics["f1_macro"]:
                            best_metrics = m
                            best_model_name = name
                            best_est = est

                    best_metrics = best_metrics or {"acc": np.nan, "f1_macro": np.nan, "precision": np.nan, "recall": np.nan, "cm": None, "report": ""}

                    try:
                        labs_sorted = sorted(list(set(y)))
                        cm_png = round_dir / f"cm_{args.task}_{best_model_name}.png"
                        plot_confusion_matrix(best_metrics["cm"], labs_sorted, cm_png, f"Round {r} best={best_model_name}")
                        
                        (round_dir / f"report_{args.task}_{best_model_name}.txt").write_text(
                            str(best_metrics.get("report", "")), encoding="utf-8"
                        )
                        import joblib
                        joblib.dump({"vectorizer": vec, "model": best_est}, round_dir / f"model_{args.task}_{best_model_name}.joblib")
                    except Exception as e:
                        logger.info(f"[ROUND {r}] Failed to save artifacts: {e}")

                    if best_est is not None and vectorizer is not None:
                        explain_with_shap(best_est, vectorizer, Xtr, Xte, yte, round_dir, best_model_name, logger)
                        explain_with_lime(best_est, vectorizer, Xte, yte, round_dir, best_model_name, logger, num_samples=5)

            except Exception as e:
                logger.info(f"[ROUND {r}] Training failed: {e}")

        history_rows.append({
            "round": r,
            "task": args.task,
            "strategy": strategy,
            "best_model": best_model_name,
            "acc": float(best_metrics.get("acc", np.nan)) if isinstance(best_metrics, dict) else np.nan,
            "precision": float(best_metrics.get("precision", np.nan)) if isinstance(best_metrics, dict) else np.nan,
            "recall": float(best_metrics.get("recall", np.nan)) if isinstance(best_metrics, dict) else np.nan,
            "f1_macro": float(best_metrics.get("f1_macro", np.nan)) if isinstance(best_metrics, dict) else np.nan,
            "class_counts_json": json.dumps(class_counts, ensure_ascii=False),
        })

        try:
            _acc = float(best_metrics.get("acc", np.nan)) if isinstance(best_metrics, dict) else float("nan")
            _prec = float(best_metrics.get("precision", np.nan)) if isinstance(best_metrics, dict) else float("nan")
            _rec = float(best_metrics.get("recall", np.nan)) if isinstance(best_metrics, dict) else float("nan")
            _f1  = float(best_metrics.get("f1_macro", np.nan)) if isinstance(best_metrics, dict) else float("nan")
        except Exception:
            _acc, _prec, _rec, _f1 = float("nan"), float("nan"), float("nan"), float("nan")

        logger.info(
            "[ROUND %d] metrics: best_model=%s acc=%.4f prec=%.4f rec=%.4f f1_macro=%.4f class_counts=%s",
            r,
            str(best_model_name),
            _acc, _prec, _rec, _f1,
            json.dumps(class_counts, ensure_ascii=False),
        )

        df_history = pd.DataFrame(history_rows)
        if check_convergence(df_history, patience=args.patience, threshold=args.convergence_threshold):
            logger.info(f"[ROUND {r}] CONVERGENCE DETECTED (patience={args.patience}). Stopping early.")
            break

        n_candidates = max(args.new_mutants_per_round * max(1, cand_mult), args.new_mutants_per_round)

        candidates = propose_mutants_from_pool(
            pool=pool,
            existing_signatures=existing_sig_set,
            round_idx=r,
            n_new=n_candidates,
            fault_orders=fault_orders,
            task=args.task,
            class_counts=class_counts,
            seed=args.seed,
        )

        if not candidates:
            logger.info(f"[ROUND {r}] Could not generate candidates. Stopping.")
            break

        executed_meta = pd.read_csv(executed_meta_path) if executed_meta_path.exists() else pd.DataFrame()
        selector_model = train_selector_model(executed_meta, seed=args.seed + r)

        selected_mutants = select_mutants(
            candidates=candidates,
            n_select=args.new_mutants_per_round,
            strategy=strategy,
            class_counts=class_counts,
            selector_model=selector_model,
            seed=args.seed + 100 * r,
        )

        logger.info(f"[ROUND {r}] strategy={strategy} candidates={len(candidates)} selected={len(selected_mutants)}")

        for pm in selected_mutants:
            proposed_rows.append({
                "round": r,
                "strategy": strategy,
                "mutant_id": pm.mutant_id,
                "fault_count": pm.fault_count,
                "fault_order": fault_order_label(pm.fault_count),
                "fault_components": pm.fault_components,
                "source_edit_ids": pm.source_edit_ids,
                "edit_locations": " || ".join([e.location for e in pm.edits]),
            })

        for pm in selected_mutants:
            logger.info(f"[ROUND {r}] Run mutant {pm.mutant_id} | {pm.fault_components} | k={pm.fault_count}")

            mutant_root = round_dir / pm.mutant_id
            model_dir = mutant_root / "model"
            runs_dir = mutant_root / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)

            model_eventb, applied, failed, mlog = apply_mutant_to_model(
                baseline_export_dir, baseline_eventb_name, pm, model_dir
            )
            for row in mlog:
                row["round"] = r
                row["strategy"] = strategy
                mutate_log_rows.append(row)

            runs_df, states_df, pf_df = harvest_mutant_traces(
                probcli=probcli,
                model_eventb=model_eventb,
                tests=tests,
                mutant=pm,
                runs_dir=runs_dir,
                timeout_ms=args.timeout_ms,
                clpfd=clpfd,
                logger=logger,
            )

            runs_df.to_csv(mutant_root / "runs.csv", index=False)
            states_df.to_csv(mutant_root / "states_expanded.csv", index=False)
            if not pf_df.empty:
                pf_df.to_csv(mutant_root / "parse_fails.csv", index=False)

            run_rows_accum.append(runs_df)
            new_states_accum.append(states_df)
            if not pf_df.empty:
                pf_df["round"] = r
                parse_fail_rows.extend(pf_df.to_dict("records"))

            feat = mutant_meta_features(pm)
            if args.task == "single":
                lbl = pm.fault_components.split("+")[0] if pm.fault_components else "UNKNOWN"
            else:
                lbl = pm.fault_components if pm.fault_components else "UNKNOWN"

            meta_row = {
                "round": r,
                "strategy": strategy,
                "mutant_id": pm.mutant_id,
                "label": lbl,
                "fault_components": pm.fault_components,
                "fault_count": pm.fault_count,
                "feat_json": json.dumps(feat, ensure_ascii=False),
                "edit_locations": " || ".join([e.location for e in pm.edits]),
                "applied_edits": int(applied),
                "failed_edits": int(failed),
                "runs_ok_rate": float(runs_df["ok"].mean())
                if (not runs_df.empty and "ok" in runs_df.columns)
                else np.nan,
            }

            if executed_meta_path.exists():
                oldm = pd.read_csv(executed_meta_path)
                newm = pd.concat([oldm, pd.DataFrame([meta_row])], ignore_index=True)
            else:
                newm = pd.DataFrame([meta_row])
            newm.to_csv(executed_meta_path, index=False)

        # end: for pm in selected_mutants

        # Merge newly harvested states into training pool for next round
        if new_states_accum:
            df_new = pd.concat(new_states_accum, ignore_index=True)

            if "mutant_id" not in df_new.columns or "tc_id" not in df_new.columns:
                raise RuntimeError("harvest produced invalid states rows")

            if df_states.empty:
                df_states = df_new.copy()
            else:
                df_states = pd.concat([df_states, df_new], ignore_index=True)

            new_states_accum = []

    # end: for r in range(1, args.rounds + 1)

    df_history = pd.DataFrame(history_rows)
    df_proposed = pd.DataFrame(proposed_rows)
    df_mutlog = pd.DataFrame(mutate_log_rows)
    df_parsefail = pd.DataFrame(parse_fail_rows)
    df_runs = pd.concat(run_rows_accum, ignore_index=True) if run_rows_accum else pd.DataFrame()
    df_states_final = df_states.copy()

    for dfx in [df_history, df_proposed, df_mutlog, df_parsefail, df_runs, df_states_final]:
        for c in dfx.columns:
            if dfx[c].dtype == "object":
                dfx[c] = dfx[c].map(clean_xlsx)

    safe_write_excel_multi(
        master_xlsx,
        {
            "states_expanded": df_states_final,
            "runs": df_runs,
            "proposed_mutants": df_proposed,
            "mutate_log": df_mutlog,
            "parse_fails": df_parsefail,
            "training_history": df_history,
        },
    )

    df_history.to_csv(out_dir / "training_history.csv", index=False)
    df_proposed.to_csv(out_dir / "proposed_mutants.csv", index=False)

    if not df_history.empty:
        metrics_png = out_dir / "metrics_over_rounds.png"
        plot_metrics_over_rounds(df_history, metrics_png, f"Metrics Evolution - {strategy}")
        logger.info(f"[PLOTS] Saved: {metrics_png}")

    logger.info(f"[DONE] Wrote: {master_xlsx}")
    logger.info(f"[DONE] States rows: {len(df_states_final)}")
    logger.info(f"[DONE] Proposed mutants: {len(df_proposed)}")
    logger.info(f"[DONE] Executed meta: {executed_meta_path}")


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probcli", required=True, help="Path to probcli executable")
    ap.add_argument(
        "--baseline_export_dir",
        required=True,
        help="Baseline ProB export dir (contains *.eventb)",
    )
    ap.add_argument(
        "--baseline_eventb_name",
        default="Ventilator_Composed_mch.eventb",
        help="Event-B .eventb filename inside baseline export dir",
    )

    ap.add_argument("--tests_xlsx", required=True, help="Test suite xlsx (TC_ID, Sequence)")
    ap.add_argument(
        "--mutants_xlsx",
        required=True,
        help="Mutants catalog xlsx (ID, Edit* columns etc)",
    )
    ap.add_argument(
        "--seed_dataset_xlsx",
        default="",
        help="Existing expanded_states.xlsx (seed).",
    )

    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--timeout_ms", type=int, default=DEFAULT_TIMEOUT_MS)
    ap.add_argument("--no_clpfd", action="store_true")

    ap.add_argument(
        "--task",
        default="any",
        choices=["single", "double", "triple", "any"],
        help="Label task used for training and selection.",
    )
    ap.add_argument("--include_base", type=int, default=0, help="1 to keep BASE rows; 0 to drop.")

    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--new_mutants_per_round", type=int, default=3)
    ap.add_argument(
        "--fault_orders",
        default="1,2,3",
        help="Allowed fault orders for proposals (comma-separated).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=3, help="Convergence patience (rounds)")
    ap.add_argument(
        "--convergence_threshold",
        type=float,
        default=0.01,
        help="F1 improvement threshold for convergence",
    )

    ap.add_argument(
        "--strategies",
        default="random,uncertainty_entropy,uncertainty_margin,diversity_jaccard",
        help="Comma-separated strategies (random, uncertainty_entropy, uncertainty_margin, diversity_jaccard).",
    )
    ap.add_argument(
        "--strategy",
        default="",
        help="Single strategy (overrides --strategies).",
    )
    ap.add_argument(
        "--candidate_multiplier",
        type=int,
        default=10,
        help="Generate candidate_multiplier*N candidates per round; then select N by strategy.",
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.seed_dataset_xlsx:
        p = Path(args.seed_dataset_xlsx).resolve()
        if not p.exists():
            raise FileNotFoundError(p)

    if args.strategy.strip():
        strategies = [args.strategy.strip()]
    else:
        strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    bad = [s for s in strategies if s not in AL_STRATEGIES]
    if bad:
        raise ValueError(f"unknown strategies {bad}; use one of {AL_STRATEGIES}")

    root_out = Path(args.out_dir).resolve()
    root_out.mkdir(parents=True, exist_ok=True)

    summary = []
    for strat in strategies:
        strat_args = argparse.Namespace(**vars(args))
        strat_args.strategy_name = strat
        strat_args.out_dir = str(root_out / f"strategy_{strat}")

        run_active_learning(strat_args)

        hist_csv = Path(strat_args.out_dir) / "training_history.csv"
        if hist_csv.exists():
            h = pd.read_csv(hist_csv)
            if not h.empty:
                last = h.iloc[-1].to_dict()
                summary.append(
                    {
                        "strategy": strat,
                        "rounds": int(last.get("round", 0)),
                        "acc": float(last.get("acc", float("nan"))),
                        "precision": float(last.get("precision", float("nan"))),
                        "recall": float(last.get("recall", float("nan"))),
                        "f1_macro": float(last.get("f1_macro", float("nan"))),
                        "best_model": str(last.get("best_model", "")),
                    }
                )

    if summary:
        df_sum = pd.DataFrame(summary)
        out_sum = root_out / "strategies_summary.csv"
        df_sum.to_csv(out_sum, index=False)
        print(f"[DONE] Wrote: {out_sum}")


if __name__ == "__main__":
    main()

