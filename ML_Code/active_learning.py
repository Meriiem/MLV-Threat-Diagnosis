from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_TIMEOUT_MS = 8000

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
    "Activate*Alarm": ["ActivateBatteryAlarm", "ActivateVoltageAlarm", "ActivatePressureAlarm", "ActivateTemperatureAlarm"]
}

GUI_HINTS = ("GUI", "Navigate", "Select", "Freeze", "Unfreeze", "Display", "Screen")
ALRM_HINTS = ("Alarm", "Voltage", "Pressure", "Temperature", "Battery", "ALRM")
CTRL_HINTS = ("RespRate", "InspPressure", "IERatio", "Ventilation", "PCV", "Controller", "CTRL")

_ILLEGAL_XLSX_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def clean_xlsx(v):
    if isinstance(v, str):
        return _ILLEGAL_XLSX_RE.sub("", v)
    return v


def pick_col(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"missing column for {label}, tried {candidates}, found {list(df.columns)}")


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


def read_tests_xlsx(path, mapping):
    df = pd.read_excel(path)
    tc_col = pick_col(df, mapping["tc_id_col"], "testcase id")
    seq_col = pick_col(df, mapping["sequence_col"], "sequence")
    tests = []
    for _, row in df.iterrows():
        tc = str(row.get(tc_col, "")).strip()
        seq = str(row.get(seq_col, "")).strip()
        if not tc or tc.lower() == "nan" or not seq or seq.lower() == "nan":
            continue
        tests.append(TestCase(tc_id=tc, sequence=seq))
    return tests


def read_mutants_xlsx(path, mapping):
    df = pd.read_excel(path)
    mid_col = pick_col(df, mapping["mutant_id_col"], "mutant id")
    try:
        mt_col = pick_col(df, mapping.get("mutation_type_col", []), "mutation type")
    except Exception:
        mt_col = None
    mutants = []
    for _, row in df.iterrows():
        mid = str(row.get(mid_col, "")).strip()
        if not mid or mid.lower() == "nan":
            continue
        mutation_type = str(row.get(mt_col, "")).strip() if mt_col else ""
        edits = []
        for loc_cols, orig_cols, mut_cols in mapping["edits"]:
            try:
                loc_col = pick_col(df, loc_cols, "edit target")
                orig_col = pick_col(df, orig_cols, "edit original")
                mut_col = pick_col(df, mut_cols, "edit mutated")
            except ValueError:
                continue
            loc = str(row.get(loc_col, "")).strip()
            orig = str(row.get(orig_col, "")).strip()
            mut = str(row.get(mut_col, "")).strip()
            if loc and loc.lower() != "nan" and orig and orig.lower() != "nan" and mut and mut.lower() != "nan":
                edits.append(Edit(location=loc, original=orig, mutated=mut, mutation_type=mutation_type))
        mutants.append(Mutant(mutant_id=mid, edits=edits))
    return mutants, df


def infer_component_from_location(loc):
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


def normalize_combo(comps):
    comps2 = [c for c in comps if c and c != "UNKNOWN"]
    if not comps2:
        return "UNKNOWN"
    return "+".join(sorted(comps2))


def fault_order_label(n):
    return {1: "single", 2: "double", 3: "triple"}.get(int(n), str(n))


_LOC_EVENT_LABEL_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*/\s*@\s*(grd\d+|act\d+)\s*$")
_LOC_EVENT_ACTION_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*/\s*action\s*$", re.IGNORECASE)
_LOC_INV_RE = re.compile(r"inv(\d+)", re.IGNORECASE)
_ACT_IN_SNIPPET_RE = re.compile(r"@act(\d+)", re.IGNORECASE)


def normalize_eventb_expr(s):
    s = (s or "").strip()
    s = s.replace("\u2254", ":=").replace("\u2264", "<=").replace("\u2265", ">=").replace("\u2260", "!=")
    s = s.replace("\u2227", "&&").replace("\u2228", "||").replace("\u21d4", "<->").replace("\u21d2", "->")
    return re.sub(r"\s+", " ", s)


def detect_machine_name(eventb_text):
    m = re.search(r"rodinpos\('([^']+)'\s*,", eventb_text)
    if not m:
        raise ValueError("cannot detect machine name in .eventb")
    return m.group(1)


def parse_location(loc, original, mutated):
    loc = (loc or "").strip()
    inv_m = _LOC_INV_RE.search(loc)
    if inv_m and "inv" in loc.lower():
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
            raise ValueError(f"location {loc} has /action but no @actK to infer")
        return (ev, f"act{am.group(1)}")
    m3 = re.match(r"^\s*([A-Za-z0-9_]+)\s*/\s*@\s*(grd\d+|act\d+)", loc)
    if m3:
        return (m3.group(1), m3.group(2))
    if re.fullmatch(r"@?inv\d+", loc, flags=re.IGNORECASE):
        nm = re.search(r"inv(\d+)", loc, flags=re.IGNORECASE)
        return ("invariant", f"inv{nm.group(1)}")
    raise ValueError(f"unrecognized location format: {loc}")


def find_event_segment(text, event_name, machine_name):
    marker1 = f"event(rodinpos('{machine_name}','{event_name}'"
    start = text.find(marker1)
    if start == -1:
        marker2 = f"event(rodinpos('{machine_name}','{event_name}','"
        start = text.find(marker2)
    if start == -1:
        m = re.search(rf"event\(rodinpos\('{re.escape(machine_name)}','{re.escape(event_name)}'[^)]*\)", text)
        if not m:
            raise ValueError(f"event {event_name} not found for machine {machine_name}")
        start = m.start()
    next_marker = f"event(rodinpos('{machine_name}','"
    end = text.find(next_marker, start + 10)
    if end == -1:
        end = len(text)
    return start, end


def find_label_pos(segment, label, machine_name):
    for pat in [f"rodinpos('{machine_name}',{label},", f"rodinpos('{machine_name}','{label}',"]:
        pos = segment.find(pat)
        if pos != -1:
            return pos
    return -1


def extract_labeled_term(segment, label, machine_name):
    pos = find_label_pos(segment, label, machine_name)
    if pos == -1:
        raise ValueError(f"label {label} not found for machine {machine_name}")
    paren = segment.rfind("(", 0, pos)
    if paren == -1:
        raise ValueError("no ( before label")
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
                return functor_start, j + 1, segment[functor_start:j + 1]
        j += 1
    raise ValueError("unbalanced parens extracting term")


def replace_functor_name(term, new_functor):
    k = term.find("(")
    if k == -1:
        raise ValueError("bad term, no (")
    return new_functor + term[k:]


def prolog_identifier_candidates(token):
    t = (token or "").strip()
    if t.upper() == "TRUE":
        return ["boolean_true(none)"]
    if t.upper() == "FALSE":
        return ["boolean_false(none)"]
    if re.fullmatch(r"-?\d+", t):
        return [f"integer(none,{t})", t]
    t_plain = t.strip("'\"")
    return [f"identifier(none,{t_plain})", f"identifier(none,'{t_plain}')", f'identifier(none,"{t_plain}")']


def to_prolog_identifier(token):
    t = (token or "").strip()
    if t.upper() == "TRUE":
        return "boolean_true(none)"
    if t.upper() == "FALSE":
        return "boolean_false(none)"
    if re.fullmatch(r"-?\d+", t):
        return f"integer(none,{t})"
    if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
        return f"identifier(none,'{t.strip(chr(39)+chr(34))}')"
    if re.match(r"^[a-z_][a-z0-9_]*$", t):
        return f"identifier(none,{t})"
    return f"identifier(none,'{t}')"


def top_level_args(term):
    lp = term.find("(")
    rp = term.rfind(")")
    if lp == -1 or rp == -1 or rp <= lp:
        return []
    body = term[lp + 1:rp]
    args, depth, cur = [], 0, []
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


def replace_last_arg(term, new_last):
    functor = term[:term.find("(")]
    args = top_level_args(term)
    if not args:
        raise ValueError("bad term for replace_last_arg")
    args[-1] = new_last
    return f"{functor}(" + ",".join(args) + ")"


def detect_simple_mutation(original, mutated):
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


def patch_relop_in_term(term, original, mutated):
    o = normalize_eventb_expr(original)
    m = normalize_eventb_expr(mutated)
    ro = re.search(r"(<=|>=|!=|=|<|>)", o)
    rm = re.search(r"(<=|>=|!=|=|<|>)", m)
    if not ro or not rm:
        raise ValueError(f"cannot detect relop change: {original} -> {mutated}")
    rel_map = {"<=": "less_equal", "<": "less", ">": "greater", ">=": "greater_equal", "=": "equal", "!=": "not_equal"}
    f_from, f_to = rel_map.get(ro.group(1)), rel_map.get(rm.group(1))
    if not f_from or not f_to:
        raise ValueError(f"unsupported relop mapping {ro.group(1)} -> {rm.group(1)}")
    if f_from == f_to:
        return term
    if (f_from + "(") in term:
        return term.replace(f_from + "(", f_to + "(", 1)
    raise ValueError("could not apply relop patch")


def patch_or_to_and_in_term(term):
    if term.startswith("disjunct("):
        return replace_functor_name(term, "conjunct")
    return term.replace("disjunct(", "conjunct(", 1)


def patch_and_to_or_in_term(term):
    if term.startswith("conjunct("):
        return replace_functor_name(term, "disjunct")
    return term.replace("conjunct(", "disjunct(", 1)


def patch_iff_to_impl_in_term(term):
    if term.startswith("equivalence(") or term.startswith("equiv("):
        return replace_functor_name(term, "implication")
    if "equivalence(" in term:
        return term.replace("equivalence(", "implication(", 1)
    if "equiv(" in term:
        return term.replace("equiv(", "implication(", 1)
    return term


def find_first_match_any(term, needles):
    for n in needles:
        if n and n in term:
            return n
    return None


def patch_action_value_in_assign_term(assign_term, original, mutated):
    o2 = re.sub(r"^@act\d+\s+", "", normalize_eventb_expr(original)).strip()
    m2 = re.sub(r"^@act\d+\s+", "", normalize_eventb_expr(mutated)).strip()
    mo = re.search(r":=\s*([A-Za-z0-9_'\-]+)", o2)
    mm = re.search(r":=\s*([A-Za-z0-9_'\-]+)", m2)
    if not mo or not mm:
        raise ValueError(f"cannot parse action value: {original} -> {mutated}")
    from_cands = prolog_identifier_candidates(mo.group(1))
    to_primary = to_prolog_identifier(mm.group(1))
    hit = find_first_match_any(assign_term, from_cands)
    if hit:
        return assign_term.replace(hit, to_primary, 1)
    return assign_term


def extract_balanced_brackets(s, start):
    if start < 0 or start >= len(s) or s[start] != "[":
        raise ValueError("bracket extraction must start at [")
    depth, i = 0, start
    while i < len(s):
        if s[i] == "[":
            depth += 1
        elif s[i] == "]":
            depth -= 1
            if depth == 0:
                return start, i + 1, s[start:i + 1]
        i += 1
    raise ValueError("unbalanced brackets")


def patch_action_omission_in_assign_term(assign_term):
    lb1 = assign_term.find("[")
    if lb1 == -1:
        raise ValueError("no [ found in assignment term")
    s1, e1, list1 = extract_balanced_brackets(assign_term, lb1)
    lb2 = assign_term.find("[", e1)
    if lb2 == -1:
        raise ValueError("no second [ found in assignment term")
    s2, e2, _ = extract_balanced_brackets(assign_term, lb2)
    return assign_term[:s2] + list1 + assign_term[e2:]


def apply_edit_to_eventb_prolog(full_text, edit):
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


def run_cmd_stream(args, cwd=None, timeout_s=None):
    p = subprocess.Popen(args, cwd=str(cwd) if cwd else None, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, bufsize=1, universal_newlines=True, shell=False)
    out_buf, err_buf = [], []
    out_lock, err_lock = threading.Lock(), threading.Lock()
    MAX_CHARS = 200_000

    def append(buf, lock, s):
        with lock:
            buf.append(s)
            joined_len = sum(len(x) for x in buf)
            if joined_len > MAX_CHARS:
                while buf and joined_len > MAX_CHARS:
                    joined_len -= len(buf[0])
                    buf.pop(0)

    def reader(pipe, buf, lock):
        try:
            for line in iter(pipe.readline, ""):
                append(buf, lock, line.rstrip("\n") + "\n")
        except Exception:
            pass
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    t_out = threading.Thread(target=reader, args=(p.stdout, out_buf, out_lock), daemon=True)
    t_err = threading.Thread(target=reader, args=(p.stderr, err_buf, err_lock), daemon=True)
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


def write_placeholder(path, title, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = (payload if isinstance(payload, str) else repr(payload)).replace("\r\n", "\n")
    path.write_text(f"{title}\n{safe}\n", encoding="utf-8", errors="replace")


def run_testcase(probcli, model_eventb, tc, out_dir, timeout_ms, clpfd):
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = (out_dir / f"{tc.tc_id}.trace").resolve()
    states_path = (out_dir / f"{tc.tc_id}_states.txt").resolve()
    seq = tc.sequence.strip()
    model_cwd = model_eventb.parent.resolve()
    model_arg = model_eventb.name
    retry_timeouts = [timeout_ms, max(timeout_ms * 3, timeout_ms + 5000), max(timeout_ms * 10, timeout_ms + 20000)]
    last = {"rc_trace": 999, "rc_replay": 999, "ok": False, "timed_out_trace": False, "timed_out_replay": False}
    for attempt, tmo in enumerate(retry_timeouts, start=1):
        cmd1 = [str(probcli), model_arg, "-init", "-p", "CLPFD", "TRUE" if clpfd else "FALSE",
                "-p", "TIME_OUT", str(tmo), "-cbc_sequence", seq, "-his", str(trace_path), "-his_option", "trace_file"]
        rc1, out1, err1, to1 = run_cmd_stream(cmd1, cwd=model_cwd, timeout_s=(tmo / 1000.0) + 10.0)
        load_failed = "error occurred" in (err1 or "").lower() or "error occurred" in (out1 or "").lower()
        ok1 = (not load_failed) and trace_path.exists() and trace_path.stat().st_size > 0
        rc2, out2, err2, to2, ok2 = 999, "", "", False, False
        if ok1:
            cmd2 = [str(probcli), model_arg, "-trace_replay", "prolog", str(trace_path),
                    "-his", str(states_path), "-his_option", "show_states"]
            rc2, out2, err2, to2 = run_cmd_stream(cmd2, cwd=model_cwd, timeout_s=(tmo / 1000.0) + 10.0)
            load_failed2 = "error occurred" in (err2 or "").lower() or "error occurred" in (out2 or "").lower()
            ok2 = (not load_failed2) and states_path.exists() and states_path.stat().st_size > 0
        last = {"rc_trace": rc1, "rc_replay": rc2, "ok": bool(ok1 and ok2),
                "stdout_trace": out1 or "", "stderr_trace": err1 or "", "stdout_replay": out2 or "", "stderr_replay": err2 or "",
                "timed_out_trace": bool(to1), "timed_out_replay": bool(to2), "attempt": attempt, "timeout_used_ms": tmo}
        if last["ok"]:
            break
    if not trace_path.exists() or trace_path.stat().st_size == 0:
        write_placeholder(trace_path, "NO_TRACE", f"tc={tc.tc_id}\nsequence={tc.sequence}\n{last.get('stderr_trace','')}\n{last.get('stdout_trace','')}")
    if not states_path.exists() or states_path.stat().st_size == 0:
        write_placeholder(states_path, "NO_STATES", f"tc={tc.tc_id}\nsequence={tc.sequence}\n{last.get('stderr_replay','')}\n{last.get('stdout_replay','')}")

    def tr(s, n=6000):
        s = s or ""
        return s if len(s) <= n else s[:n] + "\n...[truncated]..."

    return {"tc_id": tc.tc_id, "sequence": tc.sequence, "trace_path": str(trace_path), "states_path": str(states_path),
            "rc_trace": last["rc_trace"], "rc_replay": last["rc_replay"], "ok": bool(last["ok"]),
            "attempt": int(last.get("attempt", 1)), "timeout_used_ms": int(last.get("timeout_used_ms", timeout_ms)),
            "timed_out_trace": bool(last.get("timed_out_trace", False)), "timed_out_replay": bool(last.get("timed_out_replay", False)),
            "stderr_trace": tr(last.get("stderr_trace", "")), "stderr_replay": tr(last.get("stderr_replay", "")),
            "stdout_trace": tr(last.get("stdout_trace", "")), "stdout_replay": tr(last.get("stdout_replay", ""))}


_STATE_SPLIT_RE = re.compile(r"^\s*State\s+(\d+)\s*[:\-]?\s*$", re.IGNORECASE | re.MULTILINE)
_EVENT_LINE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*(?:\([^)]*\))?)\s*;\s*$", re.MULTILINE)
_VARS_BLOCK_RE = re.compile(r"/\*\s*Variables\s*\n.*?\n\s*\*/", re.DOTALL | re.IGNORECASE)
_TOKEN_RE = re.compile(r"(?P<vars>/\*\s*Variables\s*\n.*?\n\s*\*/)|(?P<event>^\s*[A-Za-z_][A-Za-z0-9_]*(?:\([^)]*\))?\s*;\s*$)",
                       re.MULTILINE | re.DOTALL | re.IGNORECASE)
_ASSIGN_LINE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$")
_ASSIGN_INLINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$")


def dedup_last_per_step(rows):
    keep = {}
    for r in rows:
        keep[int(r.get("step", 0))] = r
    return [keep[k] for k in sorted(keep.keys())]


def parse_states_state_headers(raw, sequence):
    matches = list(_STATE_SPLIT_RE.finditer(raw))
    chunks = []
    if matches:
        for i, mm in enumerate(matches):
            st_idx = int(mm.group(1))
            start = mm.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
            chunks.append((st_idx, raw[start:end]))
    else:
        chunks = [(0, raw)]
    events = [x.strip() for x in sequence.split(";") if x.strip()]
    rows = []
    for st_idx, body in chunks:
        assigns = {}
        for line in body.splitlines():
            mm = _ASSIGN_LINE_RE.match(line)
            if mm:
                assigns[mm.group(1).strip()] = mm.group(2).strip()
        step = st_idx
        ev = "" if step == 0 else (events[step - 1] if (step - 1) < len(events) else "")
        row = {"step": step, "event": ev}
        row.update(assigns)
        rows.append(row)
    rows.sort(key=lambda r: int(r.get("step", 0)))
    return rows


def parse_states_event_variable_blocks(raw):
    step, last_event, rows = 0, "", []
    for m in _TOKEN_RE.finditer(raw):
        if m.group("event"):
            last_event = m.group("event").strip().rstrip(";").strip()
            step += 1
            continue
        vb = m.group("vars")
        inner = re.sub(r"^/\*\s*Variables\s*\n", "", vb, flags=re.IGNORECASE)
        inner = re.sub(r"\n\s*\*/\s*$", "", inner)
        parts = re.split(r"\s*&\s*", inner.strip())
        assigns = {}
        for a in parts:
            a = a.strip()
            if not a:
                continue
            mm = _ASSIGN_INLINE_RE.match(a)
            if mm:
                assigns[mm.group(1).strip()] = mm.group(2).strip()
        ev = "" if step == 0 else last_event
        row = {"step": step if step > 0 else 0, "event": ev}
        row.update(assigns)
        rows.append(row)
    return dedup_last_per_step(rows)


def parse_states_file(path, sequence):
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


def copy_baseline_export(baseline_dir, dest_dir):
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(baseline_dir, dest_dir)


def apply_mutant_to_model(baseline_export_dir, baseline_eventb_name, mutant, out_model_dir):
    copy_baseline_export(baseline_export_dir, out_model_dir)
    model_eventb = out_model_dir / baseline_eventb_name
    text = model_eventb.read_text(encoding="utf-8", errors="replace")
    applied, failed, log_rows = 0, 0, []
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


def harvest_mutant_traces(probcli, model_eventb, tests, mutant, runs_dir, timeout_ms, clpfd):
    all_runs, all_states, parse_fails = [], [], []
    for tc in tests:
        tc_out = runs_dir / tc.tc_id
        res = run_testcase(probcli, model_eventb, tc, tc_out, timeout_ms, clpfd)
        res_row = {"mutant_id": mutant.mutant_id, "tc_id": tc.tc_id, "sequence": tc.sequence,
                   "fault_count": mutant.fault_count, "fault_order": fault_order_label(mutant.fault_count),
                   "fault_components": mutant.fault_components, "mutation_type": mutant.mutation_type,
                   "source_edit_ids": mutant.source_edit_ids}
        res_row.update(res)
        all_runs.append(res_row)
        rows, ok, err = parse_states_file(Path(res["states_path"]), tc.sequence)
        if not rows:
            parse_fails.append({"mutant_id": mutant.mutant_id, "tc_id": tc.tc_id, "states_path": res["states_path"],
                                 "parse_ok": ok, "parse_error": err})
            all_states.append({
                "mutant_id": mutant.mutant_id, "tc_id": tc.tc_id, "step": -1, "event": "__PARSE_FAIL__",
                "fault_count": mutant.fault_count, "fault_order": fault_order_label(mutant.fault_count),
                "fault_components": mutant.fault_components, "mutation_type": mutant.mutation_type,
                "source_edit_ids": mutant.source_edit_ids, "parse_ok": False, "parse_error": err or "parse_failed",
                "states_path": res["states_path"], "ok": bool(res.get("ok", False)),
                "rc_trace": res.get("rc_trace", ""), "rc_replay": res.get("rc_replay", ""),
                "timed_out_trace": bool(res.get("timed_out_trace", False)), "timed_out_replay": bool(res.get("timed_out_replay", False)),
            })
            continue
        for r in rows:
            out = {"mutant_id": mutant.mutant_id, "tc_id": tc.tc_id, "step": int(r.get("step", 0)), "event": str(r.get("event", "")),
                   "fault_count": mutant.fault_count, "fault_order": fault_order_label(mutant.fault_count),
                   "fault_components": mutant.fault_components, "mutation_type": mutant.mutation_type,
                   "source_edit_ids": mutant.source_edit_ids, "parse_ok": True, "parse_error": "", "states_path": res["states_path"]}
            for k, v in r.items():
                if k not in ("step", "event"):
                    out[k] = v
            all_states.append(out)
    return pd.DataFrame(all_runs), pd.DataFrame(all_states), pd.DataFrame(parse_fails)


def safe_str(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x)


def build_samples_from_states(df_states):
    required = {"mutant_id", "tc_id", "step", "event"}
    miss = required - set(df_states.columns)
    if miss:
        raise ValueError(f"expanded states missing columns: {sorted(miss)}")
    df_states = df_states.copy()
    df_states["step"] = pd.to_numeric(df_states["step"], errors="coerce").fillna(-1).astype(int)
    non_var = {"mutant_id", "tc_id", "step", "event", "parse_ok", "parse_error", "states_path",
               "mutation_type", "fault_count", "fault_order", "fault_components", "threat_class",
               "edit_count", "edit_targets"}
    var_cols = [c for c in df_states.columns if c not in non_var]
    sample_keys, sparse_dicts, meta_rows = [], [], []
    for (mid, tc), g in df_states.groupby(["mutant_id", "tc_id"], sort=False):
        g = g.sort_values("step", ascending=True)
        feats = {}
        if "parse_ok" in g.columns and g["parse_ok"].astype(str).iloc[0].lower() == "false":
            feats["RUN_PARSE_FAIL"] = 1.0
        if "no_trace" in g.columns and int(pd.to_numeric(g["no_trace"], errors="coerce").fillna(0).iloc[0]) == 1:
            feats["RUN_NO_TRACE"] = 1.0
        for ev in g["event"].astype(str).tolist():
            ev = ev.strip()
            if ev:
                feats[f"EV={ev}"] = feats.get(f"EV={ev}", 0.0) + 1.0
        for _, row in g.iterrows():
            for c in var_cols:
                v = safe_str(row.get(c, "")).strip()
                if v:
                    key = f"{c}={v}"
                    feats[key] = feats.get(key, 0.0) + 1.0
        prev = None
        for _, row in g.iterrows():
            if prev is None:
                prev = row
                continue
            for c in var_cols:
                a = safe_str(prev.get(c, "")).strip()
                b = safe_str(row.get(c, "")).strip()
                if a and b and a != b:
                    dk = f"DELTA:{c}:{a}->{b}"
                    feats[dk] = feats.get(dk, 0.0) + 1.0
            prev = row
        sample_keys.append(f"{mid}|{tc}")
        sparse_dicts.append(feats)
        meta_rows.append({"mutant_id": str(mid), "tc_id": str(tc)})
    return sample_keys, sparse_dicts, meta_rows


def make_labels(df_states, task):
    if "fault_components" not in df_states.columns:
        raise ValueError("expanded dataset must contain fault_components")
    if "fault_count" not in df_states.columns:
        df_states = df_states.copy()
        if "fault_order" in df_states.columns:
            df_states["fault_count"] = df_states["fault_order"].map(lambda s: {"single": 1, "double": 2, "triple": 3}.get(str(s), np.nan))
    lab = {}
    for (mid, tc), g in df_states.groupby(["mutant_id", "tc_id"], sort=False):
        comps = safe_str(g["fault_components"].iloc[0]).strip()
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
            raise ValueError(f"unknown task {task}")
    return lab


def train_sklearn_models(X, y, seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import StratifiedKFold
    models = {}
    models["logreg"] = LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed).fit(X, y)
    models["nb"] = MultinomialNB().fit(X, y)
    svm = LinearSVC(class_weight="balanced", random_state=seed)
    try:
        classes, counts = np.unique(y, return_counts=True)
        n_splits = min(3, int(counts.min()) if len(counts) else 0)
        if n_splits < 2:
            raise ValueError("too few samples per class")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        models["svm_cal"] = CalibratedClassifierCV(svm, cv=cv, method="sigmoid").fit(X, y)
    except Exception:
        models["svm"] = svm.fit(X, y)
    models["rf"] = RandomForestClassifier(n_estimators=400, random_state=seed, class_weight="balanced_subsample", n_jobs=-1).fit(X, y)
    models["gb"] = GradientBoostingClassifier(n_estimators=200, random_state=seed, max_depth=5).fit(X, y)
    try:
        import xgboost as xgb
        models["xgboost"] = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=seed,
                                               eval_metric="mlogloss", n_jobs=-1).fit(X, y)
    except ImportError:
        pass
    return models


def eval_model(est, X_test, y_test):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
    y_pred = est.predict(X_test)
    return {
        "acc": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "report": classification_report(y_test, y_pred, zero_division=0),
        "cm": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
    }


def edit_signature(e):
    s = f"{e.location}||{normalize_eventb_expr(e.original)}||{normalize_eventb_expr(e.mutated)}"
    return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()


def build_edit_pool(mutants):
    pool, seen = [], set()
    for m in mutants:
        for e in m.edits:
            sig = edit_signature(e)
            if sig in seen:
                continue
            seen.add(sig)
            pool.append((m.mutant_id, e, infer_component_from_location(e.location)))
    return pool


def pick_underrepresented_targets(task, counts):
    if not counts:
        return ["GUI", "ALRM", "CTRL"]
    if task == "single":
        keys = [k for k in ["GUI", "ALRM", "CTRL"] if k in counts] or list(counts.keys())
        return sorted(keys, key=lambda k: counts.get(k, 0))
    return sorted(counts.keys(), key=lambda k: counts.get(k, 0))


def propose_mutants_from_pool(pool, existing_signatures, round_idx, n_new, fault_orders, task, class_counts, seed):
    rng = random.Random(seed + 1000 * round_idx)
    targets_ranked = pick_underrepresented_targets(task, class_counts)
    by_comp = {}
    for src_id, e, comp in pool:
        by_comp.setdefault(comp, []).append((src_id, e))

    def sample_edit_for(comp):
        cand = by_comp.get(comp, [])
        return rng.choice(cand) if cand else None

    def combo_to_components(combo, k):
        if "+" in combo:
            return [c.strip() for c in combo.split("+") if c.strip()]
        if k == 1:
            return [combo]
        comps = [combo]
        while len(comps) < k:
            comps.append(rng.choice(["GUI", "ALRM", "CTRL"]))
        return comps

    out, used_in_round, i, tries = [], set(), 0, 0
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
        chosen = []
        for c in comps:
            s = sample_edit_for(c)
            if s is None:
                break
            chosen.append((s[0], s[1], c))
        if len(chosen) != k:
            continue
        sig_key = "|".join(sorted(edit_signature(ed) for _, ed, _ in chosen))
        if sig_key in existing_signatures or sig_key in used_in_round:
            continue
        new_id = f"AL_R{round_idx:02d}_{i+1:04d}"
        edits = [dataclasses.replace(ed, mutation_type="ACTIVE_LEARNING") for _, ed, _ in chosen]
        pm = ProposedMutant(mutant_id=new_id, edits=edits, fault_count=k,
                             fault_components=normalize_combo([c for _, _, c in chosen]),
                             mutation_type="ACTIVE_LEARNING",
                             source_edit_ids=",".join(src for src, _, _ in chosen))
        out.append(pm)
        used_in_round.add(sig_key)
        existing_signatures.add(sig_key)
        i += 1
    return out


def mutant_meta_features(pm):
    feats = {f"k={pm.fault_count}": 1.0, f"combo={pm.fault_components}": 1.0}
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


def train_selector_model(executed_meta, seed=42):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    if executed_meta.empty or not {"label", "feat_json"}.issubset(set(executed_meta.columns)):
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
    clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed).fit(X, y)
    return {"vectorizer": vec, "model": clf}


def entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def margin(p):
    if p.size == 0:
        return 0.0
    s = np.sort(p)[::-1]
    return 1.0 if len(s) == 1 else float(s[0] - s[1])


def score_uncertainty(selector, cand_feats, mode):
    if selector is None:
        return 0.0
    X = selector["vectorizer"].transform([cand_feats])
    proba = selector["model"].predict_proba(X)[0]
    if mode == "entropy":
        return entropy(proba)
    if mode == "margin":
        return -margin(proba)
    return 0.0


def jaccard_distance(a, b):
    if not a and not b:
        return 0.0
    uni = len(a | b)
    return 1.0 - (len(a & b) / uni) if uni else 0.0


def cand_token_set(pm):
    toks = set()
    for e in pm.edits:
        for tok in re.split(r"[^A-Za-z0-9_]+", e.location or ""):
            tok = tok.strip()
            if tok:
                toks.add(tok)
    toks.add(pm.fault_components)
    toks.add(f"k{pm.fault_count}")
    return toks


def select_mutants(candidates, n_select, strategy, class_counts, selector_model, seed):
    rng = random.Random(seed)
    if not candidates:
        return []
    cands = candidates[:]
    rng.shuffle(cands)
    if strategy == "random":
        return cands[:n_select]
    if strategy in ("uncertainty_entropy", "uncertainty_margin"):
        mode = "entropy" if strategy.endswith("entropy") else "margin"
        scored = [(score_uncertainty(selector_model, mutant_meta_features(pm), mode), pm) for pm in cands]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [pm for _, pm in scored[:n_select]]
    if strategy == "diversity_jaccard":
        picked = [cands.pop(0)]
        picked_sets = [cand_token_set(picked[0])]
        while cands and len(picked) < n_select:
            best_i, best_d = 0, -1.0
            for i, pm in enumerate(cands):
                d = min(jaccard_distance(cand_token_set(pm), ps) for ps in picked_sets)
                if d > best_d:
                    best_d, best_i = d, i
            pm = cands.pop(best_i)
            picked.append(pm)
            picked_sets.append(cand_token_set(pm))
        return picked
    return cands[:n_select]


def compute_class_counts(labels):
    out = {}
    for y in labels:
        out[y] = out.get(y, 0) + 1
    return out


def load_or_init_dataset(seed_dataset_xlsx):
    if seed_dataset_xlsx and seed_dataset_xlsx.exists():
        xl = pd.ExcelFile(seed_dataset_xlsx)
        sheet = "states_expanded" if "states_expanded" in xl.sheet_names else xl.sheet_names[0]
        return pd.read_excel(seed_dataset_xlsx, sheet_name=sheet)
    return pd.DataFrame()


def safe_write_excel_multi(path, sheets):
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for name, df in sheets.items():
            for c in df.columns:
                if df[c].dtype == "object":
                    df[c] = df[c].map(clean_xlsx)
            df.to_excel(w, index=False, sheet_name=name)


def run_active_learning(args):
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tests = read_tests_xlsx(Path(args.tests_xlsx).resolve(), TEST_XLSX_MAPPING)
    mutants, mutants_raw_df = read_mutants_xlsx(Path(args.mutants_xlsx).resolve(), MUTANT_XLSX_MAPPING)
    pool = build_edit_pool(mutants)
    seed_dataset = Path(args.seed_dataset_xlsx).resolve() if args.seed_dataset_xlsx else None
    df_states = load_or_init_dataset(seed_dataset)

    if not df_states.empty and "fault_components" not in df_states.columns and "ID" in mutants_raw_df.columns:
        meta = mutants_raw_df.copy()
        meta["mutant_id"] = meta["ID"].astype(str)
        keep_cols = ["mutant_id"] + [c for c in ["Fault_Count", "Fault_Components", "Mutation_Type", "Threat_Class"] if c in meta.columns]
        meta = meta[keep_cols].drop_duplicates("mutant_id")
        df_states = df_states.merge(meta, on="mutant_id", how="left")
        if "Fault_Count" in df_states.columns and "fault_count" not in df_states.columns:
            df_states["fault_count"] = pd.to_numeric(df_states["Fault_Count"], errors="coerce")
        if "Fault_Components" in df_states.columns and "fault_components" not in df_states.columns:
            df_states["fault_components"] = df_states["Fault_Components"].astype(str)

    if not df_states.empty and not args.include_base:
        df_states = df_states[df_states["mutant_id"].astype(str).str.upper() != "BASE"].copy()

    master_xlsx = out_dir / "active_learning_dataset.xlsx"
    history_rows, proposed_rows, mutate_log_rows, parse_fail_rows = [], [], [], []
    run_rows_accum, new_states_accum = [], []
    existing_sig_set = set()
    executed_meta_path = out_dir / "executed_mutants_meta.csv"

    strategy = args.strategy_name
    cand_mult = args.candidate_multiplier
    probcli = Path(args.probcli).resolve()
    baseline_export_dir = Path(args.baseline_export_dir).resolve()
    baseline_eventb_name = args.baseline_eventb_name
    clpfd = not args.no_clpfd
    fault_orders = [int(x) for x in str(args.fault_orders).split(",") if x.strip()]

    dev_pool_size = args.development_pool_size or 3596
    percentages = [float(p) for p in str(args.percentages).split(",") if p.strip()]
    checkpoint_counts = [max(1, math.ceil(dev_pool_size * p / 100.0)) for p in percentages]

    total_selected = 0
    checkpoint_idx = 0

    for r in range(1, args.rounds + 1):
        round_dir = out_dir / f"round_{r:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        best_model_name, best_metrics, class_counts = "none", {"acc": np.nan, "f1_macro": np.nan, "precision": np.nan, "recall": np.nan}, {}

        if not df_states.empty:
            try:
                keys, feat_dicts, _ = build_samples_from_states(df_states)
                label_map = make_labels(df_states, args.task)
                kept_idx = [i for i, k in enumerate(keys) if k in label_map]
                if len(kept_idx) >= 20:
                    keys2 = [keys[i] for i in kept_idx]
                    feats2 = [feat_dicts[i] for i in kept_idx]
                    y = [label_map[k] for k in keys2]
                    class_counts = compute_class_counts(y)
                    from sklearn.feature_extraction import DictVectorizer
                    from sklearn.model_selection import train_test_split
                    vec = DictVectorizer(sparse=True)
                    X = vec.fit_transform(feats2)
                    strat = y if len(set(y)) > 1 else None
                    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=args.seed + r, stratify=strat)
                    models = train_sklearn_models(Xtr, ytr, seed=args.seed + r)
                    best_est, best_model_name, best_metrics = None, None, None
                    for name, est in models.items():
                        m = eval_model(est, Xte, yte)
                        if best_metrics is None or m["f1_macro"] > best_metrics["f1_macro"]:
                            best_metrics, best_model_name, best_est = m, name, est
                    best_metrics = best_metrics or {"acc": np.nan, "f1_macro": np.nan, "precision": np.nan, "recall": np.nan}
            except Exception as e:
                print(f"round {r} training failed: {e}")

        history_rows.append({
            "round": r, "task": args.task, "strategy": strategy, "best_model": best_model_name,
            "acc": float(best_metrics.get("acc", np.nan)), "precision": float(best_metrics.get("precision", np.nan)),
            "recall": float(best_metrics.get("recall", np.nan)), "f1_macro": float(best_metrics.get("f1_macro", np.nan)),
            "total_selected": total_selected, "percent_of_pool": 100.0 * total_selected / dev_pool_size,
            "class_counts_json": json.dumps(class_counts, ensure_ascii=False),
        })

        if checkpoint_idx < len(checkpoint_counts) and total_selected >= checkpoint_counts[checkpoint_idx]:
            print(f"reached checkpoint {percentages[checkpoint_idx]}% ({total_selected} candidates), reporting scores")
            checkpoint_idx += 1
            if checkpoint_idx >= len(checkpoint_counts):
                print("all requested percentage checkpoints reached, stopping")
                break

        n_candidates = max(args.new_mutants_per_round * max(1, cand_mult), args.new_mutants_per_round)
        candidates = propose_mutants_from_pool(pool, existing_sig_set, r, n_candidates, fault_orders, args.task, class_counts, args.seed)
        if not candidates:
            print(f"round {r}: could not generate candidates, stopping")
            break

        executed_meta = pd.read_csv(executed_meta_path) if executed_meta_path.exists() else pd.DataFrame()
        selector_model = train_selector_model(executed_meta, seed=args.seed + r)
        selected_mutants = select_mutants(candidates, args.new_mutants_per_round, strategy, class_counts, selector_model, args.seed + 100 * r)
        total_selected += len(selected_mutants)

        for pm in selected_mutants:
            proposed_rows.append({"round": r, "strategy": strategy, "mutant_id": pm.mutant_id, "fault_count": pm.fault_count,
                                   "fault_order": fault_order_label(pm.fault_count), "fault_components": pm.fault_components,
                                   "source_edit_ids": pm.source_edit_ids, "edit_locations": " || ".join(e.location for e in pm.edits)})

        for pm in selected_mutants:
            print(f"round {r}: running {pm.mutant_id} ({pm.fault_components}, k={pm.fault_count})")
            mutant_root = round_dir / pm.mutant_id
            model_dir, runs_dir = mutant_root / "model", mutant_root / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            model_eventb, applied, failed, mlog = apply_mutant_to_model(baseline_export_dir, baseline_eventb_name, pm, model_dir)
            for row in mlog:
                row["round"] = r
                row["strategy"] = strategy
                mutate_log_rows.append(row)
            runs_df, states_df, pf_df = harvest_mutant_traces(probcli, model_eventb, tests, pm, runs_dir, args.timeout_ms, clpfd)
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
            lbl = pm.fault_components.split("+")[0] if (args.task == "single" and pm.fault_components) else (pm.fault_components or "UNKNOWN")
            meta_row = {"round": r, "strategy": strategy, "mutant_id": pm.mutant_id, "label": lbl,
                        "fault_components": pm.fault_components, "fault_count": pm.fault_count,
                        "feat_json": json.dumps(feat, ensure_ascii=False), "edit_locations": " || ".join(e.location for e in pm.edits),
                        "applied_edits": int(applied), "failed_edits": int(failed),
                        "runs_ok_rate": float(runs_df["ok"].mean()) if (not runs_df.empty and "ok" in runs_df.columns) else np.nan}
            if executed_meta_path.exists():
                newm = pd.concat([pd.read_csv(executed_meta_path), pd.DataFrame([meta_row])], ignore_index=True)
            else:
                newm = pd.DataFrame([meta_row])
            newm.to_csv(executed_meta_path, index=False)

        if new_states_accum:
            df_new = pd.concat(new_states_accum, ignore_index=True)
            if "mutant_id" not in df_new.columns or "tc_id" not in df_new.columns:
                raise RuntimeError("harvest produced invalid states rows")
            df_states = df_new.copy() if df_states.empty else pd.concat([df_states, df_new], ignore_index=True)
            new_states_accum = []

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

    safe_write_excel_multi(master_xlsx, {
        "states_expanded": df_states_final, "runs": df_runs, "proposed_mutants": df_proposed,
        "mutate_log": df_mutlog, "parse_fails": df_parsefail, "training_history": df_history,
    })
    df_history.to_csv(out_dir / "training_history.csv", index=False)
    df_proposed.to_csv(out_dir / "proposed_mutants.csv", index=False)
    print(f"done, wrote {master_xlsx}")
    print(f"states rows: {len(df_states_final)}, proposed mutants: {len(df_proposed)}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probcli", required=True)
    ap.add_argument("--baseline_export_dir", required=True)
    ap.add_argument("--baseline_eventb_name", default="Ventilator_Composed_mch.eventb")
    ap.add_argument("--tests_xlsx", required=True)
    ap.add_argument("--mutants_xlsx", required=True)
    ap.add_argument("--seed_dataset_xlsx", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--timeout_ms", type=int, default=DEFAULT_TIMEOUT_MS)
    ap.add_argument("--no_clpfd", action="store_true")
    ap.add_argument("--task", default="any", choices=["single", "double", "triple", "any"])
    ap.add_argument("--include_base", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--new_mutants_per_round", type=int, default=3)
    ap.add_argument("--fault_orders", default="2,3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strategy_name", default="random",
                     choices=["random", "uncertainty_entropy", "uncertainty_margin", "diversity_jaccard"])
    ap.add_argument("--candidate_multiplier", type=int, default=10)
    ap.add_argument("--percentages", default="4,8,16")
    ap.add_argument("--development_pool_size", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    run_active_learning(args)


if __name__ == "__main__":
    main()
