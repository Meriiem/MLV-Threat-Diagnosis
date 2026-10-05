from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold

COMPONENTS = ["ALRM", "CTRL", "GUI"]
CATEGORICAL_VARS = [
    "alarm_state",
    "alarm_type",
    "controller_state",
    "gui_state",
    "is_alarm_displayed",
    "is_gui_frozen",
    "is_ventilating",
    "vent_mode",
]
NUMERIC_VARS = [
    "battery_level",
    "ie_ratio",
    "insp_pressure",
    "pressure_reading",
    "resp_rate",
    "temperature_reading",
    "voltage_level",
]


def clean(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def split_set(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"\s*\+\s*", clean(value)) if part.strip()]


def aggregate_trace(group: pd.DataFrame) -> dict[str, float]:
    frame = group.sort_values("step") if "step" in group.columns else group
    features: dict[str, float] = {}
    if "event" in frame.columns:
        for event in frame["event"].map(clean):
            if event:
                features[f"EV:{event}"] = 1.0
                features[f"EV_COUNT:{event}"] = features.get(f"EV_COUNT:{event}", 0.0) + 1.0
    for column in CATEGORICAL_VARS:
        if column not in frame.columns:
            continue
        for value in frame[column].map(clean):
            if value:
                features[f"{column}={value}"] = 1.0
    for column in NUMERIC_VARS:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if len(values):
            features[column] = float(values.iloc[-1])
            features[f"{column}:min"] = float(values.min())
            features[f"{column}:max"] = float(values.max())
            features[f"{column}:range"] = float(values.max() - values.min())
    if "ok" in frame.columns:
        features["RUN_OK"] = float(str(frame["ok"].iloc[0]).lower() in {"1", "true"})
    if "parse_ok" in frame.columns:
        features["PARSE_OK"] = float(str(frame["parse_ok"].iloc[0]).lower() in {"1", "true"})
    if "no_trace" in frame.columns:
        features["NO_TRACE"] = float(pd.to_numeric(frame["no_trace"], errors="coerce").fillna(0).iloc[0] > 0)
    return features


def build_trace_table(states: pd.DataFrame, label_lookup: dict[str, dict[str, str]]) -> pd.DataFrame:
    rows = []
    states = states.copy()
    states["mutant_id"] = states["mutant_id"].map(clean)
    states["tc_id"] = states["tc_id"].map(clean)
    if "step" in states.columns:
        states["step"] = pd.to_numeric(states["step"], errors="coerce").fillna(-1).astype(int)
    for (mutant_id, tc_id), group in states.groupby(["mutant_id", "tc_id"], sort=False):
        if mutant_id not in label_lookup:
            continue
        labels = label_lookup[mutant_id]
        rows.append(
            {
                "mutant_id": mutant_id,
                "tc_id": tc_id,
                "features": aggregate_trace(group),
                "component_set": labels["component_set"],
                "threat_family_set": labels["threat_family_set"],
                "fault_order": labels["fault_order"],
            }
        )
    return pd.DataFrame(rows)


def label_matrix(values: Iterable[str], classes: list[str]) -> np.ndarray:
    index = {name: position for position, name in enumerate(classes)}
    matrix = np.zeros((len(values), len(classes)), dtype=int)
    for row_index, value in enumerate(values):
        for token in split_set(value):
            if token in index:
                matrix[row_index, index[token]] = 1
    return matrix


@dataclass
class BinaryModel:
    constant: int | None
    model: object | None


class MultiBinaryClassifier:
    def __init__(self, seed: int, n_estimators: int = 120):
        self.seed = seed
        self.n_estimators = n_estimators
        self.models: list[BinaryModel] = []

    def fit(self, x, y: np.ndarray) -> "MultiBinaryClassifier":
        self.models = []
        for column in range(y.shape[1]):
            target = y[:, column]
            values = np.unique(target)
            if len(values) == 1:
                self.models.append(BinaryModel(int(values[0]), None))
                continue
            model = LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=-1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=self.seed + column,
                class_weight="balanced",
                verbosity=-1,
                n_jobs=-1,
            )
            model.fit(x, target)
            self.models.append(BinaryModel(None, model))
        return self

    def predict_proba(self, x) -> np.ndarray:
        probabilities = np.zeros((x.shape[0], len(self.models)), dtype=float)
        for column, item in enumerate(self.models):
            if item.constant is not None:
                probabilities[:, column] = float(item.constant)
            else:
                probabilities[:, column] = item.model.predict_proba(x)[:, 1]
        return probabilities

    def predict(self, x, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(x) >= threshold).astype(int)


def multi_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "exact_match": float(accuracy_score(y_true, y_pred)),
    }


def fit_component_threat_models(
    x_train,
    component_train: np.ndarray,
    threat_train: np.ndarray,
    groups: np.ndarray,
    seed: int,
    task: str,
) -> dict[str, object]:
    started = time.perf_counter()
    component_model = MultiBinaryClassifier(seed).fit(x_train, component_train)
    threat_model = None
    if task == "threat":
        threat_model = MultiBinaryClassifier(seed + 100).fit(x_train, threat_train)
    if task == "joint":
        unique_groups = np.unique(groups)
        folds = min(5, len(unique_groups))
        if folds >= 2:
            oof = np.zeros((x_train.shape[0], component_train.shape[1]), dtype=float)
            splitter = GroupKFold(n_splits=folds)
            for fold, (train_indices, valid_indices) in enumerate(splitter.split(np.zeros(len(groups)), groups=groups)):
                fold_model = MultiBinaryClassifier(seed + 1000 + fold).fit(x_train[train_indices], component_train[train_indices])
                oof[valid_indices] = fold_model.predict_proba(x_train[valid_indices])
        else:
            oof = component_model.predict_proba(x_train)
        x_augmented = np.hstack([x_train.toarray() if hasattr(x_train, "toarray") else x_train, oof])
        threat_model = MultiBinaryClassifier(seed + 200).fit(x_augmented, threat_train)
    return {
        "component_model": component_model,
        "threat_model": threat_model,
        "train_seconds": float(time.perf_counter() - started),
        "task": task,
    }


def evaluate_models(models: dict[str, object], x_test, component_test: np.ndarray, threat_test: np.ndarray) -> dict[str, float]:
    started = time.perf_counter()
    task = str(models["task"])
    component_model = models["component_model"]
    component_probabilities = component_model.predict_proba(x_test)
    component_predictions = (component_probabilities >= 0.5).astype(int)
    component_results = multi_metrics(component_test, component_predictions)
    threat_results = {key: math.nan for key in component_results}
    if task == "threat":
        threat_predictions = models["threat_model"].predict(x_test)
        threat_results = multi_metrics(threat_test, threat_predictions)
    if task == "joint":
        x_augmented = np.hstack([x_test.toarray() if hasattr(x_test, "toarray") else x_test, component_probabilities])
        threat_predictions = models["threat_model"].predict(x_augmented)
        threat_results = multi_metrics(threat_test, threat_predictions)
    if task == "component":
        primary = component_results["f1_macro"]
    elif task == "threat":
        primary = threat_results["f1_macro"]
    else:
        primary = (component_results["f1_macro"] + threat_results["f1_macro"]) / 2.0
    return {
        "primary_f1": float(primary),
        "component_f1_macro": float(component_results["f1_macro"]),
        "component_f1_micro": float(component_results["f1_micro"]),
        "component_exact_match": float(component_results["exact_match"]),
        "threat_f1_macro": float(threat_results["f1_macro"]),
        "threat_f1_micro": float(threat_results["f1_micro"]),
        "threat_exact_match": float(threat_results["exact_match"]),
        "infer_seconds": float(time.perf_counter() - started),
    }


def vectorize_train_test(train_features: list[dict[str, float]], test_features: list[dict[str, float]]):
    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform(train_features)
    x_test = vectorizer.transform(test_features)
    return vectorizer, x_train, x_test


def metadata_features(row: pd.Series) -> dict[str, float]:
    features: dict[str, float] = {f"fault_count={int(row['fault_count'])}": 1.0}
    for field, prefix in [("edit_targets", "target"), ("events", "event"), ("mutation_kinds", "kind")]:
        for value in [part.strip() for part in clean(row.get(field, "")).split("||") if part.strip()]:
            features[f"{prefix}={value}"] = features.get(f"{prefix}={value}", 0.0) + 1.0
            for token in re.split(r"[^A-Za-z0-9_]+", value):
                if token:
                    features[f"token={token}"] = features.get(f"token={token}", 0.0) + 1.0
    return features


def candidate_tokens(row: pd.Series) -> set[str]:
    tokens = {f"k={int(row['fault_count'])}"}
    for field in ["edit_targets", "events", "mutation_kinds"]:
        for value in [part.strip() for part in clean(row.get(field, "")).split("||") if part.strip()]:
            tokens.add(value)
            tokens.update(token for token in re.split(r"[^A-Za-z0-9_]+", value) if token)
    return tokens


def jaccard_distance(left: set[str], right: set[str]) -> float:
    union = left | right
    return 0.0 if not union else 1.0 - len(left & right) / len(union)


def selector_probabilities(labeled: pd.DataFrame, candidates: pd.DataFrame, task: str, seed: int, component_classes: list[str], threat_classes: list[str]) -> np.ndarray:
    vectorizer = DictVectorizer(sparse=True)
    x_labeled = vectorizer.fit_transform([metadata_features(row) for _, row in labeled.iterrows()])
    x_candidates = vectorizer.transform([metadata_features(row) for _, row in candidates.iterrows()])
    component_labels = label_matrix(labeled["component_set"].tolist(), component_classes)
    threat_labels = label_matrix(labeled["threat_family_set"].tolist(), threat_classes)
    if task == "component":
        labels = component_labels
    elif task == "threat":
        labels = threat_labels
    else:
        labels = np.hstack([component_labels, threat_labels])
    model = MultiBinaryClassifier(seed, n_estimators=80).fit(x_labeled, labels)
    return model.predict_proba(x_candidates)


def select_candidates(
    labeled: pd.DataFrame,
    candidates: pd.DataFrame,
    count: int,
    strategy: str,
    task: str,
    seed: int,
    component_classes: list[str],
    threat_classes: list[str],
) -> list[str]:
    if count <= 0 or candidates.empty:
        return []
    count = min(count, len(candidates))
    if strategy == "random":
        return candidates.sample(n=count, random_state=seed)["candidate_id"].tolist()
    if strategy == "diversity_jaccard":
        available = candidates.copy()
        reference_sets = [candidate_tokens(row) for _, row in labeled.iterrows()]
        selected = []
        selected_sets = []
        while len(selected) < count and not available.empty:
            best_index = None
            best_score = -1.0
            for index, row in available.iterrows():
                tokens = candidate_tokens(row)
                references = reference_sets + selected_sets
                score = 1.0 if not references else min(jaccard_distance(tokens, reference) for reference in references)
                if score > best_score:
                    best_score = score
                    best_index = index
            row = available.loc[best_index]
            selected.append(str(row["candidate_id"]))
            selected_sets.append(candidate_tokens(row))
            available = available.drop(index=best_index)
        return selected
    probabilities = selector_probabilities(labeled, candidates, task, seed, component_classes, threat_classes)
    clipped = np.clip(probabilities, 1e-9, 1.0 - 1e-9)
    if strategy == "uncertainty_entropy":
        scores = np.mean(-(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped)), axis=1)
    elif strategy == "uncertainty_margin":
        scores = -np.mean(np.abs(clipped - 0.5), axis=1)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    order = np.argsort(-scores)[:count]
    return candidates.iloc[order]["candidate_id"].tolist()


def stabilization_budget(history: pd.DataFrame, tolerance: float = 0.005, window: int = 2) -> float:
    if len(history) < window + 1:
        return math.nan
    values = history.sort_values("budget_pct")["internal_cv_f1"].tolist()
    budgets = history.sort_values("budget_pct")["budget_pct"].tolist()
    for index in range(window, len(values)):
        previous = values[index - window:index]
        if max(abs(values[index] - value) for value in previous) <= tolerance:
            return float(budgets[index])
    return math.nan


def internal_group_cv(
    table: pd.DataFrame,
    component_classes: list[str],
    threat_classes: list[str],
    task: str,
    seed: int,
) -> float:
    groups = table["mutant_id"].to_numpy()
    unique_groups = np.unique(groups)
    folds = min(3, len(unique_groups))
    if folds < 2:
        return math.nan
    splitter = GroupKFold(n_splits=folds)
    scores = []
    for fold, (train_indices, valid_indices) in enumerate(splitter.split(np.zeros(len(table)), groups=groups)):
        train = table.iloc[train_indices]
        valid = table.iloc[valid_indices]
        _, x_train, x_valid = vectorize_train_test(train["features"].tolist(), valid["features"].tolist())
        comp_train = label_matrix(train["component_set"].tolist(), component_classes)
        comp_valid = label_matrix(valid["component_set"].tolist(), component_classes)
        threat_train = label_matrix(train["threat_family_set"].tolist(), threat_classes)
        threat_valid = label_matrix(valid["threat_family_set"].tolist(), threat_classes)
        models = fit_component_threat_models(x_train, comp_train, threat_train, train["mutant_id"].to_numpy(), seed + fold, task)
        scores.append(evaluate_models(models, x_valid, comp_valid, threat_valid)["primary_f1"])
    return float(np.mean(scores))


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
