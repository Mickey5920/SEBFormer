"""Training-fitted baseline encoding and 128-event input tensors."""

import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

VERSION = "train_only_baseline_v1"
GROUPS = (
    "demo",
    "physical",
    "biomarkers",
    "lifestyle",
    "mental",
    "repro",
    "environmental",
    "genetic",
    "other",
)


def strata_keys(demo, physical):
    sex = demo["sex"].fillna("MISSING").astype(str)
    age = pd.to_numeric(demo["age"], errors="coerce")
    bmi = pd.to_numeric(physical["bmi"], errors="coerce")
    agebin = (
        pd.cut(
            age,
            [-np.inf, 35, 45, 55, 65, 75, 85, np.inf],
            right=False,
            labels=["lt35", "35-44", "45-54", "55-64", "65-74", "75-84", "ge85"],
        )
        .astype("string")
        .fillna("MISSING")
    )
    bmibin = (
        pd.cut(
            bmi,
            [-np.inf, 0, 18.5, 25, 30, 100, np.inf],
            right=False,
            labels=[
                "invalid_negative",
                "underweight",
                "normal",
                "overweight",
                "obese",
                "ge100",
            ],
        )
        .astype("string")
        .fillna("MISSING")
    )
    two = sex + "|" + agebin
    three = two + "|" + bmibin
    return two.reset_index(drop=True), three.reset_index(drop=True)


def numeric_series(raw):
    return (
        pd.to_numeric(raw, errors="coerce")
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
    )


def fill_numeric(series, spec, strata2=None, strata3=None):
    s = series.copy()
    if spec["strata3"]:
        s = s.fillna(strata3.map(spec["strata3"]))
    if spec["strata2"]:
        s = s.fillna(strata2.map(spec["strata2"]))
    return s.fillna(spec["median"])


def fit_frame(frame, train_idx, group, strata2=None, strata3=None):
    train_idx = np.asarray(train_idx, dtype=np.int64)
    if len(train_idx) == 0 or len(np.unique(train_idx)) != len(train_idx):
        raise ValueError("Invalid training row indices")
    if np.any(train_idx < 0) or np.any(train_idx >= len(frame)):
        raise ValueError("Training indices out of bounds")
    fit = frame.iloc[train_idx].reset_index(drop=True)
    two = None if strata2 is None else strata2.iloc[train_idx].reset_index(drop=True)
    three = None if strata3 is None else strata3.iloc[train_idx].reset_index(drop=True)
    spec = {
        "version": VERSION,
        "group": group,
        "n_fit": len(fit),
        "columns": [],
        "feature_names": [],
    }
    for col in fit:
        raw = fit[col]
        num = numeric_series(raw)
        nonnull = int(raw.notna().sum())
        nnum = int(num.notna().sum())
        is_numeric = nonnull == 0 or nnum >= max(1, int(nonnull * 0.95))
        if is_numeric:
            med = float(num.median()) if nnum else 0.0
            rule = {
                "name": col,
                "kind": "numeric",
                "median": med,
                "all_missing_in_train": nnum == 0,
                "strata2": {},
                "strata3": {},
                "clip": None,
            }
            if group in {"physical", "biomarkers"}:
                if two is None or three is None:
                    raise ValueError("Missing stratification covariates")
                rule["strata3"] = {
                    str(k): float(v)
                    for k, v in num.groupby(three).median().dropna().items()
                }
                rule["strata2"] = {
                    str(k): float(v)
                    for k, v in num.groupby(two).median().dropna().items()
                }
                filled = fill_numeric(num, rule, two, three)
                lo, hi = np.quantile(filled.to_numpy(), [0.01, 0.99])
                if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                    rule["clip"] = [float(lo), float(hi)]
            spec["columns"].append(rule)
            spec["feature_names"].extend([col, col + "__missing"])
        else:
            categories = sorted(raw[raw.notna()].astype(str).unique())
            spec["columns"].append(
                {
                    "name": col,
                    "kind": "categorical",
                    "categories": categories,
                    "missing_code": 0,
                    "unknown_code": 1,
                }
            )
            spec["feature_names"].append(col)
    return spec


def transform_frame(frame, spec, strata2=None, strata3=None):
    if list(frame.columns) != [s["name"] for s in spec["columns"]]:
        raise ValueError("Feature schema/order changed")
    output = np.empty((len(frame), len(spec["feature_names"])), dtype=np.float32)
    j = 0
    for rule in spec["columns"]:
        raw = frame[rule["name"]]
        if rule["kind"] == "numeric":
            num = numeric_series(raw)
            missing = num.isna()
            filled = fill_numeric(num, rule, strata2, strata3)
            if rule["clip"] is not None:
                filled = filled.clip(*rule["clip"])
            output[:, j] = filled.to_numpy(dtype=np.float32)
            output[:, j + 1] = missing.to_numpy(dtype=np.float32)
            j += 2
        else:
            codes = {v: i + 2 for i, v in enumerate(rule["categories"])}
            encoded = raw.astype(str).map(codes).fillna(rule["unknown_code"])
            encoded = encoded.where(raw.notna(), rule["missing_code"])
            output[:, j] = encoded.to_numpy(dtype=np.float32)
            j += 1
    if not np.isfinite(output).all():
        raise ValueError("Preprocessing produced nonfinite values")
    return output


def effective_binary_mask(labels, at_risk, observed):
    y = np.asarray(labels)
    a = np.asarray(at_risk)
    o = np.asarray(observed)
    if y.shape != a.shape or y.shape != o.shape:
        raise ValueError("Mask shape mismatch")
    if not np.isin(a, [0, 1]).all() or not np.isin(o, [0, 1]).all():
        raise ValueError("Masks must be binary")
    mask = a.astype(bool) & o.astype(bool)
    if not np.isfinite(y[mask]).all() or not np.isin(y[mask], [0, 1]).all():
        raise ValueError("Observed labels must be finite binary values")
    return mask


class BaselinePreprocessor:
    """Fit on nine training DataFrames, then reuse on validation/test DataFrames."""

    def __init__(self):
        self.state = {}

    @staticmethod
    def _frames(frames):
        if set(frames) != set(GROUPS):
            raise ValueError(f"Expected the nine groups: {GROUPS}")
        index = frames["demo"].index
        if not len(index) or not index.is_unique:
            raise ValueError("Use nonempty, unique row indices")
        if any(not frames[g].index.equals(index) for g in GROUPS):
            raise ValueError("All groups must have the same participant row order")
        if any(
            not len(frames[g].columns) or not frames[g].columns.is_unique
            for g in GROUPS
        ):
            raise ValueError("Each group needs uniquely named feature columns")
        return {g: frames[g].reset_index(drop=True) for g in GROUPS}

    def fit(self, training_frames):
        frames = self._frames(training_frames)
        two, three = strata_keys(frames["demo"], frames["physical"])
        rows = np.arange(len(frames["demo"]))
        state = {}
        for group in GROUPS:
            spec = fit_frame(frames[group], rows, group, two, three)
            prepared = transform_frame(frames[group], spec, two, three)
            scaler = StandardScaler()
            for start in range(0, len(prepared), 4096):
                scaler.partial_fit(
                    np.asarray(prepared[start : start + 4096], dtype=np.float64)
                )
            state[group] = dict(
                spec=spec, mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist()
            )
        self.state = state
        return self

    @property
    def group_dims(self):
        if not self.state:
            raise RuntimeError("Fit or load the preprocessor first")
        return [
            sum(
                2 if rule["kind"] == "numeric" else len(rule["categories"]) + 2
                for rule in self.state[group]["spec"]["columns"]
            )
            for group in GROUPS
        ]

    def transform(self, frames):
        dimensions = self.group_dims
        frames = self._frames(frames)
        two, three = strata_keys(frames["demo"], frames["physical"])
        outputs = []
        for group, dimension in zip(GROUPS, dimensions):
            state = self.state[group]
            raw = transform_frame(frames[group], state["spec"], two, three)
            encoded = np.zeros((len(raw), dimension), dtype=np.float32)
            source = target = 0
            for rule in state["spec"]["columns"]:
                if rule["kind"] == "numeric":
                    encoded[:, target] = (
                        raw[:, source] - state["mean"][source]
                    ) / state["scale"][source]
                    encoded[:, target + 1] = raw[:, source + 1]
                    source += 2
                    target += 2
                else:
                    width = len(rule["categories"]) + 2
                    codes = raw[:, source].astype(np.int64)
                    encoded[np.arange(len(raw)), target + codes] = 1.0
                    source += 1
                    target += width
            if not np.isfinite(encoded).all():
                raise ValueError("Nonfinite encoded baseline values")
            outputs.append(encoded)
        return outputs

    def save(self, path):
        if not self.state:
            raise RuntimeError("Fit the preprocessor before saving")
        Path(path).write_text(
            json.dumps(self.state, indent=2, allow_nan=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path):
        instance = cls()
        instance.state = json.loads(Path(path).read_text(encoding="utf-8"))
        return instance


def build_vocab(*event_streams, min_freq=1):
    """Build one shared vocabulary from supplied code/time event lists.

    Each stream is a list of participants, each with [(code, years), ...].
    Reuse the resulting vocabulary for every subsequent transform.
    """
    if min_freq < 1:
        raise ValueError("min_freq must be positive")
    counts = Counter(
        str(code) for stream in event_streams for events in stream for code, _ in events
    )
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for code in sorted(counts):
        if counts[code] >= min_freq and code not in vocab:
            vocab[code] = len(vocab)
    return vocab


def encode_events(event_lists, vocab):
    """Preserve supplied order; keep first 12 and last 116 if length exceeds 128.

    Supply pre-baseline coded records in date/code order. Baseline phenotype
    fields use field-name order, retain repetitions, and have time zero.
    """
    if vocab.get("<PAD>") != 0 or vocab.get("<UNK>") != 1:
        raise ValueError("Vocabulary must reserve PAD=0 and UNK=1")
    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError("Vocabulary IDs must be consecutive and unique")
    output = np.zeros((len(event_lists), 128, 2), dtype=np.float32)
    for i, events in enumerate(event_lists):
        events = [(str(code), float(years)) for code, years in events]
        if any(not np.isfinite(years) or years > 0 for _, years in events):
            raise ValueError(
                "Times must be finite years relative to baseline, at or before zero"
            )
        if len(events) > 128:
            events = events[:12] + events[-116:]
        for j, (code, years) in enumerate(events):
            output[i, j] = (vocab.get(code, 1), years)
    return output
