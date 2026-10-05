from __future__ import annotations
import csv, math
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit

NORMALIZE_INITIAL = {"5-5-5":"5-5-5", "10-10-10":"10-10-10", "30-30-30":"30-30-30"}
XMAP = {"Initial set only":0.0, "1%":1.0, "2%":2.0, "4%":4.0, "8%":8.0, "16%":16.0}

def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if "initial_set" in row:
            row["initial_set"] = NORMALIZE_INITIAL.get(row["initial_set"], row["initial_set"])
    return rows

def saturation(x, gain, rate, y0):
    x = np.asarray(x, dtype=float)
    return y0 + gain * (1.0 - np.exp(-rate * x))

def fit_saturation(points):
    points = sorted(points)
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    y0 = float(y[np.where(x == 0.0)[0][0]])
    def model(xx, gain, rate):
        return saturation(xx, gain, rate, y0)
    popt, pcov = curve_fit(model, x, y, p0=[max(0.05, max(y)-y0), 0.5], bounds=([0.0,1e-6],[100.0,20.0]), maxfev=50000)
    gain, rate = map(float, popt)
    fitted = model(x, gain, rate)
    ss_res = float(np.sum((y-fitted)**2))
    ss_tot = float(np.sum((y-y.mean())**2))
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 1e-12 else 1.0
    return {"y0":y0, "gain":gain, "rate_k":rate, "plateau":y0+gain, "x95":math.log(20.0)/rate, "r2":r2}
