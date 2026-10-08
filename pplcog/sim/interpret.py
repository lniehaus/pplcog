"""Interpretation of the Stage C residual: partial dependences of the Gaussian process per phase type.

Each input is swept over its training range with the other inputs held at the mean of the
training rows of the same phase type. The phase parity is binary, so it is never swept and never
set to its mean (there the Gaussian process reverts to its prior).
"""

from __future__ import annotations

import numpy as np

from ..params import Rover
from .features import FEATURE_NAMES
from .learning import StageCArtefact

INPUT_NAMES = ["parity", "slope_deg"] + FEATURE_NAMES
PARITY_LABEL = {1: "contraction", 0: "extension"}


def training_rows(C: StageCArtefact) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(C, "X_train"):
        return np.asarray(C.X_train, float), np.asarray(C.y_train, float)
    gp = C.gp
    return gp.X_train_ * C.x_std + C.x_mean, gp.y_train_ * gp._y_train_std + gp._y_train_mean


def gp_partial_dependences(C: StageCArtefact, n_grid: int = 41) -> dict:
    X, y = training_rows(C)
    Xn = (X - C.x_mean) / C.x_std
    out = {"names": [n for n in INPUT_NAMES if n != "parity"], "grid": {}, "mean": {}, "std": {}}
    for j, name in enumerate(INPUT_NAMES):
        if name == "parity":
            continue
        lo, hi = np.percentile(Xn[:, j], [2, 98])
        grid = np.linspace(lo, hi, n_grid)
        out["grid"][name] = grid * C.x_std[j] + C.x_mean[j]
        out["mean"][name], out["std"][name] = {}, {}
        for par in (1, 0):
            par_n = (par - C.x_mean[0]) / C.x_std[0]
            base = Xn[np.isclose(Xn[:, 0], par_n)].mean(axis=0)
            Xs = np.repeat(base[None, :], grid.size, axis=0)
            Xs[:, j] = grid
            m, s = C.gp.predict(Xs, return_std=True)
            out["mean"][name][par], out["std"][name][par] = m, s
    # relevance from the automatic-relevance-determination length scales (small = relevant)
    ls = np.asarray(C.gp.kernel_.k1.k2.length_scale, float)
    out["length_scale"] = dict(zip(INPUT_NAMES, map(float, ls)))
    return out


def rule_numbers(pd: dict, rv: Rover) -> dict:
    """The readable rule: contraction offset at low slope, slope where it vanishes, extension offsets."""
    R = rv.rail_len
    g = pd["grid"]["slope_deg"]
    con = pd["mean"]["slope_deg"][1] / R
    ext = pd["mean"]["slope_deg"][0] / R
    below = np.flatnonzero(con < 0.1)
    slope_vanish = float(g[below[0]]) if below.size else float("nan")
    ranked = sorted(pd["length_scale"].items(), key=lambda kv: kv[1])
    return {"contraction_offset_over_R_at_0deg": float(con[0]),
            "contraction_offset_over_R_max": float(con.max()),
            "contraction_offset_vanishes_above_deg": slope_vanish,
            "extension_offset_over_R_mean": float(ext.mean()),
            "extension_offset_over_R_min": float(ext.min()),
            "extension_slope_at_min_deg": float(g[int(np.argmin(ext))]),
            "relevant_inputs_by_length_scale": [n for n, _ in ranked[:5]],
            "length_scales": pd["length_scale"],
            "response_range_over_R": {n: float(max(np.ptp(pd["mean"][n][1]), np.ptp(pd["mean"][n][0])) / R)
                                      for n in pd["names"]}}
