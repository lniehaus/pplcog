"""Soil population, terrain classes, and the Stage A and Stage C learning components."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..params import Actuator, Rover, Soil
from ..terramechanics import sample_soils
from .features import FEATURE_NAMES
from .params_sim import SimParams, SoilWR
from .wheel_soil import WheelTable


def soil_population(p: SimParams, base: Soil | None = None, wr: SoilWR | None = None) -> list[SoilWR]:
    """n_soils draws of the uncertain soil parameters (base Bekker / Mohr-Coulomb set plus Wong-Reece extras)."""
    base = Soil() if base is None else base
    wr = SoilWR() if wr is None else wr
    rng = np.random.default_rng(p.seed)
    bases = sample_soils(base, rng, p.n_soils)
    out = []
    for b in bases:
        d = wr.sample(rng)
        out.append(replace(d, base=replace(b, K=d.base.K)))
    return out


def traction_index(soil: SoilWR, rv: Rover, p: SimParams, tab: WheelTable | None = None) -> float:
    tab = WheelTable.build(soil, rv, p) if tab is None else tab
    return tab.traction_index(rv.M * rv.g / 4, 0.3)


def strata(indices: np.ndarray, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Class label per soil from quantiles of the traction index, and the cut points."""
    cuts = np.quantile(indices, np.linspace(0, 1, n_classes + 1)[1:-1])
    return np.searchsorted(cuts, indices, side="right"), cuts


def feature_vector(feat: dict) -> np.ndarray:
    return np.array([feat[k] for k in FEATURE_NAMES], dtype=float)


@dataclass
class StageAArtefact:
    classifier: object                     # sklearn HistGradientBoostingClassifier
    table: dict                            # (class, slope_bin) -> xi set point
    slope_bins: np.ndarray                 # bin edges in degrees
    cuts: np.ndarray                       # traction index cut points
    cv_accuracy: float
    test_accuracy: float
    importances: dict

    def classify(self, feat: dict, theta: float) -> int:
        x = np.concatenate([feature_vector(feat), [np.degrees(theta)]])
        return int(self.classifier.predict(x[None, :])[0])

    def set_point(self, cls: int, theta: float) -> float:
        b = int(np.clip(np.searchsorted(self.slope_bins, np.degrees(theta), side="right") - 1, 0, len(self.slope_bins) - 2))
        return self.table[(cls, b)]


@dataclass
class StageCArtefact:
    gp: object                             # sklearn GaussianProcessRegressor
    x_mean: np.ndarray
    x_std: np.ndarray
    cv_rmse: float
    target_std: float
    n_train: int

    def _x(self, feat: dict, parity: int, theta: float) -> np.ndarray:
        x = np.concatenate([[parity, np.degrees(theta)], feature_vector(feat)])
        return (x - self.x_mean) / self.x_std

    def predict(self, feat: dict, parity: int, theta: float) -> float:
        return float(self.gp.predict(self._x(feat, parity, theta)[None, :])[0])

    def predict_std(self, feat: dict, parity: int, theta: float) -> float:
        _, sd = self.gp.predict(self._x(feat, parity, theta)[None, :], return_std=True)
        return float(sd[0])


# ---------------------------------------------------------------------------
# Fitting

def _cot_capped(s, cap: float = 20.0) -> float:
    v = s.cot if s.completed else np.inf
    return float(min(v, cap)) if np.isfinite(v) else cap


def _objective(s, cap: float = 20.0) -> float:
    """Measurable training objective: cost of transport, with failures penalised."""
    return _cot_capped(s, cap)


def soils_key(soils: list[SoilWR]) -> str:
    """Deterministic fingerprint of a soil population (dataclass hashing is not stable across processes)."""
    import joblib
    return joblib.hash([(s.k_c, s.k_phi, s.n, s.c, s.phi_s, s.K, s.a0, s.a1, s.lam, s.gamma_s) for s in soils])


MIN_COMPLETED_A = 3     # training soils that must complete at the chosen rail position (study decision)


def physics_set_point(rv: Rover, theta: float, rho_star: float = 1.0) -> float:
    """Constant rail position for Stage A when the grid search is uninformative: the Eq. (3) set point
    averaged over the contracted and the extended wheelbase, clipped to the rail."""
    from .cog_controllers import feedforward_xi
    return float(np.mean([feedforward_xi(L, theta, rv, rho_star) for L in (rv.L_min, rv.L_max)]))


def select_set_point(scores: np.ndarray, n_completed: np.ndarray, grid: np.ndarray, fallback_xi: float,
                     min_completed: int = MIN_COMPLETED_A) -> tuple[float, bool]:
    """Grid position with the lowest mean objective, or the fallback when fewer than min_completed
    training soils completed there (then the capped objective is flat and argmin would return the
    first grid entry)."""
    j = int(np.argmin(scores))
    if n_completed[j] < min_completed:
        return float(fallback_xi), True
    return float(grid[j]), False


def fit_stage_a(rv: Rover, act: Actuator, p: SimParams, soils: list[SoilWR], labels: np.ndarray,
                train_idx: list[int], test_idx: list[int], cuts: np.ndarray, n_jobs: int = -1,
                soils_key: str | None = None) -> "StageAArtefact":
    from joblib import Parallel, delayed
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import GroupKFold, cross_val_score
    from .experiment import one_run

    grid = np.linspace(rv.rail_lo, rv.rail_hi, p.grid_A)
    centre = 0.5 * (rv.rail_lo + rv.rail_hi)
    slopes = p.slopes_deg
    jobs, keys = [], []
    for si in train_idx:
        for th in slopes:
            for xi in grid:
                jobs.append(delayed(one_run)(rv, act, p, soils, si, th, "FIX", float(xi), p.train_cycles, False))
                keys.append((si, th, float(xi)))
    res = dict(zip(keys, Parallel(n_jobs=n_jobs)(jobs)))
    # per-class, per-slope set point minimising the mean capped cost of transport; where too few training
    # soils complete, the grid search carries no information and the physics set point is used instead
    slope_bins = np.array(list(slopes) + [slopes[-1] + 10.0])
    table, fallback = {}, []
    for cls in range(p.n_classes):
        members = [si for si in train_idx if labels[si] == cls]
        for b, th in enumerate(slopes):
            runs = [[res[(si, th, float(xi))] for si in members] for xi in grid]
            scores = np.array([np.mean([_objective(s) for s in row]) for row in runs])
            n_completed = np.array([sum(s.completed for s in row) for row in runs])
            xi, used_fallback = select_set_point(scores, n_completed, grid, physics_set_point(rv, np.radians(th)))
            table[(cls, b)] = xi
            if used_fallback:
                fallback.append((cls, b))
    # classifier on the probing-cycle features (centre position, first cycle)
    def probe_rows(idx):
        X, y, g = [], [], []
        for si in idx:
            for th in slopes:
                xi_c = float(grid[np.argmin(np.abs(grid - centre))])
                s = res.get((si, th, xi_c))
                if s is None:
                    s = one_run(rv, act, p, soils, si, th, "FIX", xi_c, p.train_cycles, False)
                if s.cycles:
                    X.append([s.cycles[0]["f_" + k] for k in FEATURE_NAMES] + [th]); y.append(labels[si]); g.append(si)
        return np.array(X), np.array(y), np.array(g)
    X, y, g = probe_rows(train_idx)
    clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, random_state=0)
    cv = float(np.mean(cross_val_score(clf, X, y, cv=GroupKFold(n_splits=5), groups=g)))
    clf.fit(X, y)
    Xt, yt, _ = probe_rows(test_idx)
    test_acc = float(np.mean(clf.predict(Xt) == yt)) if len(yt) else np.nan
    imp = permutation_importance(clf, Xt, yt, n_repeats=10, random_state=0) if len(yt) else None
    names = FEATURE_NAMES + ["slope_deg"]
    importances = {n: float(v) for n, v in zip(names, imp.importances_mean)} if imp is not None else {}
    art = StageAArtefact(clf, table, slope_bins, cuts, cv, test_acc, importances)
    art.grid_results = {str(k): _objective(v) for k, v in res.items()}
    art.grid_completed = {str(k): bool(v.completed) for k, v in res.items()}
    art.table_fallback = fallback
    return art


def fit_stage_c(rv: Rover, act: Actuator, p: SimParams, soils: list[SoilWR], train_idx: list[int],
                n_jobs: int = -1, soils_key: str | None = None) -> "StageCArtefact":
    from joblib import Parallel, delayed
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    from sklearn.model_selection import GroupKFold
    from .cog_controllers import ConstantOffsets, delta_max_for_run
    from .experiment import one_run

    slopes = p.slopes_deg
    def search(si, th):
        theta = np.radians(th)
        Dm = delta_max_for_run(rv, theta, p)
        base = one_run(rv, act, p, soils, si, th, "C", None, p.train_cycles, False)
        feat = base.cycles[0] if base.cycles else None
        if Dm <= 0 or feat is None:
            return si, th, 0.0, 0.0, feat, _objective(base), _objective(base)
        grid = np.linspace(-Dm, Dm, p.grid_C)
        best = (_objective(base), 0.0, 0.0)
        for d_odd in grid:                       # coordinate search, odd phases first
            s = one_run(rv, act, p, soils, si, th, "C", ConstantOffsets(float(d_odd), 0.0), p.train_cycles, False)
            if _objective(s) < best[0]:
                best = (_objective(s), float(d_odd), 0.0)
        for d_even in grid:
            s = one_run(rv, act, p, soils, si, th, "C", ConstantOffsets(best[1], float(d_even)), p.train_cycles, False)
            if _objective(s) < best[0]:
                best = (_objective(s), best[1], float(d_even))
        return si, th, best[1], best[2], feat, _objective(base), best[0]
    out = Parallel(n_jobs=n_jobs)(delayed(search)(si, th) for si in train_idx for th in slopes)
    X, y, g = [], [], []
    for si, th, d_odd, d_even, feat, j0, j1 in out:
        if feat is None:
            continue
        fv = [feat["f_" + k] for k in FEATURE_NAMES]
        X.append([1, th] + fv); y.append(d_odd); g.append(si)
        X.append([0, th] + fv); y.append(d_even); g.append(si)
    X, y, g = np.array(X), np.array(y), np.array(g)
    x_mean, x_std = X.mean(0), X.std(0) + 1e-9
    Xn = (X - x_mean) / x_std
    kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(X.shape[1]),
                                       length_scale_bounds=(p.gp_ls_min, 1e5)) + WhiteKernel(1e-2)  # lower bound in standardised units
    make = lambda: GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=0)
    errs = []
    for tr, te in GroupKFold(n_splits=5).split(Xn, y, g):
        m = make().fit(Xn[tr], y[tr]); errs.append(np.mean((m.predict(Xn[te]) - y[te]) ** 2))
    cv_rmse = float(np.sqrt(np.mean(errs)))
    gp = make().fit(Xn, y)
    art = StageCArtefact(gp, x_mean, x_std, cv_rmse, float(np.std(y)), int(len(y)))
    art.X_train, art.y_train, art.groups = X, y, g      # raw training rows, reused by optional experiments
    art.search_results = [{"soil": si, "slope_deg": th, "d_odd": d_odd, "d_even": d_even, "cot_ff": j0, "cot_best": j1}
                          for si, th, d_odd, d_even, _, j0, j1 in out]
    return art
