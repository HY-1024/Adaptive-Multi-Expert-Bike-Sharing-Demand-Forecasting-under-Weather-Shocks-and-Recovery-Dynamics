from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import capital_data_and_statistics as monthly

SEED = 20260910
ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [1, 2, 3, 6]
N_JOBS = max(1, min(8, (os.cpu_count() or 2) - 1))
DEVICE = os.environ.get("XGB_DEVICE", "cpu")

def nb_grad_hess(y: np.ndarray, eta: np.ndarray, alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    eta = np.clip(np.asarray(eta, dtype=float), -20.0, 20.0)
    y = np.asarray(y, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    mu = np.exp(eta)
    grad = (mu - y) / (1.0 + alpha * mu)
    hess = mu * (1.0 + alpha * y) / ((1.0 + alpha * mu) ** 2)
    return grad, np.maximum(hess, 1e-8)

def month_start(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts.dt.strftime("%Y-%m-01")).dt.tz_localize("UTC")

def make_multi_frame(hourly: pd.DataFrame) -> pd.DataFrame:
    samples = {h: monthly.make_samples(hourly, h).copy() for h in HORIZONS}
    base = samples[1].copy()
    keep_base = [
        "utc_hour",
        "departures",
        "source_missing_hour",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "rain",
        "snowfall",
        "wind_speed_10m",
        "issue_stage",
        "online_event_id",
        "eval_event_cluster_id",
        "shock_active",
        "recovery_active",
        "current_rain_duration",
        "ended_rain_duration",
        "ended_rain_cum",
        "hours_since_rain_end",
    ] + [f"rain_lag_{i}" for i in range(7)] + [
        "lag_0",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_6",
        "lag_24",
        "lag_48",
        "lag_168",
        "log_lag_0",
        "log_lag_1",
        "log_lag_24",
        "log_lag_168",
        "roll_mean_3",
        "roll_mean_24",
        "roll_mean_168",
        "roll_std_24",
        "rain_sum_3",
        "rain_sum_6",
    ]
    out = base[keep_base].copy()
    for h, s in samples.items():
        cols = [
            "utc_hour",
            "target",
            "target_time",
            "target_actual_stage",
            "target_hour",
            "target_dow",
            "target_month",
            "target_doy",
            "is_weekend",
            "is_commute",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "year_sin",
            "year_cos",
            "recovery_elapsed_at_target",
        ]
        tmp = s[cols].rename(
            columns={
                "target": f"y_h{h}",
                "target_time": f"target_time_h{h}",
                "target_actual_stage": f"target_actual_stage_h{h}",
                "target_hour": f"target_hour_h{h}",
                "target_dow": f"target_dow_h{h}",
                "target_month": f"target_month_h{h}",
                "target_doy": f"target_doy_h{h}",
                "is_weekend": f"is_weekend_h{h}",
                "is_commute": f"is_commute_h{h}",
                "hour_sin": f"hour_sin_h{h}",
                "hour_cos": f"hour_cos_h{h}",
                "dow_sin": f"dow_sin_h{h}",
                "dow_cos": f"dow_cos_h{h}",
                "year_sin": f"year_sin_h{h}",
                "year_cos": f"year_cos_h{h}",
                "recovery_elapsed_at_target": f"recovery_elapsed_at_target_h{h}",
            }
        )
        out = out.merge(tmp, on="utc_hour", how="inner")
    out["issue_month_start"] = month_start(out["utc_hour"])
    out["issue_month"] = out["issue_month_start"].dt.strftime("%Y-%m")
    out["label_available_time"] = out[f"target_time_h{max(HORIZONS)}"]
    out["state_for_fusion"] = np.where(
        out["shock_active"].eq(1),
        "shock",
        np.where(out["recovery_active"].eq(1), "recovery", "normal"),
    )
    out["demand_delta_1"] = out["lag_0"] - out["lag_1"]
    out["demand_delta_3"] = out["lag_0"] - out["lag_3"]
    out["demand_delta_24"] = out["lag_0"] - out["lag_24"]
    out["rain_indicator"] = out["rain"].fillna(0.0).gt(0).astype(float)
    out["rain_lag_level"] = np.log1p(out[[f"rain_lag_{i}" for i in range(1, 4)]].sum(axis=1))
    out["rain_lag_tail"] = np.log1p(out[[f"rain_lag_{i}" for i in range(4, 7)]].sum(axis=1))
    out["rain_lag_contrast"] = out["rain_lag_level"] - out["rain_lag_tail"]
    out["recovery_elapsed_clipped"] = np.minimum(out["hours_since_rain_end"].replace(999.0, 72.0), 72.0)
    out["recovery_decay_12h"] = np.where(
        out["recovery_active"].eq(1),
        np.log1p(out["ended_rain_cum"].fillna(0.0)) * np.exp(-out["recovery_elapsed_clipped"] / 12.0),
        0.0,
    )
    out["ended_rain_intensity"] = np.where(
        out["ended_rain_duration"].fillna(0.0) > 0,
        out["ended_rain_cum"].fillna(0.0) / np.maximum(out["ended_rain_duration"].fillna(0.0), 1.0),
        0.0,
    )
    return out.sort_values("utc_hour").reset_index(drop=True)


def target_calendar_cols() -> List[str]:
    cols = []
    for h in HORIZONS:
        cols += [
            f"target_hour_h{h}",
            f"target_dow_h{h}",
            f"target_month_h{h}",
            f"is_weekend_h{h}",
            f"is_commute_h{h}",
            f"hour_sin_h{h}",
            f"hour_cos_h{h}",
            f"dow_sin_h{h}",
            f"dow_cos_h{h}",
            f"year_sin_h{h}",
            f"year_cos_h{h}",
        ]
    return cols


def public_numeric_cols(h: int) -> List[str]:
    return [
        "lag_0",
        "lag_24",
        "lag_168",
        "log_lag_0",
        "log_lag_24",
        "log_lag_168",
        f"hour_sin_h{h}",
        f"hour_cos_h{h}",
        f"dow_sin_h{h}",
        f"dow_cos_h{h}",
        f"year_sin_h{h}",
        f"year_cos_h{h}",
        f"is_weekend_h{h}",
        f"is_commute_h{h}",
    ]


def demand_expert_numeric_cols() -> List[str]:
    return [
        "lag_0",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_6",
        "lag_24",
        "lag_48",
        "lag_168",
        "log_lag_0",
        "log_lag_1",
        "log_lag_24",
        "log_lag_168",
        "demand_delta_1",
        "demand_delta_3",
        "demand_delta_24",
        "roll_mean_3",
        "roll_mean_24",
        "roll_mean_168",
        "roll_std_24",
    ] + target_calendar_cols()


def fit_preprocessor(train: pd.DataFrame, numeric_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    ).fit(train)


def estimate_alpha_matrix(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return np.array([monthly.estimate_alpha(y[:, i], mu[:, i]) for i in range(y.shape[1])], dtype=float)


@dataclass
class DemandModel:
    preprocessor: ColumnTransformer
    booster: xgb.Booster
    alpha: np.ndarray
    init_margin: np.ndarray
    eta_clip_high: float


def fit_demand_model(train: pd.DataFrame) -> DemandModel:
    numeric = demand_expert_numeric_cols()
    cat = []
    pre = fit_preprocessor(train, numeric, cat)
    X = np.asarray(pre.transform(train), dtype=float)
    y = train[[f"y_h{h}" for h in HORIZONS]].to_numpy(float)
    init_mu = np.tile(np.maximum(y.mean(axis=0), 1e-6), (len(y), 1))
    alpha = estimate_alpha_matrix(y, init_mu)
    init_margin = np.log(np.maximum(y.mean(axis=0), 1e-6))
    dtrain = xgb.DMatrix(X, label=y)
    dtrain.set_base_margin(np.tile(init_margin, (len(y), 1)))

    def obj(predt: np.ndarray, dmat: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        labels = dmat.get_label().reshape(predt.shape)
        grad, hess = nb_grad_hess(labels, predt, alpha)
        return grad / len(HORIZONS), hess / len(HORIZONS)

    booster = xgb.train(
        {
            "tree_method": "hist",
            "device": DEVICE,
            "multi_strategy": "multi_output_tree",
            "max_depth": 3,
            "eta": 0.06,
            "subsample": 0.9,
            "colsample_bynode": 0.9,
            "lambda": 3.0,
            "max_delta_step": 1.0,
            "verbosity": 0,
            "nthread": N_JOBS,
            "seed": SEED,
        },
        dtrain,
        num_boost_round=120,
        obj=obj,
    )
    return DemandModel(pre, booster, alpha, init_margin, eta_clip_high=float(np.log(max(y.max() * 3.0, 1.0))))


def predict_demand_model(model: DemandModel, frame: pd.DataFrame) -> np.ndarray:
    X = np.asarray(model.preprocessor.transform(frame), dtype=float)
    dtest = xgb.DMatrix(X)
    dtest.set_base_margin(np.tile(model.init_margin, (len(frame), 1)))
    eta = model.booster.predict(dtest)
    eta = np.clip(eta, -20.0, model.eta_clip_high)
    return np.exp(eta)

@dataclass
class NBExpert:
    preprocessor: ColumnTransformer
    alpha: float
    model: object
    status: str
    reason: str
    cap: float


def fit_nb_expert(train: pd.DataFrame, y_col: str, numeric_cols: List[str], cat_cols: List[str]) -> NBExpert:
    pre = fit_preprocessor(train, numeric_cols, cat_cols)
    X = np.asarray(pre.transform(train), dtype=float)
    Xc = sm.add_constant(X, has_constant="add")
    y = train[y_col].to_numpy(float)
    poisson = PoissonRegressor(alpha=0.02, max_iter=1000)
    poisson.fit(Xc, y)
    mu0 = np.clip(poisson.predict(Xc), 1e-6, None)
    alpha = monthly.estimate_alpha(y, mu0)
    status = "ok"
    reason = ""
    model: object = poisson
    try:
        glm = sm.GLM(y, Xc, family=sm.families.NegativeBinomial(alpha=alpha))
        model = glm.fit_regularized(alpha=0.025, L1_wt=0.0, maxiter=160)
    except Exception as exc:
        status = "failed_used_poisson"
        reason = f"{type(exc).__name__}: {exc}"
    return NBExpert(pre, alpha, model, status, reason, cap=float(max(np.nanmax(y) * 3.0, 1.0)))


def predict_nb_expert(model: NBExpert, frame: pd.DataFrame) -> np.ndarray:
    X = np.asarray(model.preprocessor.transform(frame), dtype=float)
    Xc = sm.add_constant(X, has_constant="add")
    raw = np.asarray(model.model.predict(Xc), dtype=float)
    return np.clip(np.nan_to_num(raw, nan=0.0, posinf=model.cap, neginf=0.0), 0.0, model.cap)

def fit_calibrator(y: np.ndarray, mu: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    ok = np.isfinite(y) & np.isfinite(mu)
    if ok.sum() < 200:
        return {"a": 0.0, "b": 1.0, "status": "identity_insufficient_data"}
    y = y[ok]
    mu = mu[ok]
    scale = float(max(np.mean(y), 1.0))

    def obj(theta: np.ndarray) -> float:
        a, b = theta
        pred = np.maximum(0.0, a + b * mu)
        return float(np.mean((y - pred) ** 2) / (scale**2) + 0.1 * (a / scale) ** 2 + 0.1 * (b - 1.0) ** 2)

    res = minimize(obj, np.array([0.0, 1.0]), method="L-BFGS-B", bounds=[(-scale, scale), (0.1, 5.0)])
    if not res.success:
        return {"a": 0.0, "b": 1.0, "status": f"identity_failed_{res.message}"}
    return {"a": float(res.x[0]), "b": float(res.x[1]), "status": "ok"}
