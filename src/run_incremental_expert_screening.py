from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import six_expert_statistical_core as base


SEED = 20260910
HORIZONS = [1, 2, 3, 6]
RUN_ID = os.environ.get("INCREMENTAL_RUN_ID", time.strftime("incremental_expert_%Y%m%d_%H%M%S"))
RUN_DIR = ROOT / "runs" / RUN_ID
TABLE_DIR = RUN_DIR / "tables"
FIG_DIR = RUN_DIR / "figures"
REPORT_DIR = RUN_DIR / "reports"
MODEL_DIR = RUN_DIR / "models"
LOG_DIR = RUN_DIR / "logs"
REPORT_ARCHIVE_DIR = ROOT / "reports" / "six_expert_experiments" / RUN_ID
STATUS_PATH = RUN_DIR / "status.json"
LATEST_STATUS_PATH = ROOT / "reports" / "six_expert_experiments" / "latest_incremental_status.json"

PERIODS = [
    ("calibration_fit", pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-07-01", tz="UTC")),
    ("fusion_fit", pd.Timestamp("2024-07-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")),
    ("screening_dev", pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-04-01", tz="UTC")),
    ("final_eval", pd.Timestamp("2025-04-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")),
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_ENABLED = DEVICE == "cuda"


def ensure_dirs() -> None:
    for path in [RUN_DIR, TABLE_DIR, FIG_DIR, REPORT_DIR, MODEL_DIR, LOG_DIR, REPORT_ARCHIVE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def write_status(stage: str, message: str, **extra: object) -> None:
    payload = {
        "run_id": RUN_ID,
        "run_dir": str(RUN_DIR),
        "stage": stage,
        "message": message,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": DEVICE,
        **extra,
    }
    STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    LATEST_STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(message)


def rmse(y: Iterable[float], pred: Iterable[float]) -> float:
    y_arr = np.asarray(y, dtype=float)
    p_arr = np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((y_arr - p_arr) ** 2)))


def model_period(issue_time: pd.Series) -> pd.Series:
    conditions = [(issue_time >= start) & (issue_time < end) for _, start, end in PERIODS]
    choices = [name for name, _, _ in PERIODS]
    return pd.Series(np.select(conditions, choices, default="unused"), index=issue_time.index)


def load_frame() -> pd.DataFrame:
    base.HORIZONS = HORIZONS
    hourly = base.monthly.add_online_events(base.monthly.read_capital_cached())
    frame = base.make_multi_frame(hourly)
    frame["split"] = model_period(frame["utc_hour"])
    frame = add_long_rain_lags(frame)
    frame = add_sequence_calendar(frame)
    return frame.sort_values("utc_hour").reset_index(drop=True)


def add_long_rain_lags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    rain = out["rain"].fillna(0.0)
    for lag in range(7, 73):
        out[f"rain_lag_{lag}"] = rain.shift(lag).fillna(0.0)
    return out


def add_sequence_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    local = out["utc_hour"].dt.tz_convert("America/New_York")
    out["issue_hour_sin"] = np.sin(2 * np.pi * local.dt.hour / 24.0)
    out["issue_hour_cos"] = np.cos(2 * np.pi * local.dt.hour / 24.0)
    out["issue_dow_sin"] = np.sin(2 * np.pi * local.dt.dayofweek / 7.0)
    out["issue_dow_cos"] = np.cos(2 * np.pi * local.dt.dayofweek / 7.0)
    out["issue_commute"] = local.dt.hour.isin([7, 8, 9, 16, 17, 18, 19]).astype(float)
    return out


def period_train_pred(frame: pd.DataFrame, period_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _, start, end = next(p for p in PERIODS if p[0] == period_name)
    train = frame[frame["label_available_time"] < start].copy()
    pred = frame[(frame["utc_hour"] >= start) & (frame["utc_hour"] < end)].copy()
    return train, pred


def public_numeric_cols(h: int) -> List[str]:
    return base.public_numeric_cols(h)


def fit_nb_from_preprocessor(train: pd.DataFrame, y_col: str, preprocessor: ColumnTransformer, alpha_ridge: float = 0.025) -> base.NBExpert:
    X = np.asarray(preprocessor.fit_transform(train), dtype=float)
    Xc = sm.add_constant(X, has_constant="add")
    y = train[y_col].to_numpy(float)
    poisson = PoissonRegressor(alpha=0.02, max_iter=1200)
    poisson.fit(Xc, y)
    mu0 = np.clip(poisson.predict(Xc), 1e-6, None)
    alpha = base.monthly.estimate_alpha(y, mu0)
    status = "ok"
    reason = ""
    model: object = poisson
    try:
        glm = sm.GLM(y, Xc, family=sm.families.NegativeBinomial(alpha=alpha))
        model = glm.fit_regularized(alpha=alpha_ridge, L1_wt=0.0, maxiter=180)
    except Exception as exc:
        status = "failed_used_poisson"
        reason = f"{type(exc).__name__}: {exc}"
    return base.NBExpert(preprocessor, alpha, model, status, reason, cap=float(max(np.nanmax(y) * 3.0, 1.0)))


def fit_predict_e1(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[pd.DataFrame] = []
    diag_rows: List[Dict[str, object]] = []
    for period, _, _ in PERIODS:
        train, pred = period_train_pred(frame, period)
        write_status("running", f"E1 fitting for {period}", current_model="E1")
        model = base.fit_demand_model(train)
        mu = base.predict_demand_model(model, pred)
        for j, h in enumerate(HORIZONS):
            pred[f"mu_E1_raw_h{h}"] = mu[:, j]
        rows.append(pred)
        diag_rows.append({"expert": "E1", "config": "fixed_nb_xgb", "period": period, "train_rows": len(train), "pred_rows": len(pred), "status": "ok"})
    out = pd.concat(rows, ignore_index=True).sort_values("utc_hour").reset_index(drop=True)
    return out, pd.DataFrame(diag_rows)


def weather_preprocessor(h: int, variant: str) -> Tuple[ColumnTransformer, List[str]]:
    public_cols = public_numeric_cols(h)
    spline_cols = ["temperature_2m", "rain_log_current", "relative_humidity_2m", "wind_speed_10m"]
    extra_cols = []
    if variant in {"spline_th", "full_tensor"}:
        extra_cols.append("temp_humidity_interaction")
    if variant == "full_tensor":
        extra_cols += ["rain_wind_interaction", "rain_commute_interaction"]
    cat_cols = [f"target_hour_h{h}", f"target_dow_h{h}", f"target_month_h{h}"]
    pre = ColumnTransformer(
        [
            ("public", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), public_cols),
            ("spline", Pipeline([("imp", SimpleImputer(strategy="median")), ("spline", SplineTransformer(n_knots=4, degree=3, include_bias=False)), ("scale", StandardScaler())]), spline_cols),
            ("extra", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), extra_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )
    return pre, public_cols + spline_cols + extra_cols + cat_cols


def add_ts_weather_features(frame: pd.DataFrame, h: int) -> pd.DataFrame:
    out = frame.copy()
    rain_log = np.log1p(out["rain"].fillna(0.0).astype(float))
    temp = out["temperature_2m"].astype(float)
    hum = out["relative_humidity_2m"].astype(float)
    wind = out["wind_speed_10m"].astype(float)
    out["rain_log_current"] = rain_log
    out["temp_humidity_interaction"] = (temp - temp.median()) * (hum - hum.median())
    out["rain_wind_interaction"] = rain_log * wind
    out["rain_commute_interaction"] = rain_log * out[f"is_commute_h{h}"].astype(float)
    return out


def fit_predict_e2(frame: pd.DataFrame, config: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[pd.DataFrame] = []
    diag_rows: List[Dict[str, object]] = []
    for period, _, _ in PERIODS:
        train, pred = period_train_pred(frame, period)
        write_status("running", f"E2 {config} fitting for {period}", current_model=f"E2/{config}")
        for h in HORIZONS:
            wtrain = add_ts_weather_features(train, h)
            wpred = add_ts_weather_features(pred, h)
            pre, _ = weather_preprocessor(h, config)
            model = fit_nb_from_preprocessor(wtrain, f"y_h{h}", pre)
            pred[f"mu_E2_raw_h{h}"] = base.predict_nb_expert(model, wpred)
            diag_rows.append({"expert": "E2", "config": config, "period": period, "horizon": h, "train_rows": len(train), "pred_rows": len(pred), "status": model.status, "reason": model.reason})
        rows.append(pred[["utc_hour"] + [f"mu_E2_raw_h{h}" for h in HORIZONS]].copy())
    return pd.concat(rows, ignore_index=True), pd.DataFrame(diag_rows)


def add_mser_features(frame: pd.DataFrame, h: int, variant: str) -> pd.DataFrame:
    out = frame.copy()
    out["rain_now_log"] = np.log1p(out["rain"].fillna(0.0))
    out["ongoing_rain_duration_log"] = np.log1p(out["current_rain_duration"].fillna(0.0))
    out["ongoing_rain_cum_log"] = np.where(out["shock_active"].eq(1), np.log1p(out["rain_sum_6"].fillna(0.0)), 0.0)
    elapsed = np.where(out["recovery_active"].eq(1), out["hours_since_rain_end"].fillna(999.0).to_numpy(float) + h, 72.0)
    elapsed = np.minimum(elapsed, 72.0)
    out["ended_duration_log"] = np.where(out["recovery_active"].eq(1), np.log1p(out["ended_rain_duration"].fillna(0.0)), 0.0)
    out["ended_cum_log"] = np.where(out["recovery_active"].eq(1), np.log1p(out["ended_rain_cum"].fillna(0.0)), 0.0)
    out["hours_since_end_log"] = np.where(out["recovery_active"].eq(1), np.log1p(elapsed), 0.0)
    for tau in [3.0, 12.0, 36.0]:
        decay = np.exp(-elapsed / tau)
        out[f"event_cum_decay_tau{int(tau)}"] = out["ended_cum_log"] * decay
        out[f"event_duration_decay_tau{int(tau)}"] = out["ended_duration_log"] * decay
        out[f"event_cd_decay_tau{int(tau)}"] = out["ended_cum_log"] * out["ended_duration_log"] * decay
        memory = np.zeros(len(out), dtype=float)
        if variant in {"memory", "full", "history_only"}:
            for lag in range(1, 73):
                memory += np.log1p(out[f"rain_lag_{lag}"].fillna(0.0).to_numpy(float)) * math.exp(-(lag + h) / tau)
        out[f"continuous_memory_tau{int(tau)}"] = memory
    return out


def mser_cols(h: int, variant: str) -> Tuple[List[str], List[str]]:
    numeric = public_numeric_cols(h) + ["ended_duration_log", "ended_cum_log", "hours_since_end_log"]
    if variant != "history_only":
        numeric += ["rain_indicator", "rain_now_log", "ongoing_rain_duration_log", "ongoing_rain_cum_log"]
    numeric += [f"event_cum_decay_tau{tau}" for tau in [3, 12, 36]]
    if variant in {"memory", "full", "history_only"}:
        numeric += [f"continuous_memory_tau{tau}" for tau in [3, 12, 36]]
    if variant == "full":
        numeric += [f"event_duration_decay_tau{tau}" for tau in [3, 12, 36]]
        numeric += [f"event_cd_decay_tau{tau}" for tau in [3, 12, 36]]
    cat = [f"target_hour_h{h}", f"target_dow_h{h}", f"target_month_h{h}"]
    return numeric, cat


def fit_predict_e3(frame: pd.DataFrame, config: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[pd.DataFrame] = []
    diag_rows: List[Dict[str, object]] = []
    for period, _, _ in PERIODS:
        train, pred = period_train_pred(frame, period)
        write_status("running", f"E3 {config} fitting for {period}", current_model=f"E3/{config}")
        for h in HORIZONS:
            rtrain = add_mser_features(train, h, config)
            rpred = add_mser_features(pred, h, config)
            num, cat = mser_cols(h, config)
            model = base.fit_nb_expert(rtrain, f"y_h{h}", num, cat)
            pred[f"mu_E3_raw_h{h}"] = base.predict_nb_expert(model, rpred)
            diag_rows.append({"expert": "E3", "config": config, "period": period, "horizon": h, "train_rows": len(train), "pred_rows": len(pred), "status": model.status, "reason": model.reason})
        rows.append(pred[["utc_hour"] + [f"mu_E3_raw_h{h}" for h in HORIZONS]].copy())
    return pd.concat(rows, ignore_index=True), pd.DataFrame(diag_rows)


DEEP_FEATURES = [
    "departures",
    "temperature_2m",
    "rain",
    "relative_humidity_2m",
    "wind_speed_10m",
    "shock_active",
    "recovery_active",
    "current_rain_duration",
    "ended_rain_cum",
    "hours_since_rain_end",
    "issue_hour_sin",
    "issue_hour_cos",
    "issue_dow_sin",
    "issue_dow_cos",
    "issue_commute",
]


class CausalConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=self.pad, dilation=dilation)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.pad:
            y = y[:, :, :-self.pad]
        return x + self.drop(self.act(y))


class TCNExpert(nn.Module):
    def __init__(self, input_dim: int, channels: int, blocks: int, dropout: float, output_bias: np.ndarray) -> None:
        super().__init__()
        self.inp = nn.Conv1d(input_dim, channels, 1)
        layers = [CausalConvBlock(channels, 3, 2**i, dropout) for i in range(blocks)]
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(channels, len(HORIZONS))
        nn.init.constant_(self.head.weight, 0.0)
        self.head.bias.data.copy_(torch.tensor(output_bias, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        z = self.net(torch.relu(self.inp(z)))
        return self.head(z[:, :, -1])


class GRUExpert(nn.Module):
    def __init__(self, input_dim: int, hidden: int, layers: int, dropout: float, output_bias: np.ndarray) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, len(HORIZONS))
        nn.init.constant_(self.head.weight, 0.0)
        self.head.bias.data.copy_(torch.tensor(output_bias, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return self.head(self.drop(h[-1]))


class TransformerExpert(nn.Module):
    def __init__(self, input_dim: int, d_model: int, layers: int, heads: int, ffn: int, dropout: float, window: int, output_bias: np.ndarray) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, window, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=ffn, dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Linear(d_model, len(HORIZONS))
        nn.init.constant_(self.head.weight, 0.0)
        self.head.bias.data.copy_(torch.tensor(output_bias, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x) + self.pos[:, : x.shape[1], :]
        z = self.encoder(z)
        return self.head(z[:, -1])


def nb_torch_loss(eta: torch.Tensor, y: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    eta = torch.clamp(eta, -20.0, 20.0)
    mu = torch.exp(eta)
    r = 1.0 / alpha
    p = r / (r + mu)
    return -(torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1.0) + r * torch.log(p) + y * torch.log1p(-p)).mean()


@dataclass
class DeepConfig:
    expert: str
    config: str
    window: int
    params: Dict[str, object]


def deep_configs() -> Dict[str, List[DeepConfig]]:
    return {
        "E4": [
            DeepConfig("E4", "tcn_w72_c32_b3", 72, {"channels": 32, "blocks": 3, "dropout": 0.1}),
            DeepConfig("E4", "tcn_w168_c32_b3", 168, {"channels": 32, "blocks": 3, "dropout": 0.1}),
            DeepConfig("E4", "tcn_w72_c64_b3", 72, {"channels": 64, "blocks": 3, "dropout": 0.1}),
            DeepConfig("E4", "tcn_w168_c64_b4", 168, {"channels": 64, "blocks": 4, "dropout": 0.1}),
        ],
        "E5": [
            DeepConfig("E5", "gru_w72_h32_l1", 72, {"hidden": 32, "layers": 1, "dropout": 0.1}),
            DeepConfig("E5", "gru_w168_h32_l1", 168, {"hidden": 32, "layers": 1, "dropout": 0.1}),
            DeepConfig("E5", "gru_w72_h64_l1", 72, {"hidden": 64, "layers": 1, "dropout": 0.1}),
            DeepConfig("E5", "gru_w168_h64_l2", 168, {"hidden": 64, "layers": 2, "dropout": 0.1}),
        ],
        "E6": [
            DeepConfig("E6", "tf_w72_d32_l1_h2", 72, {"d_model": 32, "layers": 1, "heads": 2, "ffn": 64, "dropout": 0.1}),
            DeepConfig("E6", "tf_w168_d32_l1_h2", 168, {"d_model": 32, "layers": 1, "heads": 2, "ffn": 64, "dropout": 0.1}),
            DeepConfig("E6", "tf_w72_d64_l2_h4", 72, {"d_model": 64, "layers": 2, "heads": 4, "ffn": 128, "dropout": 0.1}),
            DeepConfig("E6", "tf_w168_d64_l2_h4", 168, {"d_model": 64, "layers": 2, "heads": 4, "ffn": 128, "dropout": 0.1}),
        ],
    }


def make_deep_model(cfg: DeepConfig, input_dim: int, output_bias: np.ndarray) -> nn.Module:
    if cfg.expert == "E4":
        return TCNExpert(input_dim, int(cfg.params["channels"]), int(cfg.params["blocks"]), float(cfg.params["dropout"]), output_bias)
    if cfg.expert == "E5":
        return GRUExpert(input_dim, int(cfg.params["hidden"]), int(cfg.params["layers"]), float(cfg.params["dropout"]), output_bias)
    return TransformerExpert(
        input_dim,
        int(cfg.params["d_model"]),
        int(cfg.params["layers"]),
        int(cfg.params["heads"]),
        int(cfg.params["ffn"]),
        float(cfg.params["dropout"]),
        cfg.window,
        output_bias,
    )


def sequence_arrays(frame: pd.DataFrame, positions: np.ndarray, scaler: StandardScaler, window: int) -> Tuple[np.ndarray, np.ndarray]:
    features = frame[DEEP_FEATURES].replace(999.0, np.nan).ffill().fillna(0.0)
    values = scaler.transform(features)
    xs, ys = [], []
    for pos in positions:
        if pos < window - 1:
            continue
        xs.append(values[pos - window + 1 : pos + 1])
        ys.append(frame.iloc[pos][[f"y_h{h}" for h in HORIZONS]].to_numpy(float))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def train_deep_for_period(frame: pd.DataFrame, cfg: DeepConfig, period: str, seed: int, max_epochs: int, patience: int) -> Tuple[pd.DataFrame, Dict[str, object]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train, pred = period_train_pred(frame, period)
    train_positions = train.index.to_numpy()
    pred_positions = pred.index.to_numpy()
    train_valid = train_positions[train_positions >= cfg.window - 1]
    split = max(1, int(len(train_valid) * 0.82))
    fit_pos, val_pos = train_valid[:split], train_valid[split:]
    scaler = StandardScaler().fit(frame.loc[fit_pos, DEEP_FEATURES].replace(999.0, np.nan).ffill().fillna(0.0))
    x_fit, y_fit = sequence_arrays(frame.fillna(0.0), fit_pos, scaler, cfg.window)
    x_val, y_val = sequence_arrays(frame.fillna(0.0), val_pos, scaler, cfg.window)
    x_pred, _ = sequence_arrays(frame.fillna(0.0), pred_positions[pred_positions >= cfg.window - 1], scaler, cfg.window)
    pred_index = pred.index[pred_positions >= cfg.window - 1]

    y_mean = np.maximum(y_fit.mean(axis=0), 1e-6)
    alpha = np.asarray([base.monthly.estimate_alpha(y_fit[:, i], np.full(len(y_fit), y_mean[i])) for i in range(len(HORIZONS))], dtype=np.float32)
    model = make_deep_model(cfg, x_fit.shape[-1], np.log(y_mean)).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=1e-4)
    train_loader = DataLoader(TensorDataset(torch.tensor(x_fit), torch.tensor(y_fit)), batch_size=64 if DEVICE == "cuda" else 128, shuffle=True)
    val_x = torch.tensor(x_val, device=DEVICE)
    val_y = torch.tensor(y_val, device=DEVICE)
    alpha_t = torch.tensor(alpha, device=DEVICE)
    scaler_amp = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)
    best_loss = float("inf")
    best_state = None
    bad = 0
    start_time = time.perf_counter()
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=AMP_ENABLED):
                loss = nb_torch_loss(model(xb), yb, alpha_t)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        model.eval()
        with torch.no_grad():
            val_loss = float(nb_torch_loss(model(val_x), val_y, alpha_t).detach().cpu()) if len(val_x) else 0.0
        if val_loss + 1e-5 < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break
    train_time = time.perf_counter() - start_time
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    infer_start = time.perf_counter()
    preds = []
    with torch.no_grad():
        for offset in range(0, len(x_pred), 256):
            xb = torch.tensor(x_pred[offset : offset + 256], device=DEVICE)
            eta = torch.clamp(model(xb), -20.0, 20.0)
            preds.append(torch.exp(eta).detach().cpu().numpy())
    inference_time = time.perf_counter() - infer_start
    mu = np.vstack(preds) if preds else np.zeros((0, len(HORIZONS)))
    out = frame.loc[pred_index, ["utc_hour"]].copy()
    for j, h in enumerate(HORIZONS):
        out[f"mu_{cfg.expert}_raw_h{h}"] = mu[:, j]
    peak_mem = float(torch.cuda.max_memory_allocated() / (1024**2)) if DEVICE == "cuda" else 0.0
    diag = {
        "expert": cfg.expert,
        "config": cfg.config,
        "period": period,
        "seed": seed,
        "train_rows": len(x_fit),
        "val_rows": len(x_val),
        "pred_rows": len(out),
        "epochs": epoch + 1,
        "best_val_nb_nll": best_loss,
        "params": param_count,
        "train_time_sec": train_time,
        "inference_time_sec": inference_time,
        "peak_gpu_memory_mb": peak_mem,
        "device": DEVICE,
        "status": "ok",
    }
    return out, diag


def fit_predict_deep(frame: pd.DataFrame, cfg: DeepConfig, seed: int, max_epochs: int, patience: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[pd.DataFrame] = []
    diag_rows: List[Dict[str, object]] = []
    for period, _, _ in PERIODS:
        write_status("running", f"{cfg.expert} {cfg.config} fitting for {period}", current_model=f"{cfg.expert}/{cfg.config}", seed=seed)
        pred, diag = train_deep_for_period(frame, cfg, period, seed, max_epochs, patience)
        rows.append(pred)
        diag_rows.append(diag)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(diag_rows)


def fit_calibrators(wide: pd.DataFrame, experts: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = wide.copy()
    rows = []
    fit = out[out["split"].eq("calibration_fit")]
    for h in HORIZONS:
        y = fit[f"y_h{h}"].to_numpy(float)
        for expert in experts:
            raw_col = f"mu_{expert}_raw_h{h}"
            cal = base.fit_calibrator(y, fit[raw_col].to_numpy(float))
            out[f"mu_{expert}_cal_h{h}"] = np.maximum(0.0, cal["a"] + cal["b"] * out[raw_col].to_numpy(float))
            rows.append({"horizon": h, "expert": expert, **cal})
    return out, pd.DataFrame(rows)


def solve_positive_weights(y: np.ndarray, X: np.ndarray, center: np.ndarray | None = None, shrink: float = 0.0) -> Dict[str, object]:
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    n = X.shape[1]
    if center is None:
        center = np.ones(n) / n
    center = np.asarray(center, dtype=float)
    center = np.clip(center, 1e-8, None)
    center = center / center.sum()
    if ok.sum() < max(80, n * 20):
        return {"weights": center, "status": "insufficient_data"}
    y, X = y[ok], X[ok]

    def softmax(z: np.ndarray) -> np.ndarray:
        z = z - np.max(z)
        ez = np.exp(z)
        return ez / ez.sum()

    def obj(z: np.ndarray) -> float:
        w = softmax(z)
        return float(np.mean((y - X @ w) ** 2) + shrink * np.sum((w - center) ** 2))

    res = minimize(obj, np.log(center), method="L-BFGS-B", options={"maxiter": 500})
    if not res.success:
        return {"weights": center, "status": f"fallback_{res.message}"}
    return {"weights": softmax(res.x), "status": "ok"}


def apply_combo(wide: pd.DataFrame, combo_id: str, experts: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cal, cal_rows = fit_calibrators(wide, experts)
    fit = cal[cal["split"].eq("fusion_fit")]
    out = cal.copy()
    weight_rows = []
    for h in HORIZONS:
        cols = [f"mu_{expert}_cal_h{h}" for expert in experts]
        if len(experts) == 1:
            for expert in experts:
                out[f"w_{expert}_h{h}"] = 1.0
            out[f"mu_{combo_id}_h{h}"] = out[cols[0]].to_numpy(float)
            weight_rows.append({"combo": combo_id, "horizon": h, "state": "global", "status": "single_expert", **{f"w_{e}": 1.0 for e in experts}, "fit_rows": len(fit)})
            continue
        global_sol = solve_positive_weights(fit[f"y_h{h}"].to_numpy(float), fit[cols].to_numpy(float))
        state_weights = {"normal": global_sol["weights"], "shock": global_sol["weights"], "recovery": global_sol["weights"]}
        weight_rows.append({"combo": combo_id, "horizon": h, "state": "global", "status": global_sol["status"], "fit_rows": len(fit), **{f"w_{e}": global_sol["weights"][i] for i, e in enumerate(experts)}})
        for state in ["normal", "shock", "recovery"]:
            sub = fit[fit["state_for_fusion"].eq(state)]
            sol = solve_positive_weights(sub[f"y_h{h}"].to_numpy(float), sub[cols].to_numpy(float), center=global_sol["weights"], shrink=0.5)
            state_weights[state] = sol["weights"]
            weight_rows.append({"combo": combo_id, "horizon": h, "state": state, "status": sol["status"], "fit_rows": len(sub), **{f"w_{e}": sol["weights"][i] for i, e in enumerate(experts)}})
        for expert in experts:
            out[f"w_{expert}_h{h}"] = np.nan
        pred = np.zeros(len(out), dtype=float)
        for state, w in state_weights.items():
            idx = out["state_for_fusion"].eq(state)
            pred[idx] = out.loc[idx, cols].to_numpy(float) @ w
            for i, expert in enumerate(experts):
                out.loc[idx, f"w_{expert}_h{h}"] = w[i]
        out[f"mu_{combo_id}_h{h}"] = pred
    return out, pd.DataFrame(weight_rows), cal_rows


def combo_long(wide: pd.DataFrame, combo_id: str, experts: List[str]) -> pd.DataFrame:
    rows = []
    base_cols = ["utc_hour", "departures", "split", "state_for_fusion", "issue_stage", "shock_active", "recovery_active", "rain", "temperature_2m", "eval_event_cluster_id"]
    for h in HORIZONS:
        tmp = wide[base_cols + [f"target_time_h{h}", f"target_actual_stage_h{h}", f"y_h{h}", f"mu_{combo_id}_h{h}"]].copy()
        tmp = tmp.rename(columns={f"target_time_h{h}": "target_time", f"target_actual_stage_h{h}": "target_actual_stage", f"y_h{h}": "actual", f"mu_{combo_id}_h{h}": "prediction"})
        tmp["horizon"] = h
        tmp["combo"] = combo_id
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True).sort_values(["utc_hour", "horizon"]).reset_index(drop=True)


def point_metrics(long_df: pd.DataFrame, eval_split: str) -> pd.DataFrame:
    final = long_df[long_df["split"].eq(eval_split)].copy()
    rows = []
    for (combo, h), g in final.groupby(["combo", "horizon"]):
        for group, mask in [
            ("overall", np.ones(len(g), dtype=bool)),
            ("normal", g["state_for_fusion"].eq("normal").to_numpy()),
            ("shock", g["state_for_fusion"].eq("shock").to_numpy()),
            ("recovery", g["state_for_fusion"].eq("recovery").to_numpy()),
        ]:
            sub = g[mask]
            if sub.empty:
                continue
            rows.append({"combo": combo, "horizon": h, "group": group, "n_hours": len(sub), "MAE": mean_absolute_error(sub["actual"], sub["prediction"]), "RMSE": rmse(sub["actual"], sub["prediction"])})
    return pd.DataFrame(rows)


def selection_score(dev_metrics: pd.DataFrame, baseline_dev: pd.DataFrame, combo_id: str) -> Dict[str, float]:
    cur = dev_metrics[dev_metrics["combo"].eq(combo_id)]
    merged = cur.merge(baseline_dev, on=["horizon", "group"], suffixes=("", "_base"))
    def ratio(group: str, metric: str = "MAE") -> float:
        sub = merged[merged["group"].eq(group)]
        return float((sub[metric] / sub[f"{metric}_base"]).mean()) if not sub.empty else float("inf")
    score = 0.45 * ratio("overall") + 0.25 * ratio("shock") + 0.20 * ratio("recovery") + 0.10 * ratio("overall", "RMSE")
    return {"selection_score": score, "overall_ratio": ratio("overall"), "shock_ratio": ratio("shock"), "recovery_ratio": ratio("recovery"), "rmse_ratio": ratio("overall", "RMSE")}


def residual_correlation(calibrated: pd.DataFrame, experts: List[str]) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        sub = calibrated[calibrated["split"].eq("final_eval")].copy()
        residuals = {}
        for expert in experts:
            col = f"mu_{expert}_cal_h{h}"
            if col in sub:
                residuals[expert] = sub[f"y_h{h}"].to_numpy(float) - sub[col].to_numpy(float)
        corr = pd.DataFrame(residuals).corr()
        for left in corr.index:
            for right in corr.columns:
                rows.append({"horizon": h, "expert_i": left, "expert_j": right, "corr": float(corr.loc[left, right])})
    return pd.DataFrame(rows)


def make_main_table(metrics: pd.DataFrame) -> pd.DataFrame:
    overall = metrics[metrics["group"].eq("overall")].copy()
    mae = overall.pivot(index="combo", columns="horizon", values="MAE").rename(columns={h: f"{h}h_MAE" for h in HORIZONS})
    rmse_p = overall.pivot(index="combo", columns="horizon", values="RMSE")
    out = mae.reset_index()
    out["Avg_MAE"] = out[[f"{h}h_MAE" for h in HORIZONS]].mean(axis=1)
    out["Avg_RMSE"] = rmse_p.mean(axis=1).reindex(out["combo"]).to_numpy()
    return out.sort_values("Avg_MAE")


def write_figures(main_table: pd.DataFrame, metrics: pd.DataFrame, selected: pd.DataFrame, corr: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(11, 5))
    plot = main_table.melt(id_vars=["combo"], value_vars=[f"{h}h_MAE" for h in HORIZONS], var_name="horizon", value_name="MAE")
    sns.barplot(data=plot, x="horizon", y="MAE", hue="combo")
    plt.title("Incremental ablation: E1 versus E1 plus one expert")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_incremental_ablation_mae.png", dpi=220)
    plt.close()

    state = metrics[metrics["group"].isin(["normal", "shock", "recovery"])].copy()
    state_avg = state.groupby(["combo", "group"], as_index=False)["MAE"].mean()
    plt.figure(figsize=(10.5, 5))
    sns.barplot(data=state_avg, x="group", y="MAE", hue="combo")
    plt.title("State grouped MAE averaged across horizons")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_state_grouped_mae.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9, 4.5))
    sns.barplot(data=selected, x="candidate", y="selection_score", hue="selected_config")
    plt.title("Screening selection score before final evaluation")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_screening_scores.png", dpi=220)
    plt.close()

    c = corr[(corr["horizon"].eq(1))].pivot(index="expert_i", columns="expert_j", values="corr")
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(c, vmin=-1, vmax=1, cmap="vlag", annot=True, fmt=".2f")
    plt.title("Final residual correlation, h=1")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_residual_correlation_h1.png", dpi=220)
    plt.close()


def write_report(main_table: pd.DataFrame, metrics: pd.DataFrame, selected: pd.DataFrame, engineering: pd.DataFrame, corr: pd.DataFrame, validation: Dict[str, object]) -> None:
    report = f"""# Incremental Expert Screening Report

Run id: `{RUN_ID}`

This run follows the new protocol: E1 is fixed as the multi-output negative-binomial XGBoost demand expert with horizons `{HORIZONS}`. Each candidate expert is tested only as an addition to E1: A0=E1, A1=E1+E2, A2=E1+E3, A3=E1+E4, A4=E1+E5, A5=E1+E6.

All feature information respects the issue-time boundary. Historical demand, weather, and event fields are read only up to issue time `t`; target calendar fields are known in advance.

## Validation

```json
{json.dumps(validation, ensure_ascii=False, indent=2)}
```

## Selected Candidate Configurations

{selected.to_markdown(index=False)}

## Main Ablation Table

{main_table.to_markdown(index=False)}

## Final State Metrics

{metrics.sort_values(["combo", "horizon", "group"]).to_markdown(index=False)}

## Engineering Metrics

{engineering.to_markdown(index=False)}

## Residual Correlation

{corr.to_markdown(index=False)}

## Figures

- `01_incremental_ablation_mae.png`
- `02_state_grouped_mae.png`
- `03_screening_scores.png`
- `04_residual_correlation_h1.png`
"""
    for path in [REPORT_DIR / "incremental_expert_report.md", REPORT_ARCHIVE_DIR / "incremental_expert_report.md"]:
        path.write_text(report, encoding="utf-8")


def update_archive_index() -> None:
    index = ROOT / "reports" / "six_expert_experiments" / "README.md"
    line = f"| `{RUN_ID}` | Incremental E1+candidate expert screening over h=1,2,3,6. | `{RUN_ID}/incremental_expert_report.md` |\n"
    text = index.read_text(encoding="utf-8") if index.exists() else "# Three-Expert Experiment Reports\n\n## Runs\n\n| Run id | Main change | Report |\n| --- | --- | --- |\n"
    if RUN_ID not in text:
        index.write_text(text.rstrip() + "\n" + line, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    ensure_dirs()
    write_status("starting", "Starting incremental expert screening", horizons=HORIZONS)
    frame = load_frame()
    frame.to_csv(TABLE_DIR / "incremental_model_frame.csv", index=False, encoding="utf-8-sig")
    wide, e1_diag = fit_predict_e1(frame)
    wide.to_csv(TABLE_DIR / "raw_predictions_E1.csv", index=False, encoding="utf-8-sig")
    diagnostics = [e1_diag]
    screening_rows = []
    raw_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    base_wide, _, _ = apply_combo(wide.copy(), "A0_E1", ["E1"])
    base_dev = point_metrics(combo_long(base_wide, "A0_E1", ["E1"]), "screening_dev")

    configs: List[Tuple[str, str]] = [("E2", c) for c in ["spline_only", "spline_th", "full_tensor"]]
    configs += [("E3", c) for c in ["multi_scale", "memory", "full"]]
    for expert, config in configs:
        if expert == "E2":
            pred, diag = fit_predict_e2(frame, config)
        else:
            pred, diag = fit_predict_e3(frame, config)
        raw_cache[(expert, config)] = pred
        diagnostics.append(diag)
        tmp = wide.merge(pred, on="utc_hour", how="left")
        combo, _, _ = apply_combo(tmp, f"screen_{expert}_{config}", ["E1", expert])
        dev = point_metrics(combo_long(combo, f"screen_{expert}_{config}", ["E1", expert]), "screening_dev")
        screening_rows.append({"candidate": expert, "config": config, **selection_score(dev, base_dev, f"screen_{expert}_{config}")})
        pd.DataFrame(screening_rows).to_csv(TABLE_DIR / "screening_scores_partial.csv", index=False, encoding="utf-8-sig")

    for expert, cfgs in deep_configs().items():
        for cfg in cfgs:
            pred, diag = fit_predict_deep(frame, cfg, SEED, args.max_epochs, args.patience)
            raw_cache[(expert, cfg.config)] = pred
            diagnostics.append(diag)
            tmp = wide.merge(pred, on="utc_hour", how="left")
            combo, _, _ = apply_combo(tmp, f"screen_{expert}_{cfg.config}", ["E1", expert])
            dev = point_metrics(combo_long(combo, f"screen_{expert}_{cfg.config}", ["E1", expert]), "screening_dev")
            screening_rows.append({"candidate": expert, "config": cfg.config, **selection_score(dev, base_dev, f"screen_{expert}_{cfg.config}")})
            pd.DataFrame(screening_rows).to_csv(TABLE_DIR / "screening_scores_partial.csv", index=False, encoding="utf-8-sig")

    screening = pd.DataFrame(screening_rows).sort_values(["candidate", "selection_score"])
    screening.to_csv(TABLE_DIR / "screening_scores.csv", index=False, encoding="utf-8-sig")
    selected = screening.groupby("candidate", as_index=False).first().rename(columns={"config": "selected_config"})
    selected.to_csv(TABLE_DIR / "selected_candidate_configs.csv", index=False, encoding="utf-8-sig")

    combo_longs = []
    weight_rows = []
    calibrator_rows = []
    calibrated_for_corr = None
    combos = [("A0_E1", ["E1"], wide.copy())]
    for idx, row in selected.iterrows():
        expert = str(row["candidate"])
        config = str(row["selected_config"])
        aid = {"E2": "A1_E1_E2", "E3": "A2_E1_E3", "E4": "A3_E1_E4", "E5": "A4_E1_E5", "E6": "A5_E1_E6"}[expert]
        combos.append((aid, ["E1", expert], wide.merge(raw_cache[(expert, config)], on="utc_hour", how="left")))

    for combo_id, experts, combo_wide in combos:
        write_status("running", f"Applying calibration and fusion for {combo_id}", current_model=combo_id)
        fitted, weights, cals = apply_combo(combo_wide, combo_id, experts)
        combo_longs.append(combo_long(fitted, combo_id, experts))
        weight_rows.append(weights)
        calibrator_rows.append(cals.assign(combo=combo_id))
        if combo_id != "A0_E1":
            calibrated_for_corr = fitted if calibrated_for_corr is None else calibrated_for_corr

    all_long = pd.concat(combo_longs, ignore_index=True)
    final_metrics = point_metrics(all_long, "final_eval")
    main_table = make_main_table(final_metrics)
    weights = pd.concat(weight_rows, ignore_index=True)
    cals = pd.concat(calibrator_rows, ignore_index=True)
    diagnostics_df = pd.concat(diagnostics, ignore_index=True)
    engineering = diagnostics_df.groupby(["expert", "config"], as_index=False).agg(
        params=("params", "max"),
        train_time_sec=("train_time_sec", "sum"),
        inference_time_sec=("inference_time_sec", "sum"),
        peak_gpu_memory_mb=("peak_gpu_memory_mb", "max"),
        status=("status", lambda x: ",".join(sorted(set(map(str, x))))),
    )
    selected_experts = ["E1"] + selected["candidate"].tolist()
    corr_source = wide.copy()
    for _, row in selected.iterrows():
        corr_source = corr_source.merge(raw_cache[(str(row["candidate"]), str(row["selected_config"]))], on="utc_hour", how="left")
    corr_cal, _ = fit_calibrators(corr_source, selected_experts)
    corr = residual_correlation(corr_cal, selected_experts)

    all_long.to_csv(TABLE_DIR / "incremental_all_combo_predictions.csv", index=False, encoding="utf-8-sig")
    final_metrics.to_csv(TABLE_DIR / "incremental_final_metrics.csv", index=False, encoding="utf-8-sig")
    main_table.to_csv(TABLE_DIR / "incremental_main_ablation_table.csv", index=False, encoding="utf-8-sig")
    weights.to_csv(TABLE_DIR / "incremental_fusion_weights.csv", index=False, encoding="utf-8-sig")
    cals.to_csv(TABLE_DIR / "incremental_calibrators.csv", index=False, encoding="utf-8-sig")
    diagnostics_df.to_csv(TABLE_DIR / "incremental_diagnostics.csv", index=False, encoding="utf-8-sig")
    engineering.to_csv(TABLE_DIR / "incremental_engineering_metrics.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(TABLE_DIR / "incremental_residual_correlation.csv", index=False, encoding="utf-8-sig")

    validation = {
        "run_id": RUN_ID,
        "horizons": HORIZONS,
        "torch_version": torch.__version__,
        "device": DEVICE,
        "amp_enabled": AMP_ENABLED,
        "screening_config_count": int(len(screening)),
        "combos": [c[0] for c in combos],
        "final_rows": int(all_long[all_long["split"].eq("final_eval")].shape[0]),
        "source_data_preserved": True,
        "selection_uses_final_eval": False,
    }
    (TABLE_DIR / "incremental_validation_checks.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    make_main_table(final_metrics)
    write_figures(main_table, final_metrics, selected, corr)
    write_report(main_table, final_metrics, selected, engineering, corr, validation)
    manifest = {
        "run_id": RUN_ID,
        "run_dir": str(RUN_DIR),
        "report": str(REPORT_ARCHIVE_DIR / "incremental_expert_report.md"),
        "status": str(STATUS_PATH),
        "horizons": HORIZONS,
        "device": DEVICE,
    }
    (RUN_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_ARCHIVE_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    update_archive_index()
    write_status("completed", "Incremental expert screening completed", report=str(REPORT_ARCHIVE_DIR / "incremental_expert_report.md"))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        ensure_dirs()
        (LOG_DIR / "error_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_status("failed", f"{type(exc).__name__}: {exc}", traceback=str(LOG_DIR / "error_traceback.txt"))
        raise
