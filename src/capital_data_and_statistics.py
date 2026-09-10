from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CAPITAL_TABLES = ROOT / "data" / "processed" / "capital_hourly"

def read_capital_cached() -> pd.DataFrame:
    departures = pd.read_csv(CAPITAL_TABLES / "capital_hourly_departures_utc.csv", parse_dates=["utc_hour"])
    weather = pd.read_csv(CAPITAL_TABLES / "capital_open_meteo_historical_weather.csv", parse_dates=["utc_hour"])
    return departures.merge(weather, on="utc_hour", how="left").sort_values("utc_hour").reset_index(drop=True)


def add_online_events(df: pd.DataFrame, dry_gap_hours: int = 2, recovery_hours: int = 24, eval_cluster_gap: int = 6) -> pd.DataFrame:
    out = df.copy()
    rain = out["rain"].fillna(0.0).to_numpy(float)
    snow = out["snowfall"].fillna(0.0).to_numpy(float)
    online_id, eval_id, stage = [], [], []
    shock_active, recovery_active = [], []
    current_duration, ended_duration, ended_cum, hours_since_end = [], [], [], []
    event_id = 0
    cluster_id = 0
    active_event = 0
    active_cluster = 0
    duration = 0.0
    total = 0.0
    last_duration = 0.0
    last_total = 0.0
    since_end = 999.0
    dry_gap = 999
    cluster_dry_gap = 999
    in_rain = False
    for r, s in zip(rain, snow):
        if r > 0:
            if active_event == 0 or dry_gap > dry_gap_hours:
                event_id += 1
                active_event = event_id
                duration = 0.0
                total = 0.0
            if active_cluster == 0 or cluster_dry_gap > eval_cluster_gap:
                cluster_id += 1
                active_cluster = cluster_id
            duration += 1.0
            total += r
            last_duration = duration
            last_total = total
            since_end = 0.0
            dry_gap = 0
            cluster_dry_gap = 0
            in_rain = True
            online_id.append(active_event)
            eval_id.append(active_cluster)
            stage.append("shock")
            shock_active.append(1.0)
            recovery_active.append(0.0)
            current_duration.append(duration)
            ended_duration.append(0.0)
            ended_cum.append(0.0)
            hours_since_end.append(0.0)
        else:
            if in_rain:
                since_end = 1.0
                in_rain = False
            elif since_end < 999:
                since_end += 1.0
            dry_gap = dry_gap + 1 if dry_gap < 999 else 999
            cluster_dry_gap = cluster_dry_gap + 1 if cluster_dry_gap < 999 else 999
            recover = 1 <= since_end <= recovery_hours and last_total > 0
            if recover:
                st = "recovery"
                oid = active_event
                cid = active_cluster
            else:
                st = "snow" if s > 0 else "normal"
                oid = 0
                cid = 0
                if dry_gap > dry_gap_hours:
                    active_event = 0
                if cluster_dry_gap > eval_cluster_gap:
                    active_cluster = 0
            online_id.append(oid)
            eval_id.append(cid)
            stage.append(st)
            shock_active.append(0.0)
            recovery_active.append(1.0 if recover else 0.0)
            current_duration.append(0.0)
            ended_duration.append(last_duration if recover else 0.0)
            ended_cum.append(last_total if recover else 0.0)
            hours_since_end.append(since_end if recover else 999.0)
    out["issue_stage"] = stage
    out["online_event_id"] = online_id
    out["eval_event_cluster_id"] = eval_id
    out["shock_active"] = shock_active
    out["recovery_active"] = recovery_active
    out["current_rain_duration"] = current_duration
    out["ended_rain_duration"] = ended_duration
    out["ended_rain_cum"] = ended_cum
    out["hours_since_rain_end"] = hours_since_end
    for lag in range(7):
        out[f"rain_lag_{lag}"] = out["rain"].fillna(0.0).shift(lag).fillna(0.0)
    # Evaluation-only target weather labels are constructed later from target-hour records.
    return out


def month_start(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts.dt.strftime("%Y-%m-01")).dt.tz_localize("UTC")


def make_samples(hourly: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = hourly.copy()
    target_idx = df.index + horizon
    valid = target_idx < len(df)
    df = df.loc[valid].copy()
    target = hourly.iloc[target_idx[valid]].reset_index(drop=True)
    df = df.reset_index(drop=True)
    issue_local = df["utc_hour"].dt.tz_convert("America/New_York")
    target_time = target["utc_hour"]
    target_local = target_time.dt.tz_convert("America/New_York")
    df["target"] = target["departures"].to_numpy(float)
    df["target_time"] = target_time
    df["target_actual_stage"] = np.where(target["rain"].fillna(0.0).gt(0), "rain", np.where(target["snowfall"].fillna(0.0).gt(0), "snow", "normal"))
    df["horizon"] = horizon
    df["issue_local_hour"] = issue_local.dt.hour
    df["target_hour"] = target_local.dt.hour
    df["target_dow"] = target_local.dt.dayofweek
    df["target_month"] = target_local.dt.month
    df["target_doy"] = target_local.dt.dayofyear
    df["is_weekend"] = df["target_dow"].isin([5, 6]).astype(float)
    df["is_commute"] = df["target_hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(float)
    df["hour_sin"] = np.sin(2 * np.pi * df["target_hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["target_hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["target_dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["target_dow"] / 7.0)
    df["year_sin"] = np.sin(2 * np.pi * df["target_doy"] / 365.25)
    df["year_cos"] = np.cos(2 * np.pi * df["target_doy"] / 365.25)
    y = hourly["departures"]
    for lag in [0, 1, 2, 3, 6, 24, 48, 168]:
        df[f"lag_{lag}"] = y.shift(lag).loc[valid].reset_index(drop=True).to_numpy(float)
        df[f"log_lag_{lag}"] = np.log1p(df[f"lag_{lag}"])
    df["roll_mean_3"] = y.shift(1).rolling(3, min_periods=3).mean().loc[valid].reset_index(drop=True)
    df["roll_mean_24"] = y.shift(1).rolling(24, min_periods=24).mean().loc[valid].reset_index(drop=True)
    df["roll_mean_168"] = y.shift(1).rolling(168, min_periods=168).mean().loc[valid].reset_index(drop=True)
    df["roll_std_24"] = y.shift(1).rolling(24, min_periods=24).std().loc[valid].reset_index(drop=True)
    df["rain_sum_3"] = hourly["rain"].fillna(0.0).rolling(3, min_periods=1).sum().loc[valid].reset_index(drop=True)
    df["rain_sum_6"] = hourly["rain"].fillna(0.0).rolling(6, min_periods=1).sum().loc[valid].reset_index(drop=True)
    df["recovery_elapsed_at_target"] = np.where(df["recovery_active"].eq(1), df["hours_since_rain_end"].to_numpy(float) + horizon, 999.0)
    df = df.dropna(subset=["target", "lag_168", "temperature_2m", "rain"]).copy()
    df["sample_id"] = np.arange(len(df))
    df["target_month_start"] = month_start(df["target_time"])
    df["month"] = df["target_month_start"].dt.strftime("%Y-%m")
    return df

def estimate_alpha(y: np.ndarray, mu: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-6, None)
    den = np.sum(mu**2)
    num = np.sum((y - mu) ** 2 - mu)
    return float(np.clip(num / den if den > 0 else 0.05, 0.01, 3.0))
