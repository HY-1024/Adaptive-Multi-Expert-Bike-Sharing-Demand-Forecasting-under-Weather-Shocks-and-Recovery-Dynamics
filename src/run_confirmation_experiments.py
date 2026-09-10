from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize

import run_incremental_expert_screening as inc


ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [1, 2, 3, 6]
EXPERTS = [f"E{i}" for i in range(1, 7)]
OBJECTIVES = ["mse", "mae", "huber", "state_huber"]
RUN_ID = os.environ.get("CONFIRMATION_RUN_ID", time.strftime("confirmation_%Y%m%d_%H%M%S"))
RUN_DIR = ROOT / "runs" / RUN_ID
TABLE_DIR = RUN_DIR / "tables"
FIG_DIR = RUN_DIR / "figures"
LOG_DIR = RUN_DIR / "logs"
STATUS_PATH = RUN_DIR / "status.json"
LATEST_STATUS_PATH = ROOT / "runs" / "latest_confirmation_status.json"
DEFAULT_INPUT = ROOT / "artifacts" / "confirmation_inputs" / "expert_predictions.csv"
SEED = 20260910


def ensure_dirs() -> None:
    for path in [RUN_DIR, TABLE_DIR, FIG_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_status(stage: str, message: str, **extra: object) -> None:
    payload = {
        "run_id": RUN_ID,
        "run_dir": str(RUN_DIR),
        "stage": stage,
        "message": message,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pid": os.getpid(),
        **extra,
    }
    for path in [STATUS_PATH, LATEST_STATUS_PATH]:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    print(f"[{payload['updated_at']}] {stage}: {message}", flush=True)


def rmse(y: Iterable[float], pred: Iterable[float]) -> float:
    y_arr = np.asarray(y, dtype=float)
    p_arr = np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((y_arr - p_arr) ** 2)))


def read_input(path: Path) -> pd.DataFrame:
    date_cols = ["utc_hour"] + [f"target_time_h{h}" for h in HORIZONS]
    frame = pd.read_csv(path, parse_dates=date_cols)
    expected = {f"mu_{e}_raw_h{h}" for e in EXPERTS for h in HORIZONS}
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"Missing expert predictions: {sorted(missing)}")
    return frame.sort_values("utc_hour").reset_index(drop=True)


def calibrate_experts(wide: pd.DataFrame, experts: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = wide.copy()
    fit = out[out["split"].eq("calibration_fit")]
    rows: List[Dict[str, object]] = []
    for h in HORIZONS:
        y = fit[f"y_h{h}"].to_numpy(float)
        for expert in experts:
            raw = f"mu_{expert}_raw_h{h}"
            cal = inc.base.fit_calibrator(y, fit[raw].to_numpy(float))
            out[f"mu_{expert}_cal_h{h}"] = np.maximum(0.0, cal["a"] + cal["b"] * out[raw].to_numpy(float))
            rows.append({"horizon": h, "expert": expert, **cal})
    return out, pd.DataFrame(rows)


def state_balance_weights(states: pd.Series) -> np.ndarray:
    counts = states.value_counts()
    raw = states.map(lambda value: 1.0 / math.sqrt(float(counts[value]))).to_numpy(float)
    raw = raw / np.mean(raw)
    return np.clip(raw, 0.5, 2.5)


def solve_weights(
    y: np.ndarray,
    X: np.ndarray,
    objective: str,
    center: np.ndarray | None = None,
    shrink: float = 0.0,
    sample_weight: np.ndarray | None = None,
) -> Dict[str, object]:
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    n_experts = X.shape[1]
    if center is None:
        center = np.ones(n_experts, dtype=float) / n_experts
    center = np.asarray(center, dtype=float)
    center = np.clip(center, 0.0, None)
    center = center / center.sum()
    if ok.sum() < max(80, n_experts * 20):
        return {"weights": center, "status": "insufficient_data", "loss": np.nan}
    y = y[ok]
    X = X[ok]
    sw = np.ones(len(y), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)[ok]
    sw = sw / np.mean(sw)
    scale = max(float(np.mean(np.abs(y))), 1.0)
    initial_resid = y - X @ center
    delta = max(float(1.5 * np.median(np.abs(initial_resid))), 1.0)

    def loss(weights: np.ndarray) -> float:
        residual = y - X @ weights
        if objective == "mse":
            base = np.average((residual / scale) ** 2, weights=sw)
        elif objective == "mae":
            base = np.average(np.abs(residual) / scale, weights=sw)
        else:
            absolute = np.abs(residual)
            huber = np.where(absolute <= delta, 0.5 * residual**2, delta * (absolute - 0.5 * delta))
            base = np.average(huber / (scale**2), weights=sw)
        return float(base + shrink * np.sum((weights - center) ** 2))

    result = minimize(
        loss,
        center,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_experts,
        constraints=[{"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)}],
        options={"maxiter": 600, "ftol": 1e-10},
    )
    weights = np.asarray(result.x if result.success else center, dtype=float)
    weights = np.clip(weights, 0.0, None)
    weights = weights / weights.sum()
    return {"weights": weights, "status": "ok" if result.success else f"fallback_{result.message}", "loss": loss(weights)}


def fit_fusion(wide: pd.DataFrame, experts: List[str], objective: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out, calibrators = calibrate_experts(wide, experts)
    fit = out[out["split"].eq("fusion_fit")]
    weight_rows: List[Dict[str, object]] = []
    for h in HORIZONS:
        columns = [f"mu_{expert}_cal_h{h}" for expert in experts]
        y = fit[f"y_h{h}"].to_numpy(float)
        X = fit[columns].to_numpy(float)
        sample_weight = state_balance_weights(fit["state_for_fusion"]) if objective == "state_huber" else None
        solver_objective = "huber" if objective == "state_huber" else objective
        global_fit = solve_weights(y, X, objective=solver_objective, sample_weight=sample_weight)
        global_weights = global_fit["weights"]
        state_weights: Dict[str, np.ndarray] = {}
        weight_rows.append({"horizon": h, "state": "global", "objective": objective, "status": global_fit["status"], **{f"w_{e}": global_weights[i] for i, e in enumerate(experts)}})
        for state in ["normal", "shock", "recovery"]:
            part = fit[fit["state_for_fusion"].eq(state)]
            state_fit = solve_weights(
                part[f"y_h{h}"].to_numpy(float),
                part[columns].to_numpy(float),
                objective=solver_objective,
                center=global_weights,
                shrink=0.5,
            )
            state_weights[state] = state_fit["weights"]
            weight_rows.append({"horizon": h, "state": state, "objective": objective, "status": state_fit["status"], **{f"w_{e}": state_fit["weights"][i] for i, e in enumerate(experts)}})
        matrix = out[columns].to_numpy(float)
        prediction = np.zeros(len(out), dtype=float)
        for state in ["normal", "shock", "recovery"]:
            mask = out["state_for_fusion"].eq(state).to_numpy()
            prediction[mask] = matrix[mask] @ state_weights[state]
        other = ~out["state_for_fusion"].isin(state_weights).to_numpy()
        prediction[other] = matrix[other] @ global_weights
        out[f"mu_fusion_h{h}"] = np.maximum(0.0, prediction)
    return out, pd.DataFrame(weight_rows), calibrators


def long_predictions(wide: pd.DataFrame, model: str) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        part = wide[["utc_hour", "split", "state_for_fusion", "eval_event_cluster_id", f"target_time_h{h}", f"y_h{h}", f"mu_fusion_h{h}"]].copy()
        part.columns = ["utc_hour", "split", "state_for_fusion", "eval_event_cluster_id", "target_time", "actual", "prediction"]
        part["horizon"] = h
        part["model"] = model
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def metric_table(long_df: pd.DataFrame, split: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    data = long_df[long_df["split"].eq(split)]
    for (model, h), hdf in data.groupby(["model", "horizon"]):
        groups = {"overall": np.ones(len(hdf), dtype=bool)}
        groups.update({state: hdf["state_for_fusion"].eq(state).to_numpy() for state in ["normal", "shock", "recovery"]})
        for group, mask in groups.items():
            part = hdf.loc[mask]
            rows.append({"split": split, "model": model, "horizon": h, "group": group, "n": len(part), "MAE": float(np.mean(np.abs(part["actual"] - part["prediction"]))), "RMSE": rmse(part["actual"], part["prediction"])})
    return pd.DataFrame(rows)


def objective_selection(metrics: pd.DataFrame) -> pd.DataFrame:
    dev = metrics[metrics["split"].eq("screening_dev")]
    baseline = dev[dev["model"].eq("mse")]
    rows = []
    for objective in OBJECTIVES:
        part = dev[dev["model"].eq(objective)]
        components: Dict[str, float] = {}
        for group in ["overall", "shock", "recovery"]:
            current = part[part["group"].eq(group)].set_index("horizon")["MAE"]
            reference = baseline[baseline["group"].eq(group)].set_index("horizon")["MAE"]
            components[group] = float((current / reference).mean())
        current_rmse = part[part["group"].eq("overall")].set_index("horizon")["RMSE"]
        reference_rmse = baseline[baseline["group"].eq("overall")].set_index("horizon")["RMSE"]
        rmse_ratio = float((current_rmse / reference_rmse).mean())
        score = 0.45 * components["overall"] + 0.25 * components["shock"] + 0.20 * components["recovery"] + 0.10 * rmse_ratio
        rows.append({"objective": objective, "selection_score": score, "overall_ratio": components["overall"], "shock_ratio": components["shock"], "recovery_ratio": components["recovery"], "rmse_ratio": rmse_ratio})
    return pd.DataFrame(rows).sort_values("selection_score").reset_index(drop=True)


def resample_mean(values_by_cluster: List[np.ndarray], rng: np.random.Generator, n_boot: int) -> Tuple[float, float]:
    if len(values_by_cluster) < 2:
        return np.nan, np.nan
    samples = []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(values_by_cluster), len(values_by_cluster))
        values = np.concatenate([values_by_cluster[i] for i in chosen])
        samples.append(float(np.mean(values)))
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def loo_bootstrap(full: pd.DataFrame, alternatives: Dict[str, pd.DataFrame], n_boot: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    full_final = full[full["split"].eq("final_eval")]
    for model, alternative in alternatives.items():
        alt_final = alternative[alternative["split"].eq("final_eval")]
        for h in HORIZONS:
            left = full_final[full_final["horizon"].eq(h)].set_index("utc_hour")
            right = alt_final[alt_final["horizon"].eq(h)].set_index("utc_hour")
            joined = left[["actual", "prediction", "target_time", "state_for_fusion", "eval_event_cluster_id"]].join(right[["prediction"]], rsuffix="_loo")
            joined["delta"] = np.abs(joined["actual"] - joined["prediction_loo"]) - np.abs(joined["actual"] - joined["prediction"])
            for group in ["overall", "normal", "shock", "recovery"]:
                part = joined if group == "overall" else joined[joined["state_for_fusion"].eq(group)]
                if group in {"shock", "recovery"}:
                    grouped = [g["delta"].to_numpy(float) for _, g in part[part["eval_event_cluster_id"].gt(0)].groupby("eval_event_cluster_id")]
                    cluster_type = "event_cluster"
                else:
                    block = part["target_time"].dt.tz_convert("UTC").dt.strftime("%Y-%U")
                    grouped = [g["delta"].to_numpy(float) for _, g in part.assign(block=block).groupby("block")]
                    cluster_type = "7_day_block"
                low, high = resample_mean(grouped, rng, n_boot)
                rows.append({"removed_expert": model.removeprefix("minus_"), "horizon": h, "group": group, "n": len(part), "cluster_type": cluster_type, "clusters": len(grouped), "delta_MAE_loo_minus_full": float(part["delta"].mean()), "ci95_low": low, "ci95_high": high})
    return pd.DataFrame(rows)


def empirical_quantile(values: Iterable[float], level: float) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return 0.0
    adjusted = min(1.0, math.ceil((len(array) + 1) * level) / len(array))
    return float(np.quantile(array, adjusted, method="higher"))


def rolling_intervals(long_df: pd.DataFrame, method: str, window: int = 2160, min_state_pool: int = 120) -> pd.DataFrame:
    outputs = []
    for h in HORIZONS:
        history = long_df[(long_df["horizon"].eq(h)) & (long_df["split"].eq("screening_dev"))].copy()
        final = long_df[(long_df["horizon"].eq(h)) & (long_df["split"].eq("final_eval"))].copy().sort_values("utc_hour").reset_index(drop=True)
        all_rows = pd.concat([history, final], ignore_index=True)
        all_rows["score"] = np.abs(all_rows["actual"] - all_rows["prediction"]) / np.maximum(np.sqrt(np.maximum(all_rows["prediction"], 0.0)), 25.0)
        candidates = all_rows[["target_time", "state_for_fusion", "score"]].sort_values("target_time").reset_index(drop=True)
        global_pool: deque[float] = deque(maxlen=window)
        state_pools: Dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))
        pointer = 0
        q_values = {90: [], 95: []}
        pool_sizes = []
        for _, row in final.iterrows():
            issue = row["utc_hour"]
            while pointer < len(candidates) and candidates.iloc[pointer]["target_time"] <= issue:
                candidate = candidates.iloc[pointer]
                score = float(candidate["score"])
                state = str(candidate["state_for_fusion"])
                global_pool.append(score)
                state_pools[state].append(score)
                pointer += 1
            state_pool = state_pools[str(row["state_for_fusion"])]
            active = state_pool if method == "state_stratified" and len(state_pool) >= min_state_pool else global_pool
            pool_sizes.append(len(active))
            q_values[90].append(empirical_quantile(active, 0.90))
            q_values[95].append(empirical_quantile(active, 0.95))
        scale = np.maximum(np.sqrt(np.maximum(final["prediction"].to_numpy(float), 0.0)), 25.0)
        final["interval_method"] = method
        final["pool_size"] = pool_sizes
        for level in [90, 95]:
            half_width = np.asarray(q_values[level]) * scale
            final[f"lo_{level}"] = np.maximum(0.0, final["prediction"].to_numpy(float) - half_width)
            final[f"hi_{level}"] = final["prediction"].to_numpy(float) + half_width
        outputs.append(final)
    return pd.concat(outputs, ignore_index=True)


def interval_metrics(intervals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, h), hdf in intervals.groupby(["interval_method", "horizon"]):
        for group in ["overall", "normal", "shock", "recovery"]:
            part = hdf if group == "overall" else hdf[hdf["state_for_fusion"].eq(group)]
            for level in [90, 95]:
                covered = part["actual"].between(part[f"lo_{level}"], part[f"hi_{level}"])
                rows.append({"method": method, "horizon": h, "group": group, "level": level, "n": len(part), "coverage": float(covered.mean()), "mean_width": float((part[f"hi_{level}"] - part[f"lo_{level}"]).mean()), "median_pool_size": float(part["pool_size"].median())})
    return pd.DataFrame(rows)


def interval_cluster_bootstrap(intervals: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for (method, h), hdf in intervals.groupby(["interval_method", "horizon"]):
        for group in ["shock", "recovery"]:
            part = hdf[(hdf["state_for_fusion"].eq(group)) & hdf["eval_event_cluster_id"].gt(0)].copy()
            for level in [90, 95]:
                part["covered"] = part["actual"].between(part[f"lo_{level}"], part[f"hi_{level}"]).astype(float)
                clusters = [g["covered"].to_numpy(float) for _, g in part.groupby("eval_event_cluster_id")]
                low, high = resample_mean(clusters, rng, n_boot)
                rows.append({"method": method, "horizon": h, "group": group, "level": level, "clusters": len(clusters), "coverage": float(part["covered"].mean()), "ci95_low": low, "ci95_high": high})
    return pd.DataFrame(rows)


def partial_correlation(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    design = np.column_stack([np.ones(len(control)), control])
    residual_x = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    residual_y = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return float(np.corrcoef(residual_x, residual_y)[0, 1])


def complementarity(wide: pd.DataFrame, variant: str) -> pd.DataFrame:
    rows = []
    for split in ["screening_dev", "final_eval"]:
        part = wide[wide["split"].eq(split)]
        for h in HORIZONS:
            y = part[f"y_h{h}"].to_numpy(float)
            e1 = y - part[f"mu_E1_cal_h{h}"].to_numpy(float)
            e2 = y - part[f"mu_E2_cal_h{h}"].to_numpy(float)
            e3 = y - part[f"mu_E3_cal_h{h}"].to_numpy(float)
            i2 = part[f"mu_E2_cal_h{h}"].to_numpy(float) - part[f"mu_E1_cal_h{h}"].to_numpy(float)
            i3 = part[f"mu_E3_cal_h{h}"].to_numpy(float) - part[f"mu_E1_cal_h{h}"].to_numpy(float)
            rows.append({"variant": variant, "split": split, "horizon": h, "raw_residual_corr_E2_E3": float(np.corrcoef(e2, e3)[0, 1]), "innovation_corr_E2_E3": float(np.corrcoef(i2, i3)[0, 1]), "partial_residual_corr_given_E1": partial_correlation(e2, e3, e1)})
    return pd.DataFrame(rows)


def add_e3_variant(base_wide: pd.DataFrame, frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    predictions, diagnostics = inc.fit_predict_e3(frame, variant)
    diagnostics.to_csv(TABLE_DIR / f"e3_{variant}_diagnostics.csv", index=False, encoding="utf-8-sig")
    out = base_wide.drop(columns=[f"mu_E3_raw_h{h}" for h in HORIZONS]).merge(predictions, on="utc_hour", how="left")
    if out[[f"mu_E3_raw_h{h}" for h in HORIZONS]].isna().any().any():
        raise ValueError(f"E3 {variant} predictions did not cover the confirmation input")
    return out


def write_figures(objective_metrics: pd.DataFrame, loo_ci: pd.DataFrame, interval_table: pd.DataFrame, complement: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    final_overall = objective_metrics[(objective_metrics["split"].eq("final_eval")) & (objective_metrics["group"].eq("overall"))]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    sns.barplot(data=final_overall, x="horizon", y="MAE", hue="model", ax=ax)
    ax.set_title("Fusion objective sensitivity on final evaluation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_fusion_objectives.png", dpi=220)
    plt.close(fig)

    loo = loo_ci[loo_ci["group"].eq("overall")]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    sns.barplot(data=loo, x="horizon", y="delta_MAE_loo_minus_full", hue="removed_expert", ax=ax)
    ax.axhline(0.0, color="black", lw=1)
    ax.set_title("Leave-one-expert-out MAE change")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_leave_one_out.png", dpi=220)
    plt.close(fig)

    intervals = interval_table[interval_table["group"].isin(["overall", "shock", "recovery"])]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), sharey=True)
    for ax, level in zip(axes, [90, 95]):
        part = intervals[intervals["level"].eq(level)]
        sns.lineplot(data=part, x="horizon", y="coverage", hue="group", style="method", markers=True, ax=ax)
        ax.axhline(level / 100, color="black", ls="--", lw=1)
        ax.set_title(f"Nominal {level}%")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_state_intervals.png", dpi=220)
    plt.close(fig)

    comp = complement[complement["split"].eq("final_eval")].melt(id_vars=["variant", "horizon"], value_vars=["raw_residual_corr_E2_E3", "innovation_corr_E2_E3", "partial_residual_corr_given_E1"], var_name="measure", value_name="correlation")
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    sns.lineplot(data=comp, x="horizon", y="correlation", hue="variant", style="measure", markers=True, ax=ax)
    ax.set_title("E2-E3 redundancy under three correlation definitions")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_e2_e3_complementarity.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run confirmation experiments for the six-expert paper")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--skip-e3-refits", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    inc.write_status = lambda stage, message, **extra: write_status("e3_refits", message, source_stage=stage, **extra)
    write_status("loading", "Loading frozen six-expert predictions")
    base_wide = read_input(args.input)

    write_status("fusion_objectives", "Comparing MSE, MAE, Huber and state-balanced Huber objectives", progress="1/4")
    objective_longs: Dict[str, pd.DataFrame] = {}
    objective_metrics = []
    objective_weights = []
    for objective in OBJECTIVES:
        fitted, weights, _ = fit_fusion(base_wide, EXPERTS, objective)
        long_df = long_predictions(fitted, objective)
        objective_longs[objective] = long_df
        objective_weights.append(weights)
        objective_metrics.extend([metric_table(long_df, "screening_dev"), metric_table(long_df, "final_eval")])
    objective_metrics_df = pd.concat(objective_metrics, ignore_index=True)
    objective_selection_df = objective_selection(objective_metrics_df)
    selected_objective = str(objective_selection_df.iloc[0]["objective"])
    primary_objective = "mse"
    objective_metrics_df.to_csv(TABLE_DIR / "fusion_objective_metrics.csv", index=False, encoding="utf-8-sig")
    objective_selection_df.to_csv(TABLE_DIR / "fusion_objective_selection.csv", index=False, encoding="utf-8-sig")
    pd.concat(objective_weights, ignore_index=True).to_csv(TABLE_DIR / "fusion_objective_weights.csv", index=False, encoding="utf-8-sig")

    write_status("leave_one_out", f"Refitting leave-one-expert-out fusion with prespecified {primary_objective}", progress="2/4")
    full_fitted, full_weights, full_calibrators = fit_fusion(base_wide, EXPERTS, primary_objective)
    full_long = long_predictions(full_fitted, "full")
    loo_longs: Dict[str, pd.DataFrame] = {}
    loo_metrics = [metric_table(full_long, "final_eval")]
    loo_weights = [full_weights.assign(model="full")]
    for removed in EXPERTS[1:]:
        remaining = [expert for expert in EXPERTS if expert != removed]
        fitted, weights, _ = fit_fusion(base_wide, remaining, primary_objective)
        model = f"minus_{removed}"
        long_df = long_predictions(fitted, model)
        loo_longs[model] = long_df
        loo_metrics.append(metric_table(long_df, "final_eval"))
        loo_weights.append(weights.assign(model=model))
    loo_metrics_df = pd.concat(loo_metrics, ignore_index=True)
    loo_ci = loo_bootstrap(full_long, loo_longs, args.bootstrap)
    loo_metrics_df.to_csv(TABLE_DIR / "leave_one_expert_out_metrics.csv", index=False, encoding="utf-8-sig")
    loo_ci.to_csv(TABLE_DIR / "leave_one_expert_out_bootstrap.csv", index=False, encoding="utf-8-sig")
    pd.concat(loo_weights, ignore_index=True).to_csv(TABLE_DIR / "leave_one_expert_out_weights.csv", index=False, encoding="utf-8-sig")

    write_status("intervals", "Evaluating global and state-stratified rolling intervals", progress="3/4")
    intervals = pd.concat([rolling_intervals(full_long, "global"), rolling_intervals(full_long, "state_stratified")], ignore_index=True)
    interval_table = interval_metrics(intervals)
    interval_ci = interval_cluster_bootstrap(intervals, args.bootstrap)
    interval_table.to_csv(TABLE_DIR / "state_interval_metrics.csv", index=False, encoding="utf-8-sig")
    interval_ci.to_csv(TABLE_DIR / "state_interval_cluster_bootstrap.csv", index=False, encoding="utf-8-sig")

    write_status("e3_refits", "Evaluating E3 variants and corrected complementarity measures", progress="4/4")
    variant_metrics = []
    complement_rows = []
    variants = ["multi_scale"] if args.skip_e3_refits else ["multi_scale", "memory", "full", "history_only"]
    frame = None if args.skip_e3_refits else inc.load_frame()
    for variant in variants:
        variant_wide = base_wide if variant == "multi_scale" else add_e3_variant(base_wide, frame, variant)
        fitted, _, _ = fit_fusion(variant_wide, EXPERTS, primary_objective)
        long_df = long_predictions(fitted, variant)
        variant_metrics.extend([metric_table(long_df, "screening_dev"), metric_table(long_df, "final_eval")])
        complement_rows.append(complementarity(fitted, variant))
    variant_metrics_df = pd.concat(variant_metrics, ignore_index=True)
    complement_df = pd.concat(complement_rows, ignore_index=True)
    variant_metrics_df.to_csv(TABLE_DIR / "e3_variant_metrics.csv", index=False, encoding="utf-8-sig")
    complement_df.to_csv(TABLE_DIR / "e2_e3_complementarity.csv", index=False, encoding="utf-8-sig")

    write_figures(objective_metrics_df, loo_ci, interval_table, complement_df)
    validation = {
        "run_id": RUN_ID,
        "horizons": HORIZONS,
        "input_rows": len(base_wide),
        "selected_objective_from_screening_dev": selected_objective,
        "primary_objective_for_confirmation": primary_objective,
        "final_eval_used_for_selection": False,
        "single_seed_deep_predictions_reused": True,
        "deep_experts_retrained": False,
        "loo_refits_calibration_and_fusion": True,
        "interval_scores_require_target_time_le_issue_time": True,
        "e3_variants": variants,
        "weights_sum_to_one": bool(np.allclose(full_weights[[f"w_{e}" for e in EXPERTS]].sum(axis=1), 1.0)),
    }
    (TABLE_DIR / "confirmation_validation_checks.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        **validation,
        "status": "completed",
        "objective_selection": objective_selection_df.to_dict(orient="records"),
        "output_tables": sorted(path.name for path in TABLE_DIR.glob("*.csv")),
        "output_figures": sorted(path.name for path in FIG_DIR.glob("*.png")),
    }
    (RUN_DIR / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_status("completed", "Confirmation experiments completed", selected_objective=selected_objective, primary_objective=primary_objective, manifest=str(RUN_DIR / "manifest.json"))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        ensure_dirs()
        (LOG_DIR / "error_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_status("failed", f"{type(exc).__name__}: {exc}", traceback=str(LOG_DIR / "error_traceback.txt"))
        raise
