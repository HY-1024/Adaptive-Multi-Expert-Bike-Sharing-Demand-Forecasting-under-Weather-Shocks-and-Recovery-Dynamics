from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "artifacts" / "figure_inputs"
PAPER_DIR = ROOT / "paper"
ASSET_DIR = PAPER_DIR / "assets"
TABLE_DIR = PAPER_DIR / "tables"
HORIZONS = [1, 2, 3, 6]
EXPERTS = ["E1", "E2", "E3", "E4", "E5", "E6"]
EXPERT_LABELS = {
    "E1": "E1 NB-XGBoost",
    "E2": "E2 Weather",
    "E3": "E3 Recovery",
    "E4": "E4 TCN",
    "E5": "E5 GRU",
    "E6": "E6 Transformer",
}
COLORS = {
    "E1": "#355070",
    "E2": "#2A9D8F",
    "E3": "#84A98C",
    "E4": "#E9C46A",
    "E5": "#F4A261",
    "E6": "#E76F51",
}


def ensure_dirs() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(ASSET_DIR / name, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def empirical_quantile(values: list[float], level: float) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0
    adjusted = min(1.0, math.ceil((len(x) + 1) * level) / len(x))
    return float(np.quantile(x, adjusted, method="higher"))


def add_rolling_intervals(pred: pd.DataFrame, window: int = 2160) -> pd.DataFrame:
    """Add issue-time-valid rolling empirical residual intervals.

    The score is |y-mu|/max(sqrt(mu), 25).  At issue time t, only scores whose
    target timestamp is no later than t are admitted to the rolling pool.
    """

    outputs: list[pd.DataFrame] = []
    for h in HORIZONS:
        history = pred[(pred["horizon"].eq(h)) & (pred["split"].eq("screening_dev"))].copy()
        final = pred[(pred["horizon"].eq(h)) & (pred["split"].eq("final_eval"))].copy()
        history = history.sort_values("target_time")
        final = final.sort_values("utc_hour").reset_index(drop=True)

        history["score"] = (
            (history["actual"] - history["prediction"]).abs()
            / np.maximum(np.sqrt(np.maximum(history["prediction"], 0.0)), 25.0)
        )
        final_scores = (
            (final["actual"] - final["prediction"]).abs()
            / np.maximum(np.sqrt(np.maximum(final["prediction"], 0.0)), 25.0)
        ).to_numpy(float)

        candidates = pd.concat(
            [
                history[["target_time", "score"]],
                pd.DataFrame({"target_time": final["target_time"], "score": final_scores}),
            ],
            ignore_index=True,
        ).sort_values("target_time").reset_index(drop=True)

        pool: deque[float] = deque(maxlen=window)
        cursor = 0
        q90: list[float] = []
        q95: list[float] = []
        pool_sizes: list[int] = []
        for issue_time in final["utc_hour"]:
            while cursor < len(candidates) and candidates.at[cursor, "target_time"] <= issue_time:
                pool.append(float(candidates.at[cursor, "score"]))
                cursor += 1
            current = list(pool)
            q90.append(empirical_quantile(current, 0.90))
            q95.append(empirical_quantile(current, 0.95))
            pool_sizes.append(len(current))

        scale = np.maximum(np.sqrt(np.maximum(final["prediction"].to_numpy(float), 0.0)), 25.0)
        final["rolling_pool_size"] = pool_sizes
        for level, qs in [(90, q90), (95, q95)]:
            half = np.asarray(qs) * scale
            final[f"lo_{level}"] = np.maximum(0.0, final["prediction"].to_numpy(float) - half)
            final[f"hi_{level}"] = final["prediction"].to_numpy(float) + half
        outputs.append(final)
    return pd.concat(outputs, ignore_index=True)


def interval_metrics(intervals: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for h in HORIZONS:
        hdf = intervals[intervals["horizon"].eq(h)]
        groups = {
            "overall": np.ones(len(hdf), dtype=bool),
            "normal": hdf["state_for_fusion"].eq("normal").to_numpy(),
            "shock": hdf["state_for_fusion"].eq("shock").to_numpy(),
            "recovery": hdf["state_for_fusion"].eq("recovery").to_numpy(),
        }
        for group, mask in groups.items():
            part = hdf.loc[mask]
            for level in [90, 95]:
                covered = part["actual"].between(part[f"lo_{level}"], part[f"hi_{level}"])
                rows.append(
                    {
                        "horizon": h,
                        "group": group,
                        "level": level,
                        "n_hours": len(part),
                        "coverage": float(covered.mean()),
                        "mean_width": float((part[f"hi_{level}"] - part[f"lo_{level}"]).mean()),
                        "median_pool_size": float(part["rolling_pool_size"].median()),
                    }
                )
    return pd.DataFrame(rows)


def figure_data_overview(frame: pd.DataFrame) -> None:
    data = frame.copy()
    data["month"] = data["utc_hour"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    monthly = data.groupby("month", as_index=False).agg(
        mean_departures=("departures", "mean"),
        rain_hours=("rain", lambda s: int((s > 0).sum())),
        mean_temperature=("temperature_2m", "mean"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 7.4), sharex=True, gridspec_kw={"hspace": 0.18})
    axes[0].plot(monthly["month"], monthly["mean_departures"], color="#355070", marker="o", lw=2)
    axes[0].set_ylabel("Mean hourly\ndepartures")
    axes[1].bar(monthly["month"], monthly["rain_hours"], width=22, color="#2A9D8F", alpha=0.9)
    axes[1].set_ylabel("Rain hours")
    axes[2].plot(monthly["month"], monthly["mean_temperature"], color="#E76F51", marker="o", lw=2)
    axes[2].axhline(0, color="#9CA3AF", lw=0.8)
    axes[2].set_ylabel("Mean temp. (C)")
    axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[2].tick_params(axis="x", rotation=35)
    fig.suptitle("Capital Bikeshare hourly demand and reconstructed weather, 2023-2025", y=0.995, fontsize=13)
    save(fig, "01_data_overview.png")


def figure_incremental_ablation(ablation: pd.DataFrame) -> None:
    order = ["A0_E1", "A1_E1_E2", "A2_E1_E3", "A3_E1_E4", "A4_E1_E5", "A5_E1_E6"]
    labels = {
        "A0_E1": "E1 only",
        "A1_E1_E2": "+ Weather",
        "A2_E1_E3": "+ Recovery",
        "A3_E1_E4": "+ TCN",
        "A4_E1_E5": "+ GRU",
        "A5_E1_E6": "+ Transformer",
    }
    long = ablation.melt(id_vars="combo", value_vars=[f"{h}h_MAE" for h in HORIZONS], var_name="horizon", value_name="MAE")
    long["horizon"] = long["horizon"].str.replace("h_MAE", "", regex=False).astype(int)
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    palette = sns.color_palette("crest", len(order))
    for color, combo in zip(palette, order):
        part = long[long["combo"].eq(combo)].sort_values("horizon")
        ax.plot(part["horizon"], part["MAE"], marker="o", lw=2.2, color=color, label=labels[combo])
    ax.set_xticks(HORIZONS)
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("MAE")
    ax.set_title("Incremental expert ablation on the final evaluation period")
    ax.legend(ncol=3, frameon=True, loc="upper left")
    save(fig, "02_incremental_ablation.png")


def figure_full_state_mae(metrics: pd.DataFrame) -> None:
    part = metrics[metrics["group"].isin(["normal", "shock", "recovery"])].copy()
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    sns.barplot(data=part, x="horizon", y="MAE", hue="group", palette={"normal": "#84A98C", "shock": "#E76F51", "recovery": "#457B9D"}, ax=ax)
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("MAE")
    ax.set_title("Six-expert fusion error by issue-time weather state")
    ax.legend(title="State", ncol=3, loc="upper left")
    save(fig, "03_full_state_mae.png")


def figure_global_weights(weights: pd.DataFrame) -> None:
    global_w = weights[weights["state"].eq("global")].set_index("horizon").sort_index()
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    bottom = np.zeros(len(global_w))
    for expert in EXPERTS:
        vals = global_w[f"w_{expert}"].to_numpy(float)
        ax.bar(global_w.index.astype(str), vals, bottom=bottom, label=EXPERT_LABELS[expert], color=COLORS[expert])
        bottom += vals
    ax.set_ylim(0, 1)
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("Predictive weight")
    ax.set_title("Global weights learned for the six-expert fusion")
    ax.legend(ncol=3, frameon=True, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.subplots_adjust(bottom=0.26)
    save(fig, "04_global_weights.png")


def figure_state_weights(weights: pd.DataFrame) -> None:
    part = weights[weights["state"].isin(["normal", "shock", "recovery"])].copy()
    part["row"] = part.apply(lambda r: f"h={int(r['horizon'])}  {r['state']}", axis=1)
    matrix = part.set_index("row")[[f"w_{e}" for e in EXPERTS]].rename(columns={f"w_{e}": e for e in EXPERTS})
    fig, ax = plt.subplots(figsize=(9.6, 7.2))
    sns.heatmap(matrix, cmap="YlOrRd", vmin=0, vmax=0.75, annot=True, fmt=".2f", linewidths=0.5, cbar_kws={"label": "Predictive weight"}, ax=ax)
    ax.set_xlabel("Expert")
    ax.set_ylabel("Horizon and issue-time state")
    ax.set_title("State-adaptive six-expert weights")
    save(fig, "05_state_weights_heatmap.png")


def figure_residual_correlations(corr: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 9.1), constrained_layout=True)
    for ax, h in zip(axes.flat, HORIZONS):
        sub = corr[corr["horizon"].eq(h)].pivot(index="expert_i", columns="expert_j", values="corr").loc[EXPERTS, EXPERTS]
        sns.heatmap(sub, cmap="vlag", vmin=0.3, vmax=1.0, annot=True, fmt=".2f", square=True, cbar=h == 6, ax=ax)
        ax.set_title(f"h={h}")
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle("Final-period residual correlations among selected expert configurations", fontsize=13)
    save(fig, "06_residual_correlations.png")


def figure_event_window(intervals: pd.DataFrame) -> None:
    h1 = intervals[intervals["horizon"].eq(1)].sort_values("target_time").copy()
    event_sizes = h1[h1["eval_event_cluster_id"].gt(0)].groupby("eval_event_cluster_id").size()
    event_id = int(event_sizes.sort_values(ascending=False).index[0])
    event = h1[h1["eval_event_cluster_id"].eq(event_id)]
    start = event["target_time"].min() - pd.Timedelta(hours=24)
    end = event["target_time"].max() + pd.Timedelta(hours=24)
    case = h1[h1["target_time"].between(start, end)].copy()
    case["target_local"] = case["target_time"].dt.tz_convert("America/New_York").dt.tz_localize(None)

    fig, ax = plt.subplots(figsize=(11.5, 5.1))
    for state, color, label in [("shock", "#E76F51", "rain in progress"), ("recovery", "#457B9D", "post-rain window")]:
        active = case["state_for_fusion"].eq(state)
        starts = active & ~active.shift(fill_value=False)
        ends = active & ~active.shift(-1, fill_value=False)
        for i, (lo, hi) in enumerate(zip(case.loc[starts, "target_local"], case.loc[ends, "target_local"])):
            ax.axvspan(lo, hi + pd.Timedelta(hours=1), color=color, alpha=0.10, label=label if i == 0 else None)
    ax.fill_between(case["target_local"], case["lo_95"], case["hi_95"], color="#F4A261", alpha=0.22, label="rolling 95% interval")
    ax.plot(case["target_local"], case["actual"], color="#1F2937", lw=1.8, label="actual")
    ax.plot(case["target_local"], case["prediction"], color="#D55E00", lw=2.1, label="six-expert prediction")
    ax.set_ylabel("Hourly departures")
    ax.set_xlabel("Target time (America/New_York)")
    ax.set_title("Six-expert forecast around a representative weather event (h=1)")
    ax.legend(ncol=3, frameon=True, loc="upper center")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    save(fig, "07_event_window.png")


def figure_interval_coverage(metrics: pd.DataFrame) -> None:
    part = metrics[metrics["group"].isin(["overall", "shock", "recovery"])].copy()
    part["series"] = part["group"] + " / " + part["level"].astype(str) + "%"
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for ax, level in zip(axes, [90, 95]):
        one = part[part["level"].eq(level)]
        sns.barplot(data=one, x="horizon", y="coverage", hue="group", palette={"overall": "#355070", "shock": "#E76F51", "recovery": "#457B9D"}, ax=ax)
        ax.axhline(level / 100, color="#111827", ls="--", lw=1.1)
        ax.set_ylim(0.65, 1.01)
        ax.set_xlabel("Horizon (hours)")
        ax.set_ylabel("Empirical coverage" if level == 90 else "")
        ax.set_title(f"Nominal {level}%")
        ax.legend(title="State" if level == 95 else None, loc="lower right")
    fig.suptitle("Issue-time-valid rolling empirical interval coverage", fontsize=13)
    save(fig, "08_interval_coverage.png")


def figure_model_comparison(full_metrics: pd.DataFrame, incremental_metrics: pd.DataFrame) -> None:
    full = full_metrics[full_metrics["group"].eq("overall")][["horizon", "MAE"]].assign(model="Full E1-E6")
    e1 = incremental_metrics[(incremental_metrics["combo"].eq("A0_E1")) & (incremental_metrics["group"].eq("overall"))][["horizon", "MAE"]].assign(model="E1 only")
    pair = incremental_metrics[(incremental_metrics["combo"].eq("A3_E1_E4")) & (incremental_metrics["group"].eq("overall"))][["horizon", "MAE"]].assign(model="E1 + TCN")
    plot = pd.concat([e1, pair, full], ignore_index=True)
    fig, ax = plt.subplots(figsize=(10.8, 5.1))
    sns.barplot(data=plot, x="horizon", y="MAE", hue="model", palette={"E1 only": "#94A3B8", "E1 + TCN": "#E9C46A", "Full E1-E6": "#355070"}, ax=ax)
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("MAE")
    ax.set_title("Baseline, strongest pair, and six-expert full model")
    ax.legend(title="Model", loc="upper left")
    save(fig, "09_model_comparison.png")


def figure_monthly_error(intervals: pd.DataFrame) -> None:
    data = intervals.copy()
    data["month"] = data["target_time"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.to_period("M").astype(str)
    data["abs_error"] = (data["actual"] - data["prediction"]).abs()
    monthly = data.groupby(["month", "horizon"], as_index=False)["abs_error"].mean()
    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    sns.lineplot(data=monthly, x="month", y="abs_error", hue="horizon", marker="o", palette="viridis", ax=ax)
    ax.set_xlabel("Target month")
    ax.set_ylabel("MAE")
    ax.set_title("Monthly stability of the six-expert forecast in the final period")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Horizon")
    save(fig, "10_monthly_error.png")


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120})

    pred = pd.read_csv(INPUT_DIR / "fusion_predictions.csv", parse_dates=["utc_hour", "target_time"])
    frame = pd.read_csv(INPUT_DIR / "overview_frame.csv", parse_dates=["utc_hour"])
    full_metrics = pd.read_csv(INPUT_DIR / "full_expert_final_metrics.csv")
    weights = pd.read_csv(INPUT_DIR / "full_expert_weights.csv")
    ablation = pd.read_csv(INPUT_DIR / "incremental_main_ablation_table.csv")
    incremental_metrics = pd.read_csv(INPUT_DIR / "incremental_final_metrics.csv")
    corr = pd.read_csv(INPUT_DIR / "incremental_residual_correlation.csv")

    intervals = add_rolling_intervals(pred)
    int_metrics = interval_metrics(intervals)
    int_metrics.to_csv(TABLE_DIR / "rolling_interval_metrics.csv", index=False, encoding="utf-8-sig")
    full_metrics.to_csv(TABLE_DIR / "full_expert_final_metrics.csv", index=False, encoding="utf-8-sig")
    weights.to_csv(TABLE_DIR / "full_expert_weights.csv", index=False, encoding="utf-8-sig")
    ablation.to_csv(TABLE_DIR / "incremental_ablation.csv", index=False, encoding="utf-8-sig")

    data_summary = {
        "source": "Capital Bikeshare official monthly trip-history ZIPs with Open-Meteo historical weather reconstruction",
        "model_frame_rows": int(len(frame)),
        "first_issue_time_utc": str(frame["utc_hour"].min()),
        "last_issue_time_utc": str(frame["utc_hour"].max()),
        "departures_total": int(frame["departures"].sum()),
        "rain_hours": int(frame["rain"].gt(0).sum()),
        "online_event_count": int(frame["online_event_id"].max()),
        "evaluation_event_cluster_count": int(frame["eval_event_cluster_id"].max()),
        "final_hours_per_horizon": int((pred["split"].eq("final_eval") & pred["horizon"].eq(1)).sum()),
        "final_prediction_rows": int(pred[pred["split"].eq("final_eval")].shape[0]),
        "interval_method": "rolling empirical normalized absolute residual quantiles; only residuals with target_time <= issue_time; 2160-score window",
    }
    (TABLE_DIR / "paper_data_summary.json").write_text(json.dumps(data_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    figure_data_overview(frame)
    figure_incremental_ablation(ablation)
    figure_full_state_mae(full_metrics)
    figure_global_weights(weights)
    figure_state_weights(weights)
    figure_residual_correlations(corr)
    figure_event_window(intervals)
    figure_interval_coverage(int_metrics)
    figure_model_comparison(full_metrics, incremental_metrics)
    figure_monthly_error(intervals)

    print(json.dumps({"paper_dir": str(PAPER_DIR), "figures": 10, "tables": 6, "summary": data_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
