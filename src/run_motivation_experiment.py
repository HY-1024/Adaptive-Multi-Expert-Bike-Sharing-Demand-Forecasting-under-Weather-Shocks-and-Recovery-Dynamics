"""Test weather-path dependence among currently dry issue times.

The experiment is deliberately separate from the paper's model-selection path:
it uses frozen E1 predictions and final-evaluation rows only, and writes data
and a standalone report under artifacts/motivation_experiment/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrix


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "motivation_experiment"
HORIZONS = [1, 2, 3, 6]
N_BOOT = 500
SEED = 20260910


def load_frame() -> pd.DataFrame:
    predictions = pd.read_csv(
        ROOT / "artifacts" / "confirmation_inputs" / "expert_predictions.csv",
        parse_dates=["utc_hour"],
    )
    weather = pd.read_csv(
        ROOT / "data" / "processed" / "capital_hourly" / "capital_open_meteo_historical_weather.csv",
        parse_dates=["utc_hour"],
    )
    frame = predictions.merge(weather[["utc_hour", "rain"]], on="utc_hour", how="left")
    frame = frame.sort_values("utc_hour").reset_index(drop=True)
    if frame["utc_hour"].diff().dropna().ne(pd.Timedelta(hours=1)).any():
        raise ValueError("motivation input is not a complete hourly sequence")

    rain = frame["rain"].fillna(0.0).to_numpy(float)
    for lag in range(1, 25):
        frame[f"rain_lag_{lag}h"] = pd.Series(rain).shift(lag).fillna(0.0)
    lag12 = [f"rain_lag_{lag}h" for lag in range(1, 13)]
    lag24 = [f"rain_lag_{lag}h" for lag in range(1, 25)]
    lag3_12 = [f"rain_lag_{lag}h" for lag in range(3, 13)]
    lag1_2 = [f"rain_lag_{lag}h" for lag in range(1, 3)]
    frame["past_rain_12h"] = frame[lag12].sum(axis=1)
    frame["past_rain_24h"] = frame[lag24].sum(axis=1)
    frame["last_rain_lag_h"] = [
        next((lag for lag in range(1, 25) if i - lag >= 0 and rain[i - lag] > 0), 999)
        if i >= 1
        else 999
        for i in range(len(frame))
    ]
    clear = frame["rain"].eq(0)
    long_dry = clear & frame["past_rain_24h"].eq(0)
    recent_rain = (
        clear
        & frame[lag3_12].sum(axis=1).ge(1.0)
        & frame[lag1_2].sum(axis=1).eq(0)
        & frame["past_rain_24h"].lt(5.0)
    )
    recovery = (
        clear
        & frame["past_rain_12h"].ge(5.0)
        & frame["last_rain_lag_h"].le(3)
    )
    frame["path_group"] = np.select(
        [long_dry, recent_rain, recovery],
        ["long_dry", "recent_rain", "recovery"],
        default="other_clear",
    )
    frame["issue_hour"] = frame["utc_hour"].dt.hour
    frame["issue_weekday"] = frame["utc_hour"].dt.weekday
    frame["issue_month"] = frame["utc_hour"].dt.month
    for horizon in HORIZONS:
        target = pd.to_datetime(frame[f"target_time_h{horizon}"], utc=True)
        frame[f"target_hour_h{horizon}"] = target.dt.hour
        frame[f"target_weekday_h{horizon}"] = target.dt.weekday
        frame[f"target_month_h{horizon}"] = target.dt.month
    frame["week_block"] = frame["utc_hour"].dt.strftime("%Y-%U")
    return frame


def effect_name(model: object, group: str) -> str:
    needle = f"T.{group}"
    names = [name for name in model.params.index if needle in name]
    if len(names) != 1:
        raise ValueError(f"could not locate coefficient for {group}: {list(model.params.index)}")
    return names[0]


def fit_controlled(data: pd.DataFrame, outcome: str, group: str) -> dict[str, float]:
    model = smf.ols(
        f"{outcome} ~ C(path_group, Treatment(reference='long_dry')) + C(hour) + C(weekday) + C(month)",
        data=data,
    ).fit(cov_type="cluster", cov_kwds={"groups": data["week_block"]})
    name = effect_name(model, group)
    return {
        "estimate": float(model.params[name]),
        "cluster_se": float(model.bse[name]),
        "cluster_pvalue": float(model.pvalues[name]),
        "n": int(len(data)),
        "n_blocks": int(data["week_block"].nunique()),
    }


def bootstrap_controlled(
    data: pd.DataFrame, outcome: str, group: str, rng: np.random.Generator
) -> tuple[float, float, int]:
    design = dmatrix(
        "C(path_group, Treatment(reference='long_dry')) + C(hour) + C(weekday) + C(month)",
        data,
        return_type="dataframe",
    )
    coefficient_name = effect_name(
        smf.ols(
            f"{outcome} ~ C(path_group, Treatment(reference='long_dry')) + C(hour) + C(weekday) + C(month)",
            data=data,
        ).fit(),
        group,
    )
    coefficient_index = list(design.columns).index(coefficient_name)
    y = data[outcome].to_numpy(float)
    blocks = data["week_block"].drop_duplicates().to_numpy()
    xtx_by_block = []
    xty_by_block = []
    for block in blocks:
        mask = data["week_block"].eq(block).to_numpy()
        xb = np.asarray(design)[mask]
        yb = y[mask]
        xtx_by_block.append(xb.T @ xb)
        xty_by_block.append(xb.T @ yb)
    xtx_by_block = np.asarray(xtx_by_block)
    xty_by_block = np.asarray(xty_by_block)
    estimates: list[float] = []
    for _ in range(N_BOOT):
        sampled_indices = rng.integers(0, len(blocks), size=len(blocks))
        xtx = xtx_by_block[sampled_indices].sum(axis=0)
        xty = xty_by_block[sampled_indices].sum(axis=0)
        try:
            estimates.append(float(np.linalg.solve(xtx, xty)[coefficient_index]))
        except np.linalg.LinAlgError:
            continue
    if len(estimates) < 100:
        raise RuntimeError(f"too few valid bootstrap replicates: {len(estimates)}")
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
        len(estimates),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    selected = frame[frame["split"].eq("final_eval") & frame["path_group"].isin(["long_dry", "recent_rain", "recovery"])].copy()
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    sample_columns = [
        "utc_hour", "path_group", "split", "rain", "past_rain_12h", "past_rain_24h",
        "last_rain_lag_h", "issue_hour", "issue_weekday", "issue_month", "week_block",
    ]
    for h in HORIZONS:
        analysis = selected.copy()
        analysis["hour"] = analysis[f"target_hour_h{h}"]
        analysis["weekday"] = analysis[f"target_weekday_h{h}"]
        analysis["month"] = analysis[f"target_month_h{h}"]
        analysis[f"signed_residual_h{h}"] = analysis[f"y_h{h}"] - analysis[f"mu_E1_raw_h{h}"]
        analysis[f"absolute_residual_h{h}"] = analysis[f"signed_residual_h{h}"].abs()
        for group in ["recent_rain", "recovery"]:
            for outcome_name, label in [
                (f"signed_residual_h{h}", "signed_residual"),
                (f"absolute_residual_h{h}", "absolute_residual"),
            ]:
                result = fit_controlled(analysis, outcome_name, group)
                low, high, valid = bootstrap_controlled(analysis, outcome_name, group, rng)
                rows.append({
                    "horizon": h,
                    "outcome": label,
                    "comparison": f"{group}_minus_long_dry",
                    **result,
                    "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high,
                    "bootstrap_valid_replicates": valid,
                })
        for group, part in analysis.groupby("path_group", sort=True):
            for outcome_name, label in [
                (f"signed_residual_h{h}", "signed_residual"),
                (f"absolute_residual_h{h}", "absolute_residual"),
            ]:
                raw_rows.append({
                    "horizon": h,
                    "path_group": group,
                    "outcome": label,
                    "n": len(part),
                    "mean": float(part[outcome_name].mean()),
                    "median": float(part[outcome_name].median()),
                    "std": float(part[outcome_name].std(ddof=1)),
                })

    residual_columns = [f"signed_residual_h{h}" for h in HORIZONS] + [f"absolute_residual_h{h}" for h in HORIZONS]
    for h in HORIZONS:
        selected[f"signed_residual_h{h}"] = selected[f"y_h{h}"] - selected[f"mu_E1_raw_h{h}"]
        selected[f"absolute_residual_h{h}"] = selected[f"signed_residual_h{h}"].abs()
        sample_columns.extend([f"target_hour_h{h}", f"target_weekday_h{h}", f"target_month_h{h}"])
    selected[sample_columns + residual_columns].to_csv(OUT / "motivation_sample_final.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(raw_rows).to_csv(OUT / "raw_group_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(OUT / "controlled_effects_bootstrap.csv", index=False, encoding="utf-8-sig")

    definitions = {
        "primary_split": "final_eval",
        "reference_group": "long_dry",
        "long_dry": "current rain=0 and previous 24h rain sum=0",
        "recent_rain": "current rain=0; rain in lag 3-12h >= 1.0 mm; lag 1-2h sum=0; previous 24h sum < 5.0 mm",
        "recovery": "current rain=0; previous 12h rain sum >= 5.0 mm; last positive-rain hour within 3h",
        "control_variables": "target_time hour-of-day, target_time weekday, target_time month (horizon-specific)",
        "residual_definition": "actual Y[t+h] minus frozen raw E1 prediction mu_E1_raw_h",
        "bootstrap": "500 resamples of calendar-week blocks; OLS with hour/weekday/month controls refit per replicate",
        "rain_source": "capital_open_meteo_historical_weather.csv",
        "prediction_source": "artifacts/confirmation_inputs/expert_predictions.csv",
        "random_seed": SEED,
        "sample_counts_final_eval": selected["path_group"].value_counts().to_dict(),
    }
    (OUT / "group_definitions.json").write_text(json.dumps(definitions, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "status": "completed",
        "n_rows_final_eval_selected": int(len(selected)),
        "horizons": HORIZONS,
        "outputs": sorted(path.name for path in OUT.iterdir()),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(definitions, pd.DataFrame(raw_rows), pd.DataFrame(rows))


def write_report(definitions: dict, raw: pd.DataFrame, effects: pd.DataFrame) -> None:
    lines = [
        "# Motivating experiment: current clear weather, different weather paths",
        "",
        "本报告只记录动机验证实验，不修改论文正文或最终模型。主分析使用 final_eval 的冻结 E1 原始预测。",
        "",
        "## 研究问题",
        "",
        "当前无雨时，过去 12--24 小时的降雨路径是否仍与 E1 预测残差有关？如果有关，则说明相同的当前天气并不意味着相同的需求状态。",
        "",
        "## 分组与口径",
        "",
        f"- long-dry：{definitions['long_dry']}。",
        f"- recent-rain：{definitions['recent_rain']}。",
        f"- recovery：{definitions['recovery']}。",
        "- 只保留当前 rain=0 的时刻；按每个 horizon 的 target time 控制 hour-of-day、weekday、month。",
        "- signed residual = actual - frozen raw E1 prediction；absolute residual = signed residual 的绝对值。",
        "- bootstrap 以 calendar-week 为块，500 次；每次重拟合控制回归。",
        "",
        "## 最终评价样本量",
        "",
        "| path group | n |",
        "|---|---:|",
    ]
    for group, n in definitions["sample_counts_final_eval"].items():
        lines.append(f"| {group} | {n} |")
    lines += ["", "## 原始残差均值", "", "| h | outcome | long-dry | recent-rain | recovery |", "|---:|---|---:|---:|---:|"]
    for h in HORIZONS:
        for outcome in ["signed_residual", "absolute_residual"]:
            sub = raw[(raw.horizon == h) & (raw.outcome == outcome)].set_index("path_group")
            lines.append(f"| {h} | {outcome} | {sub.loc['long_dry','mean']:.2f} | {sub.loc['recent_rain','mean']:.2f} | {sub.loc['recovery','mean']:.2f} |")
    lines += ["", "## 控制回归与 bootstrap", "", "| h | outcome | comparison | controlled effect | 95% CI | p(cluster) |", "|---:|---|---|---:|---|---:|"]
    for row in effects.itertuples(index=False):
        lines.append(f"| {row.horizon} | {row.outcome} | {row.comparison} | {row.estimate:.2f} | [{row.bootstrap_ci95_low:.2f}, {row.bootstrap_ci95_high:.2f}] | {row.cluster_pvalue:.4f} |")
    lines += [
        "", "## 解释边界", "",
        "若 recent-rain 的 signed/absolute residual 控制效应低于 0 且 bootstrap 区间不跨 0，说明近期降雨路径与 E1 误差存在稳定差异，支持把天气历史作为独立建模视角。该结果是预测残差层面的经验关联，不是降雨的因果效应；分组阈值也属于本实验的预先定义操作化方案。",
        "",
        "完整逐时样本、分组定义、原始统计和 bootstrap 结果见同目录 CSV/JSON 文件。",
    ]
    (OUT / "motivation_experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
