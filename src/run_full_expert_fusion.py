from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_incremental_expert_screening as inc


RUN_ID = os.environ.get("FULL_EXPERT_RUN_ID", time.strftime("full_expert_%Y%m%d_%H%M%S"))
RUN_DIR = ROOT / "runs" / RUN_ID
TABLE_DIR = RUN_DIR / "tables"
FIG_DIR = RUN_DIR / "figures"
REPORT_DIR = RUN_DIR / "reports"
LOG_DIR = RUN_DIR / "logs"
REPORT_ARCHIVE_DIR = ROOT / "reports" / "six_expert_experiments" / RUN_ID
STATUS_PATH = RUN_DIR / "status.json"
LATEST_STATUS_PATH = ROOT / "reports" / "six_expert_experiments" / "latest_full_expert_status.json"

DEFAULT_CONFIGS = {
    "E2": "spline_only",
    "E3": "multi_scale",
    "E4": "tcn_w72_c32_b3",
    "E5": "gru_w72_h64_l1",
    "E6": "tf_w168_d32_l1_h2",
}


def ensure_dirs() -> None:
    for path in [RUN_DIR, TABLE_DIR, FIG_DIR, REPORT_DIR, LOG_DIR, REPORT_ARCHIVE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_status(stage: str, message: str, **extra: object) -> None:
    payload = {
        "run_id": RUN_ID,
        "run_dir": str(RUN_DIR),
        "stage": stage,
        "message": message,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": inc.DEVICE,
        **extra,
    }
    STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    LATEST_STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_selected_configs() -> Dict[str, str]:
    """Return the screening-period winners frozen before final evaluation."""
    return DEFAULT_CONFIGS.copy()


def find_deep_config(expert: str, config_name: str) -> inc.DeepConfig:
    for cfg in inc.deep_configs()[expert]:
        if cfg.config == config_name:
            return cfg
    raise KeyError(f"Unknown deep config: {expert}/{config_name}")


def save_weight_summary(weights: pd.DataFrame) -> pd.DataFrame:
    weight_cols = [c for c in weights.columns if c.startswith("w_")]
    summary = weights[["horizon", "state", "status", "fit_rows"] + weight_cols].copy()
    summary = summary.sort_values(["horizon", "state"]).reset_index(drop=True)
    summary.to_csv(TABLE_DIR / "full_expert_weights.csv", index=False, encoding="utf-8-sig")
    long = summary.melt(id_vars=["horizon", "state", "status", "fit_rows"], value_vars=weight_cols, var_name="expert", value_name="weight")
    long["expert"] = long["expert"].str.replace("w_", "", regex=False)
    long.to_csv(TABLE_DIR / "full_expert_weights_long.csv", index=False, encoding="utf-8-sig")
    return long


def write_figures(metrics: pd.DataFrame, weight_long: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    overall = metrics[metrics["group"].eq("overall")].copy()
    plt.figure(figsize=(9.5, 4.8))
    sns.barplot(data=overall, x="horizon", y="MAE", color="#4C78A8")
    plt.title("Full E1-E6 model MAE by horizon")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_full_model_mae.png", dpi=220)
    plt.close()

    plot = weight_long[weight_long["state"].ne("global")].copy()
    g = sns.catplot(data=plot, x="state", y="weight", hue="expert", col="horizon", kind="bar", height=3.2, aspect=1.05)
    g.set_axis_labels("issue state", "fusion weight")
    g.figure.suptitle("Full E1-E6 fusion weights by horizon and state", y=1.06)
    g.figure.tight_layout()
    g.figure.savefig(FIG_DIR / "02_full_expert_weights.png", dpi=220)
    plt.close(g.figure)

    global_w = weight_long[weight_long["state"].eq("global")].copy()
    plt.figure(figsize=(9.5, 4.8))
    sns.lineplot(data=global_w, x="horizon", y="weight", hue="expert", marker="o")
    plt.title("Global full-model weights")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_global_full_weights.png", dpi=220)
    plt.close()


def write_report(configs: Dict[str, str], metrics: pd.DataFrame, weights: pd.DataFrame, engineering: pd.DataFrame, validation: Dict[str, object]) -> None:
    overall = metrics[metrics["group"].eq("overall")].copy()
    report = f"""# Full Expert Fusion Report

Run id: `{RUN_ID}`

This run puts all six expert types into one fusion model:

- E1: fixed multi-output NB-XGBoost
- E2: `{configs["E2"]}`
- E3: `{configs["E3"]}`
- E4: `{configs["E4"]}`
- E5: `{configs["E5"]}`
- E6: `{configs["E6"]}`

The run keeps horizons `{inc.HORIZONS}` and the original issue-time information boundary. Expert configurations are inherited from the previous incremental screening run, then all experts are trained again and fused together.

## Validation

```json
{json.dumps(validation, ensure_ascii=False, indent=2)}
```

## Overall Final Metrics

{overall.to_markdown(index=False)}

## State Metrics

{metrics.sort_values(["horizon", "group"]).to_markdown(index=False)}

## Full Expert Weights

{weights.to_markdown(index=False)}

## Engineering Metrics

{engineering.to_markdown(index=False)}

## Outputs

- `tables/full_expert_weights.csv`
- `tables/full_expert_final_metrics.csv`
- `figures/01_full_model_mae.png`
- `figures/02_full_expert_weights.png`
- `figures/03_global_full_weights.png`
"""
    for path in [REPORT_DIR / "full_expert_report.md", REPORT_ARCHIVE_DIR / "full_expert_report.md"]:
        path.write_text(report, encoding="utf-8")


def update_archive_index() -> None:
    index = ROOT / "reports" / "six_expert_experiments" / "README.md"
    line = f"| `{RUN_ID}` | Full E1-E6 fusion using selected expert configs; reports each expert weight. | `{RUN_ID}/full_expert_report.md` |\n"
    text = index.read_text(encoding="utf-8") if index.exists() else "# Three-Expert Experiment Reports\n\n## Runs\n\n| Run id | Main change | Report |\n| --- | --- | --- |\n"
    if RUN_ID not in text:
        index.write_text(text.rstrip() + "\n" + line, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    ensure_dirs()
    inc.write_status = write_status
    configs = load_selected_configs()
    write_status("starting", "Starting full E1-E6 fusion run", configs=configs)

    frame = inc.load_frame()
    frame.to_csv(TABLE_DIR / "full_expert_model_frame.csv", index=False, encoding="utf-8-sig")

    write_status("running", "Training E1 fixed NB-XGBoost", current_model="E1")
    wide, e1_diag = inc.fit_predict_e1(frame)
    diagnostics = [e1_diag]
    raw_parts = [wide]

    write_status("running", f"Training E2 {configs['E2']}", current_model=f"E2/{configs['E2']}")
    e2_pred, e2_diag = inc.fit_predict_e2(frame, configs["E2"])
    diagnostics.append(e2_diag)
    raw_parts.append(e2_pred)

    write_status("running", f"Training E3 {configs['E3']}", current_model=f"E3/{configs['E3']}")
    e3_pred, e3_diag = inc.fit_predict_e3(frame, configs["E3"])
    diagnostics.append(e3_diag)
    raw_parts.append(e3_pred)

    for expert in ["E4", "E5", "E6"]:
        cfg = find_deep_config(expert, configs[expert])
        write_status("running", f"Training {expert} {cfg.config}", current_model=f"{expert}/{cfg.config}")
        pred, diag = inc.fit_predict_deep(frame, cfg, inc.SEED, args.max_epochs, args.patience)
        diagnostics.append(diag)
        raw_parts.append(pred)

    full_wide = raw_parts[0]
    for part in raw_parts[1:]:
        full_wide = full_wide.merge(part, on="utc_hour", how="left")
    full_wide.to_csv(TABLE_DIR / "full_expert_raw_predictions.csv", index=False, encoding="utf-8-sig")

    experts = ["E1", "E2", "E3", "E4", "E5", "E6"]
    write_status("running", "Calibrating and fusing all experts", current_model="FULL_E1_E6")
    fitted, weights, calibrators = inc.apply_combo(full_wide, "FULL_E1_E6", experts)
    long_pred = inc.combo_long(fitted, "FULL_E1_E6", experts)
    metrics = inc.point_metrics(long_pred, "final_eval")
    diagnostics_df = pd.concat(diagnostics, ignore_index=True)
    engineering = diagnostics_df.groupby(["expert", "config"], as_index=False).agg(
        params=("params", "max"),
        train_time_sec=("train_time_sec", "sum"),
        inference_time_sec=("inference_time_sec", "sum"),
        peak_gpu_memory_mb=("peak_gpu_memory_mb", "max"),
        status=("status", lambda x: ",".join(sorted(set(map(str, x))))),
    )
    weight_long = save_weight_summary(weights)

    fitted.to_csv(TABLE_DIR / "full_expert_wide_predictions.csv", index=False, encoding="utf-8-sig")
    long_pred.to_csv(TABLE_DIR / "full_expert_long_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(TABLE_DIR / "full_expert_final_metrics.csv", index=False, encoding="utf-8-sig")
    calibrators.to_csv(TABLE_DIR / "full_expert_calibrators.csv", index=False, encoding="utf-8-sig")
    diagnostics_df.to_csv(TABLE_DIR / "full_expert_diagnostics.csv", index=False, encoding="utf-8-sig")
    engineering.to_csv(TABLE_DIR / "full_expert_engineering_metrics.csv", index=False, encoding="utf-8-sig")

    validation = {
        "run_id": RUN_ID,
        "horizons": inc.HORIZONS,
        "experts": experts,
        "configs": configs,
        "device": inc.DEVICE,
        "amp_enabled": inc.AMP_ENABLED,
        "final_rows": int(long_pred[long_pred["split"].eq("final_eval")].shape[0]),
        "weights_sum_to_one": bool(np.allclose(weights[[f"w_{e}" for e in experts]].sum(axis=1), 1.0)),
        "selection_configs_from": "frozen DEFAULT_CONFIGS selected on screening_dev",
        "source_data_preserved": True,
    }
    (TABLE_DIR / "full_expert_validation_checks.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    write_figures(metrics, weight_long)
    write_report(configs, metrics, weights, engineering, validation)
    manifest = {
        "run_id": RUN_ID,
        "run_dir": str(RUN_DIR),
        "report": str(REPORT_ARCHIVE_DIR / "full_expert_report.md"),
        "weights": str(TABLE_DIR / "full_expert_weights.csv"),
        "metrics": str(TABLE_DIR / "full_expert_final_metrics.csv"),
        "figures": str(FIG_DIR),
    }
    (RUN_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_ARCHIVE_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    update_archive_index()
    write_status("completed", "Full E1-E6 fusion run completed", report=str(REPORT_ARCHIVE_DIR / "full_expert_report.md"), weights=str(TABLE_DIR / "full_expert_weights.csv"))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        ensure_dirs()
        (LOG_DIR / "error_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_status("failed", f"{type(exc).__name__}: {exc}", traceback=str(LOG_DIR / "error_traceback.txt"))
        raise
