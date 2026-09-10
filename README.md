# 六专家共享单车需求预测（Six-Expert Bike Demand Forecasting）

本仓库是课程论文与可复现实验代码的最终版。研究任务保持不变：在预测发布时刻（issue time）仅使用当时可获得的信息，预测未来 1、2、3、6 小时的 Capital Bikeshare 小时级取车量。方法采用“先分工、再融合（divide-and-specialize, then fuse）”的六专家体系，避免把需求惯性、天气冲击、雨后恢复与多尺度时序模式全部塞入单一模型。

## 最终结论

六个专家分别提供独立预测，并通过校准后的非负权重进行状态自适应融合：

| 专家 | 最终配置 | 专门贡献 |
|---|---|---|
| E1 | Multi-output NB-XGBoost | 需求惯性与多时距联合基准 |
| E2 | `spline_only` | 当前天气的非线性响应 |
| E3 | `multi_scale` | 降雨历史与雨后恢复过程 |
| E4 | `tcn_w72_c32_b3` | 72 小时局部卷积模式 |
| E5 | `gru_w72_h64_l1` | 72 小时递归状态记忆 |
| E6 | `tf_w168_d32_l1_h2` | 168 小时长周期注意力模式 |

最终完全体在 2025-04 至 2025-12 的前向评价期取得：

| Horizon | MAE | RMSE |
|---:|---:|---:|
| 1 h | 75.691 | 117.124 |
| 2 h | 97.794 | 156.000 |
| 3 h | 116.644 | 187.735 |
| 6 h | 129.372 | 203.462 |

四时距平均 MAE 为 104.875，比 E1-only 的 153.233 低 31.56%。这里的 “best overall” 仅指本研究已评估配置中的四时距平均 MAE 最低；它不表示每个时距都逐项最优，也不把融合权重解释为因果贡献。

## 目录

```text
.
├─ README.md
├─ requirements.txt
├─ data/
│  ├─ README.md
│  ├─ raw/                         # 本机原始月度 ZIP，Git 忽略
│  └─ processed/capital_hourly/    # 可复现实验的小时级输入
├─ artifacts/figure_inputs/        # 基础论文作图所需的精简结果数据
├─ artifacts/confirmation_inputs/  # 冻结专家预测与确认实验输入
├─ artifacts/confirmation_results/ # LOO、融合目标、状态区间确认结果
├─ src/
│  ├─ capital_data_and_statistics.py
│  ├─ six_expert_statistical_core.py
│  ├─ run_incremental_expert_screening.py
│  ├─ run_full_expert_fusion.py
│  └─ build_six_expert_paper_assets.py
├─ paper/
│  ├─ main.tex                     # LaTeX 论文源码
│  └─ assets/                      # 正文实际引用的最终图
└─ output/pdf/final_paper_six_expert.pdf
```

`.venv/`、`data/raw/`、新产生的 `runs/`、LaTeX 中间文件均由 `.gitignore` 排除。它们可以保留在本机，但不会进入版本库。

## 数据来源与口径

1. **需求数据（trip records）**：Capital Bikeshare 官方公开月度数据，2023-01 至 2025-12，共 36 个 ZIP。原始下载入口为 `https://s3.amazonaws.com/capitalbikeshare-data/index.html`；逐月 URL 和文件大小记录在 [`data/source_manifest.csv`](data/source_manifest.csv)。小时需求定义为按 UTC 取整后的出发记录数（hourly departures）。
2. **天气数据（weather covariates）**：Open-Meteo Historical Weather API，地点为 Washington, D.C.（38.9072, -77.0369），变量包括温度、相对湿度、降水、雨、雪和 10 m 风速。完整请求与限制写在 [`data/weather_source.json`](data/weather_source.json)。
3. **关键限制**：Open-Meteo 数据是历史重建（historical reconstruction），不是预测发布时真实可见的 archived forecast。因此本文把当前天气作为回溯代理变量，不声称完成了严格线上天气预报回测。
4. **时间边界**：所有雨事件状态、需求滞后和恢复变量只由 issue time 及以前的数据更新；目标时刻天气只用于事后分组评价，不进入预测特征。

仓库跟踪的 `data/processed/capital_hourly/` 是从上述原始数据聚合得到的最小小时级输入。保留它是为了让六专家实验在不重新下载约 629 MB 原始 ZIP 的情况下可复现。

## 依赖

- Python 3.11+（本次完整运行使用 Python 3.13.3）
- NumPy、pandas、SciPy
- scikit-learn、statsmodels、XGBoost
- PyTorch（E4–E6）
- Matplotlib、Seaborn（图表）
- XeLaTeX / TeX Live（论文编译）

安装 Python 依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

默认计算设备为 CPU；如本机 XGBoost 支持 CUDA，可在运行前设置 `XGB_DEVICE=cuda`。PyTorch 会自行检测 CUDA，CPU 环境也可以完整运行，但深度专家训练时间更长。

## 复现实验

先进行候选结构的增量消融（incremental ablation），再用冻结的最佳配置训练六专家完全体：

```powershell
python src/run_incremental_expert_screening.py --max-epochs 60 --patience 10
python src/run_full_expert_fusion.py --max-epochs 60 --patience 10
```

运行结果写入 `runs/<run_id>/`，该目录默认不纳入 Git。论文图表可以从已保留的精简结果重新生成：

```powershell
python src/build_six_expert_paper_assets.py
```

最新确认实验的冻结输入和结果已分别保存到 `artifacts/confirmation_inputs/` 与 `artifacts/confirmation_results/`。如需从某次已完成运行重新发布确认结果，可执行：

```powershell
python src/publish_confirmation_results.py runs/confirmation_final2_20260910_111616
```

该确认运行包含 LOO、四种融合目标、全局/状态区间和 E3 结构敏感性；多随机种子不属于本版实验范围。

编译论文：

```powershell
cd paper
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

## 评价协议

- `calibration_fit`: 2024-01-01 至 2024-06-30
- `fusion_fit`: 2024-07-01 至 2024-12-31
- `screening_dev`: 2025-01-01 至 2025-03-31
- `final_eval`: 2025-04-01 至 2025-12-31

候选配置在 `screening_dev` 上冻结；`final_eval` 只用于最终报告。E1+Ek 消融衡量每个专家相对同一 E1 基准的独立预测增益；完全体融合权重表示预测分配（predictive share），不表示因果效应。

## 论文

最终中文论文采用中英文混合术语，完整报告问题重构、六专家分工、18 个候选设计、增量消融、LOO、融合目标敏感性、完全体融合和状态区间校准。LaTeX 源码位于 [`paper/main.tex`](paper/main.tex)，成品位于 [`output/pdf/final_paper_six_expert.pdf`](output/pdf/final_paper_six_expert.pdf)。
