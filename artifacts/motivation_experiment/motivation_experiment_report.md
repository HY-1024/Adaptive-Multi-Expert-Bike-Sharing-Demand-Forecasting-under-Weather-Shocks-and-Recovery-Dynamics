# Motivating experiment: current clear weather, different weather paths

本报告只记录动机验证实验，不修改论文正文或最终模型。主分析使用 final_eval 的冻结 E1 原始预测。

## 研究问题

当前无雨时，过去 12--24 小时的降雨路径是否仍与 E1 预测残差有关？如果有关，则说明相同的当前天气并不意味着相同的需求状态。

## 分组与口径

- long-dry：current rain=0 and previous 24h rain sum=0。
- recent-rain：current rain=0; rain in lag 3-12h >= 1.0 mm; lag 1-2h sum=0; previous 24h sum < 5.0 mm。
- recovery：current rain=0; previous 12h rain sum >= 5.0 mm; last positive-rain hour within 3h。
- 只保留当前 rain=0 的时刻；按每个 horizon 的 target time 控制 hour-of-day、weekday、month。
- signed residual = actual - frozen raw E1 prediction；absolute residual = signed residual 的绝对值。
- bootstrap 以 calendar-week 为块，500 次；每次重拟合控制回归。

## 最终评价样本量

| path group | n |
|---|---:|
| long_dry | 3074 |
| recovery | 248 |
| recent_rain | 133 |

## 原始残差均值

| h | outcome | long-dry | recent-rain | recovery |
|---:|---|---:|---:|---:|
| 1 | signed_residual | 50.62 | 22.81 | 34.39 |
| 1 | absolute_residual | 115.35 | 104.95 | 99.50 |
| 2 | signed_residual | 53.51 | 22.87 | 37.27 |
| 2 | absolute_residual | 129.78 | 98.67 | 128.81 |
| 3 | signed_residual | 60.66 | 17.36 | 46.90 |
| 3 | absolute_residual | 151.49 | 117.30 | 143.70 |
| 6 | signed_residual | 72.75 | 7.57 | 72.29 |
| 6 | absolute_residual | 158.12 | 117.41 | 156.05 |

## 控制回归与 bootstrap

| h | outcome | comparison | controlled effect | 95% CI | p(cluster) |
|---:|---|---|---:|---|---:|
| 1 | signed_residual | recent_rain_minus_long_dry | -30.26 | [-66.57, 8.68] | 0.0892 |
| 1 | absolute_residual | recent_rain_minus_long_dry | -11.90 | [-32.83, 16.12] | 0.3126 |
| 1 | signed_residual | recovery_minus_long_dry | -31.28 | [-62.55, 0.68] | 0.0290 |
| 1 | absolute_residual | recovery_minus_long_dry | -23.81 | [-39.31, -7.90] | 0.0026 |
| 2 | signed_residual | recent_rain_minus_long_dry | -20.34 | [-50.25, 16.88] | 0.2306 |
| 2 | absolute_residual | recent_rain_minus_long_dry | -19.11 | [-35.78, 5.24] | 0.0507 |
| 2 | signed_residual | recovery_minus_long_dry | -34.01 | [-62.38, 2.24] | 0.0277 |
| 2 | absolute_residual | recovery_minus_long_dry | -8.15 | [-29.85, 10.86] | 0.4263 |
| 3 | signed_residual | recent_rain_minus_long_dry | -31.22 | [-73.95, 7.77] | 0.1265 |
| 3 | absolute_residual | recent_rain_minus_long_dry | -13.82 | [-36.16, 19.01] | 0.2971 |
| 3 | signed_residual | recovery_minus_long_dry | -35.09 | [-65.95, 6.88] | 0.0566 |
| 3 | absolute_residual | recovery_minus_long_dry | -9.84 | [-33.15, 14.61] | 0.4314 |
| 6 | signed_residual | recent_rain_minus_long_dry | -62.91 | [-92.38, -28.56] | 0.0001 |
| 6 | absolute_residual | recent_rain_minus_long_dry | -23.43 | [-48.08, 1.43] | 0.0378 |
| 6 | signed_residual | recovery_minus_long_dry | -9.58 | [-51.58, 42.93] | 0.6811 |
| 6 | absolute_residual | recovery_minus_long_dry | 7.41 | [-18.50, 47.41] | 0.6581 |

## 解释边界

若 recent-rain 的 signed/absolute residual 控制效应低于 0 且 bootstrap 区间不跨 0，说明近期降雨路径与 E1 误差存在稳定差异，支持把天气历史作为独立建模视角。该结果是预测残差层面的经验关联，不是降雨的因果效应；分组阈值也属于本实验的预先定义操作化方案。

完整逐时样本、分组定义、原始统计和 bootstrap 结果见同目录 CSV/JSON 文件。
