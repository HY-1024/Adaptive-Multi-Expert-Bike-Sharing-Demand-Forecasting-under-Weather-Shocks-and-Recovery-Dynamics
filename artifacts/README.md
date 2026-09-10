# 论文结果工件（Paper Artifacts）

`figure_inputs/` 保存基础论文图表所需的最小数据：完全体预测、数据概览字段、最终指标、融合权重和增量消融。

`confirmation_inputs/` 保存冻结的六专家逐时预测；`confirmation_results/` 保存最新确认实验的表格、图形和来源记录，包括 leave-one-expert-out、融合目标敏感性、全局/状态区间比较与 E3 结构敏感性。论文正文不展示相关性分析。

`motivation_experiment/` 保存当前无雨但天气路径不同的动机验证实验：逐时分组样本、原始 E1 残差、控制 target-time 日历变量后的周块 bootstrap 结果、分组定义和独立 Markdown 报告。该实验目前只作为数据与报告保留，不自动写入主论文。

这些文件从已完成的最终运行中抽取，省略训练检查点、调试日志和重复中间表。基础图可运行 `python src/build_six_expert_paper_assets.py` 重新生成；确认图可运行 `python src/publish_confirmation_results.py runs/confirmation_final2_20260910_111616` 重新发布。
