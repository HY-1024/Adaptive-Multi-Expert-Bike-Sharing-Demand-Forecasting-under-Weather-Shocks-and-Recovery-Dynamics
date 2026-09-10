# 论文结果工件（Paper Artifacts）

`figure_inputs/` 保存重建论文图表所需的最小数据：完全体预测、数据概览字段、最终指标、融合权重、增量消融和专家残差相关性。

这些文件从已完成的最终运行中抽取，省略训练检查点、调试日志和重复中间表。运行 `python src/build_six_expert_paper_assets.py` 可重新生成 `paper/assets/` 与 `paper/tables/`。
