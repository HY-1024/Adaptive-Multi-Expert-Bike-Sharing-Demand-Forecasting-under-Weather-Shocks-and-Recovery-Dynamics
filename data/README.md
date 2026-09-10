# 数据说明（Data Notes）

`raw/` 保存 Capital Bikeshare 2023–2025 的 36 个原始月度 ZIP，约 629 MB，仅留在本机并由 Git 忽略。

`processed/capital_hourly/` 是最终六专家模型直接读取的小时级输入：

- `capital_hourly_departures_utc.csv`：UTC 小时取车量；
- `capital_open_meteo_historical_weather.csv`：同一 UTC 小时的历史重建天气。

需求数据来自 Capital Bikeshare 官方公开下载；天气来自 Open-Meteo Historical Weather API。逐文件下载记录见 `source_manifest.csv`，天气请求与适用限制见 `weather_source.json`。

重要口径：天气文件是 historical reconstruction，不是当时实际发布的 archived forecast。模型中的 current weather 因而是回溯代理；目标时刻天气只用于事后分组，不作为预测输入。
