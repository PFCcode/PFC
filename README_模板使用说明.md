
# PFC 故障诊断 — 数据模板与映射说明

## 文件列表
- `PFC_Fault_ML_Template.xlsx`：示例数据模板（`data` 工作表为数据，`说明` 为字段解释）。
- `class_mapping.json`：标签映射表（中文 ↔ 内部编码）。
- `model_version.example.json`：模型版本与特征清单样例（请按你的训练配置修改）。

## 使用步骤
1. 打开 `PFC_Fault_ML_Template.xlsx` 的 `data` 工作表：
   - `sample_id`：可留空，系统会自动生成；若填写请保持唯一。
   - `label`：可为空（仅推理），若用于评估请用中文（如“正常/短路/断路”）或你的既有英文标签，系统会按 `class_mapping.json` 做映射。
   - 其他列：请把你的仿真结果列名改成模板中的列名（或反过来修改模板列名与 `model_version.json` 的 `features` 保持一致）。

2. 若你的训练用特征与模板不同：
   - 编辑 `model_version.example.json` 中的 `features` 列表，与训练时**一模一样**（名称与顺序）。
   - 同步修改 Excel 列名，或在系统的“字段映射向导”中手动对齐。

3. 标签映射：
   - 若你的 Excel 用的是自定义标签（中/英文/缩写），把它们加到 `class_mapping.json` 的 `label_mapping` 或 `aliases` 中。
   - 系统会将外部标签统一映射到内部编码（如 `NORMAL`, `SHORT_CIRCUIT`）。

4. 归一化/标准化：
   - 推理务必使用与你训练相同的缩放器（如 `StandardScaler`）。默认从 `models/scaler.joblib` 读取。
   - 若没有，请从训练项目导出；或在首版MVP中允许“无缩放”仅用于演示。

5. 评估：
   - 当 `label` 列不为空时，系统计算总体准确率、Macro/Weighted F1、每类 P/R/F1、混淆矩阵及 PR 曲线；当为空时，仅做预测。

## 温馨提示
- 列名、顺序、缩放器、类别编码四件事务必与训练保持一致。
- 不需要的特征从 Excel 与 `features` 一并删除即可；新增特征也要两边同时增加。
- 首期只做“正常/短路/断路”也完全可以，其他类别可留到后续。
