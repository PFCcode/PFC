# train_cap_monitor.py
# -*- coding: utf-8 -*-
"""
电容状态监测：小样本 + 7维时域/派生特征 + 多输出随机森林 + 网格搜索
- 从文件名解析真值: C(μF), R(Ω/mΩ)
- 特征: [v_pp, i_pp, v_rms, i_rms, max_dvdt, esr_raw, c_raw]
- 兼容老版本库：RMSE=√MSE（不使用 mean_squared_error(..., squared=False)）
- 导出: models/cap_monitor.joblib + model_version.cap_monitor.json
"""

import os, re, glob, json, joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ========= 路径配置（按需修改） =========
DATA_DIR   = r'./data_caps'                       # 你的 *_data.csv 放这里
SAVE_MODEL = r'./models/cap_monitor.joblib'       # 模型导出路径
SAVE_JSON  = r'./model_version.cap_monitor.json'  # 版本文件导出路径


# ========= 工具函数 =========
def parse_labels_from_filename(fname: str):
    """
    从文件名提取 C (μF) & ESR R (Ω)，支持 μF/uF/F 与 mΩ/Ω、科学计数法
    例: ...C=470uF_R=80mΩ_data.csv / ...C=4.7e-4F_R=0.08Ω_...
    """
    name = os.path.basename(fname)
    cap_m = re.search(r'C=([0-9\.]+(?:e[-+]?\d+)?)(μF|uF|F)', name, flags=re.I)
    res_m = re.search(r'R=([0-9\.]+(?:e[-+]?\d+)?)(mΩ|Ω)', name, flags=re.I)
    if not cap_m or not res_m:
        raise ValueError(f"无法解析标签: {name}")
    # C -> μF
    val_c, unit_c = cap_m.groups()
    val_c = float(val_c)
    C_uF = val_c if unit_c.lower() != 'f' else val_c * 1e6
    # R -> Ω
    val_r, unit_r = res_m.groups()
    val_r = float(val_r)
    R_ohm = val_r / 1e3 if unit_r.lower().startswith('m') else val_r
    return C_uF, R_ohm


def read_csv_any(fp):
    """兼容本地路径或streamlit上传对象；自动尝试 gbk/utf-8。"""
    try:
        return pd.read_csv(fp, encoding="gbk")
    except Exception:
        try:
            if hasattr(fp, "seek"):
                fp.seek(0)
        except Exception:
            pass
        return pd.read_csv(fp, encoding="utf-8")


def estimate_raw_params(t, v, i):
    """
    粗估:
      ESR_raw = ΔV / ΔI
      C_raw   = ∫I dt / ΔV
    兼容老 numpy：使用 np.trapz
    """
    eps = 1e-12
    dv = float(v[0] - v[-1])
    di = float((i[0] - i[-1]) + eps)
    esr_raw = dv / di
    q = float(np.trapz(i, t))
    c_raw = q / (dv + eps)
    return esr_raw, c_raw


def extract_features(fp):
    """
    读取 CSV: 假定前三列分别为 time(s), V, I（如不同请改列索引）
    返回特征顺序必须与版本文件一致:
      ["v_pp","i_pp","v_rms","i_rms","max_dvdt","esr_raw","c_raw"]
    """
    df = read_csv_any(fp)
    t = df.iloc[:, 0].to_numpy()
    v = df.iloc[:, 1].to_numpy()
    i = df.iloc[:, 2].to_numpy()

    v_pp   = float(np.ptp(v))
    i_pp   = float(np.ptp(i))
    v_rms  = float(np.sqrt(np.mean(v ** 2)))
    i_rms  = float(np.sqrt(np.mean(i ** 2)))
    dvdt   = np.gradient(v, t)
    max_dvdt = float(np.max(np.abs(dvdt)))
    esr_raw, c_raw = estimate_raw_params(t, v, i)
    return [v_pp, i_pp, v_rms, i_rms, max_dvdt, esr_raw, c_raw]


def load_xy(files):
    """批量加载文件 → 特征 X (n×7) 与标签 y (n×2: [C_uF, R_ohm])"""
    X, y = [], []
    for fp in files:
        C_uF, R_ohm = parse_labels_from_filename(fp)
        X.append(extract_features(fp))
        y.append([C_uF, R_ohm])
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


# ========= 主流程 =========
def main():
    # 1) 扫描数据与划分
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, '*_data.csv')))
    if len(all_files) < 3:
        raise RuntimeError("请确保 DATA_DIR 下至少有 3 个 *_data.csv（训练+验证）")
    train_files = all_files[:-2]
    val_files   = all_files[-2:]

    # 2) 读入数据
    X_train, y_train = load_xy(train_files)
    X_val,   y_val   = load_xy(val_files)

    # 3) 内部验证拆分
    X_tr, X_te, y_tr, y_te = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # 4) Pipeline：StandardScaler + MultiOutput(RandomForest)
    base_rf = MultiOutputRegressor(RandomForestRegressor(random_state=42, n_jobs=-1))
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", base_rf)
    ])

    # 5) 网格搜索
    param_grid = {
        "reg__estimator__n_estimators": [200, 500],
        "reg__estimator__max_depth": [10, None],
        "reg__estimator__min_samples_leaf": [1, 2, 5],
    }
    grid = GridSearchCV(pipe, param_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=1)
    grid.fit(X_tr, y_tr)
    best = grid.best_estimator_
    print("\n最佳参数：", grid.best_params_)

    # 6) 内部测试集评估（RMSE=√MSE，兼容老版本 sklearn）
    y_hat = best.predict(X_te)
    print("\n—— 内部测试集评估 ——")
    for k, name in enumerate(["C(μF)", "R(Ω)"]):
        mae = mean_absolute_error(y_te[:, k], y_hat[:, k])
        mse = mean_squared_error(y_te[:, k], y_hat[:, k])  # 无 squared 参数
        rmse = float(np.sqrt(mse))
        print(f" {name:8s}  MAE={mae:.4g}, RMSE={rmse:.4g}")

    # 7) 留出验证集评估
    y_val_hat = best.predict(X_val)
    cmp = pd.DataFrame({
        "file": [os.path.basename(f) for f in val_files],
        "C_true(uF)": y_val[:, 0],
        "C_pred(uF)": y_val_hat[:, 0],
        "C_err(%)": np.abs(y_val_hat[:, 0] - y_val[:, 0]) / np.maximum(y_val[:, 0], 1e-12) * 100,
        "R_true(ohm)": y_val[:, 1],
        "R_pred(ohm)": y_val_hat[:, 1],
        "R_err(%)": np.abs(y_val_hat[:, 1] - y_val[:, 1]) / np.maximum(y_val[:, 1], 1e-12) * 100,
    })
    print("\n—— 验证集预测对比 ——")
    print(cmp.to_string(index=False))

    # 8) 导出模型与版本文件
    os.makedirs(os.path.dirname(SAVE_MODEL), exist_ok=True)
    joblib.dump(best, SAVE_MODEL)
    print(f"\n最优模型已保存到: {SAVE_MODEL}")

    version = {
        "model_name": "cap_monitor_rf",
        "model_version": "v1.0",
        "task": "regression_multioutput",
        "model_path": SAVE_MODEL,   # 可用相对路径
        "features": ["v_pp", "i_pp", "v_rms", "i_rms", "max_dvdt", "esr_raw", "c_raw"],
        "targets": ["capacitance_uF", "esr_ohm"],
        "requires_scaler": False,   # 标准化已在 Pipeline 内
        "notes": "小样本+7特征+MultiOutput RF；从文件名解析C/R。"
    }
    with open(SAVE_JSON, "w", encoding="utf-8") as f:
        json.dump(version, f, ensure_ascii=False, indent=2)
    print(f"版本文件已保存到: {SAVE_JSON}")


if __name__ == "__main__":
    main()
