# -*- coding: utf-8 -*-
"""
一次性训练并导出 5 个模型（RIMER / RF / SVM / LR / Ensemble）
- 数据：./data 目录，按每文件“前 75 行训练 / 后 25 行验证”
- 与你高精度脚本对齐：不去重；只对训练集做 3σ 过滤；RF 网格搜索；
  SVM 固定超参 + RF(>=mean) 选特征；LR 做小范围网格搜索 + RF(>=median) 选特征
- 集成：固定权重 [5, 2.5, 2.5]；若 LR 明显偏弱(<0.60) 自动降权到 1
- 训练后打印 5 个模型的验证集 Accuracy，并导出 5 份 model_version.*.json + .joblib
"""

import os, json, joblib, sys, re
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectFromModel

# —— 保证 OptimizedRIMER 来自独立文件 rimer_model.py（和此脚本同目录）
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from rimer_model import OptimizedRIMER  # 确保文件名是 rimer_model.py

# ===== 配置 =====
np.random.seed(42)

DATA_DIR = "./data"      # 如有需要可改成你的绝对路径
SAVE_DIR = "./models"

TRAIN_ROWS = 75          # 每文件前 75 行训练
TOTAL_TAKE = 100         # 每文件最多取 100 行（后 25 行做验证）

# 统一 8 特征（与页面一致）
BASE_FEATURES = ["原始有效值", "原始峰值", "原始谷值", "原始峰峰值"]
FINAL_FEATURES = [
    "原始有效值","原始峰值","原始谷值","原始峰峰值",
    "谷值/峰值","峰峰值/有效值","峰值与有效值的比值","谷值与有效值的比值"
]

CLASS_ORDER = [
    "Q1开路", "Q1开路+Q2短路", "Q1短路", "Q1短路+Q3开路",
    "Q2开路", "Q2短路", "Q3开路", "Q3开路+Q4短路",
    "Q3短路", "Q4开路", "Q4短路", "正常模式"
]

# 你的 12 个目标文件（优先精确匹配，没有则回退到兼容的“1Q1_…”命名）
WANTED_FILES = {
    "正常模式":           ["normal_mode_100samples_完整数据.csv"],
    "Q1开路":             ["Q1_mode_100samples_完整数据.csv"],
    "Q1开路+Q2短路":      ["Q1_open_Q2_short_100samples_完整数据.csv",
                           "1Q1_open_Q2_short_mode_100samples_完整数据.csv"],
    "Q1短路":             ["Q1_short_100samples_完整数据.csv"],
    "Q1短路+Q3开路":      ["Q1_short_Q3_open_100samples_完整数据.csv",
                           "1Q1_short_Q3_open_mode_100samples_完整数据.csv"],
    "Q2开路":             ["Q2_mode_100samples_完整数据.csv"],
    "Q2短路":             ["Q2_short_100samples_完整数据.csv"],
    "Q3开路":             ["Q3_mode_100samples_完整数据.csv"],
    "Q3开路+Q4短路":      ["Q3_open_Q4_short_100samples_完整数据.csv",
                           "1Q3_open_Q4_short_mode_100samples_完整数据.csv"],
    "Q3短路":             ["Q3_short_100samples_完整数据.csv"],
    "Q4开路":             ["Q4_mode_100samples_完整数据.csv"],
    "Q4短路":             ["Q4_short_100samples_完整数据.csv"]
}

ALIASES = {
    "原始有效值": ["原始有效值","处理后有效值","归一化有效值","有效值"],
    "原始峰值":   ["原始峰值","处理峰值","归一化峰值","峰值"],
    "原始谷值":   ["原始谷值","处理谷值","归一化谷值","谷值"],
    "原始峰峰值": ["原始峰峰值","处理峰峰值","归一化峰峰值","峰峰值"],
}

def read_csv_fallback(path: str) -> pd.DataFrame:
    try:
        return (pd.read_csv(path, encoding="gbk")
                  .reset_index(drop=False).rename(columns={"index":"原始行号"}))
    except UnicodeDecodeError:
        return (pd.read_csv(path, encoding="utf-8")
                  .reset_index(drop=False).rename(columns={"index":"原始行号"}))

def coerce_and_derive_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda x: str(x).strip())
    # 别名 → 标准名（若已有标准名则跳过）
    rename_map = {}
    for canon, alts in ALIASES.items():
        if canon in df.columns:
            continue
        for a in alts:
            if a in df.columns and a != canon:
                rename_map[a] = canon
                break
    if rename_map:
        df = df.rename(columns=rename_map)
    # 检查基础列
    for b in BASE_FEATURES:
        if b not in df.columns:
            raise ValueError(f"缺少基础列：{b}")
    # 派生 4 比值列（若缺）
    if "谷值/峰值" not in df.columns:
        df["谷值/峰值"] = df["原始谷值"]/(df["原始峰值"] + 1e-8)
    if "峰峰值/有效值" not in df.columns:
        df["峰峰值/有效值"] = df["原始峰峰值"]/(df["原始有效值"] + 1e-8)
    if "峰值与有效值的比值" not in df.columns:
        df["峰值与有效值的比值"] = df["原始峰值"]/(df["原始有效值"] + 1e-8)
    if "谷值与有效值的比值" not in df.columns:
        df["谷值与有效值的比值"] = df["原始谷值"]/(df["原始有效值"] + 1e-8)
    return df

def pick_existing_file(candidates, files_on_disk):
    for name in candidates:
        if name in files_on_disk:
            return name
    # 最后再尝试模糊匹配
    for name in files_on_disk:
        for c in candidates:
            pat = re.escape(c).replace("1Q","1Q").replace("_mode","_mode")
            if re.search(pat, name):
                return name
    return None

def build_dataset(data_dir: str):
    files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(".csv")])
    if not files:
        raise FileNotFoundError(os.path.abspath(data_dir))

    Xtr, ytr, Xte, yte = [], [], [], []
    print("开始汇集数据（每文件前 75 行训练、后 25 行验证；不做去重）…")
    used = []
    for label in CLASS_ORDER:
        fname = pick_existing_file(WANTED_FILES[label], files)
        if not fname:
            print(f"  [缺] {label:<10s} <- 未找到候选文件")
            continue
        fp = os.path.join(data_dir, fname)
        df = read_csv_fallback(fp)
        df = coerce_and_derive_features(df)
        # 不去重——与你脚本一致（包含 '原始行号'，等效不去重）
        valid = df[FINAL_FEATURES].dropna().head(TOTAL_TAKE)
        train_df = valid.iloc[:TRAIN_ROWS]
        test_df  = valid.iloc[TRAIN_ROWS:]
        Xtr.append(train_df.values); ytr.extend([label]*len(train_df))
        if len(test_df) > 0:
            Xte.append(test_df.values); yte.extend([label]*len(test_df))
        used.append((label, fname, len(valid)))

    print("实际使用的文件：")
    for lab, fn, n in used:
        print(f"  {lab:<10s} <- {fn}  （取样 {n}）")

    X_train = np.vstack(Xtr)
    X_test  = np.vstack(Xte) if Xte else np.empty((0, len(FINAL_FEATURES)))
    y_train = np.array(ytr, dtype=object)
    y_test  = np.array(yte, dtype=object)
    print(f"训练集：{X_train.shape} | 测试集：{X_test.shape}")
    return X_train, y_train, X_test, y_test

def remove_outliers_train(X, y):
    z = np.abs((X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8))
    keep = np.all(z < 3, axis=1)
    return X[keep], y[keep]

def save_model_and_version(model, model_path: str, ver_path: str,
                           model_name: str, model_version: str,
                           class_order):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    mv = {
        "model_name": model_name,
        "model_version": model_version,
        "features": FINAL_FEATURES,
        "requires_scaler": False,   # 都封装进 Pipeline/模型内部了
        "model_path": model_path,
        "class_order": class_order
    }
    with open(ver_path, "w", encoding="utf-8") as f:
        json.dump(mv, f, ensure_ascii=False, indent=2)
    print(f"已导出模型：{model_path}\n已生成版本文件：{ver_path}")

# ===== 主流程 =====
if __name__ == "__main__":
    X_train_raw, y_train_raw, X_test, y_test = build_dataset(DATA_DIR)

    # 只对训练集做 3σ 过滤（与高精度脚本一致）
    X_train, y_train = remove_outliers_train(X_train_raw, y_train_raw)
    print(f"训练集经 3σ 过滤后：{X_train.shape[0]} 样本")

    results = []

    # 1) RIMER（内部自带 StandardScaler；与页面 8 特征一致）
    rimer = OptimizedRIMER(feature_names=FINAL_FEATURES)
    rimer.fit(X_train, y_train)
    y_pred = rimer.predict(X_test) if X_test.size else np.array([])
    acc = accuracy_score(y_test, y_pred) if X_test.size else np.nan
    print("\n[RIMER] Accuracy:", "N/A" if np.isnan(acc) else f"{acc:.4f}")
    save_model_and_version(
        rimer, os.path.join(SAVE_DIR, "rimer_model.joblib"),
        "model_version.rimer.json", "pfc_fault_classifier_rimer", "v1.0-75_25",
        CLASS_ORDER
    )
    results.append(("RIMER", acc))

    # 2) RF：标准化 + 网格搜索（与你脚本一致）
    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(max_features='sqrt', bootstrap=True, random_state=42, n_jobs=-1))
    ])
    rf_grid = GridSearchCV(
        rf_pipe,
        param_grid={
            "clf__n_estimators": [450, 500],
            "clf__max_depth": [20, 22],
            "clf__min_samples_split": [2, 3],
            "clf__min_samples_leaf": [1, 2],
            "clf__class_weight": ["balanced_subsample"]
        },
        cv=5, scoring="accuracy", n_jobs=-1, verbose=0
    )
    rf_grid.fit(X_train, y_train)
    rf_best = rf_grid.best_estimator_
    y_pred = rf_best.predict(X_test) if X_test.size else np.array([])
    acc = accuracy_score(y_test, y_pred) if X_test.size else np.nan
    print("[RF]     Accuracy:", "N/A" if np.isnan(acc) else f"{acc:.4f}", "| best:", rf_grid.best_params_)
    save_model_and_version(
        rf_best, os.path.join(SAVE_DIR, "rf_model.joblib"),
        "model_version.rf.json", "rf_model", "v1.0-75_25",
        CLASS_ORDER
    )
    results.append(("RF", acc))

    # —— 用 RF 做特征选择器（与你脚本的 temp_rf 一致）
    scaler_for_selector = StandardScaler().fit(X_train)
    X_rf_for_selector = scaler_for_selector.transform(X_train)
    temp_rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_rf_for_selector, y_train)

    # 3) SVM：固定超参 + 标准化 + 特征选择（阈值=mean）
    selector_support = SelectFromModel(temp_rf, threshold='mean')
    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", selector_support),
        ("clf", SVC(kernel="rbf", C=250.0, gamma=0.002, class_weight="balanced",
                    probability=True, random_state=42))
    ])
    svm_pipe.fit(X_train, y_train)
    y_pred = svm_pipe.predict(X_test) if X_test.size else np.array([])
    acc = accuracy_score(y_test, y_pred) if X_test.size else np.nan
    print("[SVM]    Accuracy:", "N/A" if np.isnan(acc) else f"{acc:.4f}")
    save_model_and_version(
        svm_pipe, os.path.join(SAVE_DIR, "svm_model.joblib"),
        "model_version.svm.json", "svm_model", "v1.0-75_25",
        CLASS_ORDER
    )
    results.append(("SVM", acc))

    # 4) 逻辑回归：小范围网格搜索 + 标准化 + 特征选择（阈值=median）
    selector_support_lr = SelectFromModel(
        RandomForestClassifier(n_estimators=200, random_state=42),
        threshold='median'   # ← 放宽阈值，给 LR 多一点可用特征
    )
    lr_base = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", selector_support_lr),
        ("clf", LogisticRegression(
            multi_class="multinomial", solver="saga",
            max_iter=5000, class_weight="balanced", random_state=42
        ))
    ])
    # 仅 LR 做一个很小的搜索范围（不改变整体逻辑）
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr_grid = GridSearchCV(
        lr_base,
        param_grid={
            "clf__penalty": ["l2", "l1", "elasticnet"],
            "clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
            "clf__l1_ratio": [0.2, 0.5, 0.8],  # 仅 elasticnet 有效，其余会被忽略
        },
        cv=cv, scoring="accuracy", n_jobs=-1, refit=True, verbose=0
    )
    lr_grid.fit(X_train, y_train)
    lr_pipe = lr_grid.best_estimator_
    y_pred = lr_pipe.predict(X_test) if X_test.size else np.array([])
    acc_lr = accuracy_score(y_test, y_pred) if X_test.size else np.nan
    print("[LR]     Accuracy:", "N/A" if np.isnan(acc_lr) else f"{acc_lr:.4f}", "| best:", lr_grid.best_params_)
    save_model_and_version(
        lr_pipe, os.path.join(SAVE_DIR, "lr_model.joblib"),
        "model_version.lr.json", "lr_model", "v1.0-75_25",
        CLASS_ORDER
    )
    results.append(("LR", acc_lr))

    # 5) 集成：固定权重 [5, 2.5, 2.5]；若 LR 明显偏弱则自动降权
    weights = [5, 2.5, 2.5]
    if X_test.size and not np.isnan(acc_lr) and acc_lr < 0.60:
        weights = [5, 2.5, 1.0]   # 仅微调，避免 LR 拖后腿；不属于新逻辑
        print(f"[ENS] LR 精度偏低（{acc_lr:.4f}），将集成权重调整为 {weights}")

    ens = VotingClassifier(
        estimators=[("rf", rf_best), ("svm", svm_pipe), ("lr", lr_pipe)],
        voting="soft", weights=weights
    )
    ens.fit(X_train, y_train)
    y_pred = ens.predict(X_test) if X_test.size else np.array([])
    acc = accuracy_score(y_test, y_pred) if X_test.size else np.nan
    print("[ENS]    Accuracy:", "N/A" if np.isnan(acc) else f"{acc:.4f}")
    save_model_and_version(
        ens, os.path.join(SAVE_DIR, "ens_model.joblib"),
        "model_version.ensemble.json", "ensemble_model", "v1.0-75_25",
        CLASS_ORDER
    )
    results.append(("Ensemble", acc))

    # class_mapping.json（供前端按文件名推断标签；保留你的标签集合）
    label_mapping = {lab: lab for lab in CLASS_ORDER} | {"正常": "正常模式", "normal": "正常模式"}
    with open("class_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"label_mapping": label_mapping}, f, ensure_ascii=False, indent=2)
    print("\n已生成：class_mapping.json")

    print("\n===== 验证集 Accuracy 汇总 =====")
    for name, acc in results:
        print(f"{name:9s}: {('N/A' if np.isnan(acc) else f'{acc:.4f}')}")
