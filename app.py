# -*- coding: utf-8 -*-
"""
PFC 故障诊断 & 电容状态监测 —— 统一前端
"""

import os, re, json, joblib
import numpy as np
import pandas as pd
import streamlit as st

# 可视化
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             precision_recall_curve, average_precision_score)

# ========== 全局中文字体 ==========
from matplotlib import font_manager
def _use_chinese_font(prefer: str|None=None) -> str|None:
    cands = [prefer] if prefer else []
    cands += ["Microsoft YaHei","SimHei","PingFang SC","Hiragino Sans GB",
              "Heiti SC","WenQuanYi Micro Hei","Noto Sans CJK SC","Source Han Sans CN"]
    owned = {f.name for f in font_manager.fontManager.ttflist}
    chosen = None
    for n in cands:
        if n and n in owned:
            chosen = n; break
    if chosen:
        matplotlib.rcParams["font.family"] = [chosen]
        matplotlib.rcParams["font.sans-serif"] = [chosen]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return chosen
_CN_FONT = _use_chinese_font()

# ========== rerun 兼容 ==========
def _safe_rerun():
    if hasattr(st, "rerun"): st.rerun()
    elif hasattr(st, "experimental_rerun"): st.experimental_rerun()

# ========== 登录 ==========
def check_login(user: str, pwd: str) -> bool:
    try:
        auth = st.secrets.get("auth", {})
        u = auth.get("username", "admin")
        p = auth.get("password", "pfc@123")
    except Exception:
        u, p = "admin", "pfc@123"
    return (user == u) and (pwd == p)

def login_block() -> bool:
    if st.session_state.get("authed", False):
        return True
    st.title("🔐 登录系统")
    with st.form("login"):
        user = st.text_input("用户名", "")
        pwd  = st.text_input("密码", "", type="password")
        ok   = st.form_submit_button("登录")
    if ok:
        if check_login(user, pwd):
            st.session_state["authed"] = True
            st.success("登录成功")
            _safe_rerun()
        else:
            st.error("用户名或密码错误")
    return False

# ========== 故障诊断常量 ==========
FINAL_FEATURES = [
    "原始有效值","原始峰值","原始谷值","原始峰峰值",
    "谷值/峰值","峰峰值/有效值","峰值与有效值的比值","谷值与有效值的比值"
]
BASE_FEATURES = ["原始有效值","原始峰值","原始谷值","原始峰峰值"]

DEFAULT_VERSION_FILE = {
    "RIMER":    "model_version.rimer.json",
    "RF":       "model_version.rf.json",
    "SVM":      "model_version.svm.json",
    "LR":       "model_version.lr.json",
    "Ensemble": "model_version.ensemble.json",
}

FILENAME_RULES = [
    {"pattern": r"normal",                "label": "正常模式"},
    {"pattern": r"Q1[_-]?mode",           "label": "Q1开路"},
    {"pattern": r"Q1[_-]?open.*Q2[_-]?short", "label": "Q1开路+Q2短路"},
    {"pattern": r"Q1[_-]?short(?!.*Q3[_-]?open)", "label": "Q1短路"},
    {"pattern": r"Q1[_-]?short.*Q3[_-]?open",     "label": "Q1短路+Q3开路"},
    {"pattern": r"Q2[_-]?mode",           "label": "Q2开路"},
    {"pattern": r"Q2[_-]?short",          "label": "Q2短路"},
    {"pattern": r"Q3[_-]?mode",           "label": "Q3开路"},
    {"pattern": r"Q3[_-]?open.*Q4[_-]?short",     "label": "Q3开路+Q4短路"},
    {"pattern": r"Q3[_-]?short(?!.*Q4[_-]?open)", "label": "Q3短路"},
    {"pattern": r"Q4[_-]?mode",           "label": "Q4开路"},
    {"pattern": r"Q4[_-]?short",          "label": "Q4短路"},
]

ALIASES = {
    "原始有效值": ["原始有效值","处理后有效值","归一化有效值","有效值"],
    "原始峰值":   ["原始峰值","处理峰值","归一化峰值","峰值"],
    "原始谷值":   ["原始谷值","处理谷值","归一化谷值","谷值"],
    "原始峰峰值": ["原始峰峰值","处理峰峰值","归一化峰峰值","峰峰值"],
}

# ========== 工具函数 ==========
def read_csv_fallback(file):
    try:
        return pd.read_csv(file, encoding="gbk")
    except Exception:
        try:
            if hasattr(file, "seek"): file.seek(0)
        except Exception: pass
        return pd.read_csv(file, encoding="utf-8")

def _collapse_duplicates_keep_first(df: pd.DataFrame, name: str) -> pd.DataFrame:
    cols = [c for c in df.columns if c == name]
    if len(cols) > 1:
        df = df.drop(columns=cols[1:])
    return df

def coerce_and_derive_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda x: str(x).strip())
    # 别名 → 标准名
    rename_map = {}
    for canon, alts in ALIASES.items():
        if canon in df.columns: continue
        for a in alts:
            if a in df.columns and a != canon:
                rename_map[a] = canon; break
    if rename_map:
        df = df.rename(columns=rename_map)
    # 确保基础列
    for b in BASE_FEATURES:
        df = _collapse_duplicates_keep_first(df, b)
        if b not in df.columns:
            raise ValueError(f"缺少基础列：{b}")
    # 派生比值
    if "谷值/峰值" not in df.columns:
        df["谷值/峰值"] = df["原始谷值"]/(df["原始峰值"]+1e-8)
    if "峰峰值/有效值" not in df.columns:
        df["峰峰值/有效值"] = df["原始峰峰值"]/(df["原始有效值"]+1e-8)
    if "峰值与有效值的比值" not in df.columns:
        df["峰值与有效值的比值"] = df["原始峰值"]/(df["原始有效值"]+1e-8)
    if "谷值与有效值的比值" not in df.columns:
        df["谷值与有效值的比值"] = df["原始谷值"]/(df["原始有效值"]+1e-8)
    # 去重复列
    for n in FINAL_FEATURES:
        df = _collapse_duplicates_keep_first(df, n)
    return df

def infer_label_from_name(name: str, rules: list) -> str | None:
    for r in rules:
        if re.search(r["pattern"], name, flags=re.I):
            return r["label"]
    return None

def build_eval_set(uploaded_files, drop_dup: bool, sigma_mode: str, filename_rules):
    TRAIN_ROWS, TOTAL_TAKE = 75, 100
    TEST_ROWS = TOTAL_TAKE - TRAIN_ROWS
    X_list, y_list, used_labels = [], [], []
    per_class_buckets = {}

    for f in uploaded_files:
        df = read_csv_fallback(f).dropna()
        df = coerce_and_derive_features(df)
        if drop_dup:
            df = df.drop_duplicates(subset=FINAL_FEATURES)

        valid = df[FINAL_FEATURES].dropna().head(TOTAL_TAKE)
        if len(valid) < TEST_ROWS:
            continue
        test_df = valid.iloc[TRAIN_ROWS:TOTAL_TAKE]
        label = infer_label_from_name(f.name, filename_rules)
        if label is None:
            st.warning(f"无法从文件名推断标签，已跳过：{f.name}")
            continue

        X_list.append(test_df.values)
        y_list.extend([label]*len(test_df))
        used_labels.append(label)
        if sigma_mode == "per_class":
            per_class_buckets.setdefault(label, []).append(test_df.values)

    if not X_list:
        return np.empty((0, len(FINAL_FEATURES))), np.array([], dtype=object), []

    X_test = np.vstack(X_list)
    y_test = np.array(y_list, dtype=object)

    if sigma_mode == "global" and len(X_test) > 0:
        z = np.abs((X_test - X_test.mean(axis=0)) / (X_test.std(axis=0) + 1e-8))
        keep = np.all(z < 3, axis=1)
        X_test, y_test = X_test[keep], y_test[keep]
        st.info(f"[全局3σ] 过滤后：测试样本 {len(y_test)}")
    elif sigma_mode == "per_class":
        kept_X, kept_y = [], []
        for lab, mats in per_class_buckets.items():
            Xi = np.vstack(mats)
            z = np.abs((Xi - Xi.mean(axis=0)) / (Xi.std(axis=0) + 1e-8))
            keep = np.all(z < 3, axis=1)
            kept_X.append(Xi[keep]); kept_y.extend([lab]*np.sum(keep))
        if kept_X:
            X_test = np.vstack(kept_X); y_test = np.array(kept_y, dtype=object)
            st.info(f"[按类3σ] 过滤后：测试样本 {len(y_test)}")

    classes = sorted(list(set(used_labels)))
    return X_test, y_test, classes

def load_version_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_model(version_path: str):
    info = load_version_json(version_path)
    mpath = info.get("model_path")
    if not os.path.isabs(mpath):
        base = os.path.dirname(os.path.abspath(__file__))
        mpath = os.path.join(base, mpath)
    try:
        if "rimer" in os.path.basename(mpath).lower():
            from rimer_model import OptimizedRIMER  # noqa
    except Exception:
        pass
    model = joblib.load(mpath)
    classes = info.get("class_order", [])
    features = info.get("features", FINAL_FEATURES)
    return model, classes, features, info

def _get_pred_and_scores(model, X: np.ndarray, all_classes: list[str]):
    y_pred = model.predict(X)
    model_classes = getattr(model, "classes_", None)
    y_score = None
    if hasattr(model, "predict_proba"):
        try:
            y_score = model.predict_proba(X)
        except Exception:
            y_score = None
    if y_score is None and hasattr(model, "decision_function"):
        try:
            df = model.decision_function(X)
            if df.ndim == 1:
                df = np.vstack([-df, df]).T
            df = df - df.max(axis=1, keepdims=True)
            exp = np.exp(df); y_score = exp/(exp.sum(axis=1, keepdims=True)+1e-8)
        except Exception:
            y_score = None
    if y_score is None:
        base_classes = list(model_classes) if model_classes is not None else all_classes
        y_score = np.zeros((len(y_pred), len(base_classes)))
        idx = {c:i for i,c in enumerate(base_classes)}
        for i,c in enumerate(y_pred):
            if c in idx: y_score[i, idx[c]] = 1.0
        model_classes = base_classes

    if model_classes is None:
        model_classes = all_classes
    model_classes = list(model_classes)
    aligned = np.zeros((len(y_pred), len(all_classes)))
    pos = {c:i for i,c in enumerate(model_classes)}
    for j,c in enumerate(all_classes):
        if c in pos: aligned[:,j] = y_score[:,pos[c]]
    return y_pred, aligned

def _smooth(arr, win: int = 11):
    n = len(arr)
    if n < 5: return arr
    win = max(5, min(win, n if n%2==1 else n-1))
    k = np.ones(win)/win
    pad = np.r_[arr[0], arr, arr[-1]]
    sm = np.convolve(pad, k, mode="same")[1:-1]
    return sm

def _plot_pr_curves(y_true, y_score, classes):
    y_bin = label_binarize(y_true, classes=classes)
    p_micro, r_micro, _ = precision_recall_curve(y_bin.ravel(), y_score.ravel())
    ap_micro = average_precision_score(y_bin, y_score, average="micro")
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=140)
    ax.plot(_smooth(r_micro), _smooth(p_micro), lw=2.2, label=f"micro-avg AP={ap_micro:.3f}")
    for i,c in enumerate(classes):
        p,r,_ = precision_recall_curve(y_bin[:,i], y_score[:,i])
        ax.plot(_smooth(r), _smooth(p), lw=1.2, alpha=.85, label=str(c))
    ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
    ax.set_xlabel("召回率"); ax.set_ylabel("精确率"); ax.set_title("PR 曲线（平滑）")
    ax.legend(ncol=2, fontsize=8); ax.grid(alpha=.3, linestyle="--"); plt.tight_layout()
    return fig

def eval_and_plot(model, X_test: np.ndarray, y_test: np.ndarray, all_classes: list[str]):
    if X_test.size == 0:
        st.warning("测试集为空，无法评估。"); return
    y_pred, y_score = _get_pred_and_scores(model, X_test, all_classes)
    acc = accuracy_score(y_test, y_pred)
    st.subheader(f"✅ Accuracy：{acc:.4f}")

    cm = confusion_matrix(y_test, y_pred, labels=all_classes)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=all_classes, yticklabels=all_classes, ax=ax)
    ax.set_xlabel("预测类别"); ax.set_ylabel("实际类别")
    ax.set_title(f"混淆矩阵（Accuracy = {acc:.4f}）")
    ax.set_xticklabels(all_classes, rotation=45, ha='right')
    ax.set_yticklabels(all_classes, rotation=0)
    plt.tight_layout(); st.pyplot(fig)

    rep = classification_report(y_test, y_pred, labels=all_classes, output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(rep).T
    st.subheader("分类报告"); st.dataframe(rep_df.style.format(precision=4), use_container_width=True)

    appeared = sorted(list({c for c in y_test if c in all_classes}))
    if len(appeared) >= 2:
        col_idx = [all_classes.index(c) for c in appeared]
        st.subheader("PR 曲线（平滑）")
        st.pyplot(_plot_pr_curves(y_test, y_score[:, col_idx], appeared))
    else:
        st.info("PR 曲线：评估集中出现的有效类别不足 2 个，暂不绘制。")

# ========== 电容监测 ==========
CAP_VERSION_FILE_DEFAULT = "model_version.cap_monitor.json"

def _read_csv_any(fp):
    try:
        return pd.read_csv(fp, encoding="gbk")
    except Exception:
        try:
            if hasattr(fp,"seek"): fp.seek(0)
        except Exception: pass
        return pd.read_csv(fp, encoding="utf-8")

def _cap_parse_labels_from_name(name: str):
    cap_m = re.search(r'C=([0-9\.]+(?:e[-+]?\d+)?)(μF|uF|F)', name, flags=re.I)
    res_m = re.search(r'R=([0-9\.]+(?:e[-+]?\d+)?)(mΩ|Ω)', name, flags=re.I)
    C_uF = R_ohm = None
    if cap_m:
        v,u = cap_m.groups(); v = float(v)
        C_uF = v if u.lower()!='f' else v*1e6
    if res_m:
        v,u = res_m.groups(); v=float(v)
        R_ohm = v/1e3 if u.lower().startswith('m') else v
    return C_uF, R_ohm

def _cap_estimate_raw_params(t, v, i):
    eps = 1e-12
    dv = v[0]-v[-1]; di = (i[0]-i[-1])+eps
    esr_raw = dv/di
    q = float(np.trapz(i, t))
    c_raw = q/(dv+eps)
    return esr_raw, c_raw

def cap_extract_features_from_csv(file_like):
    df = _read_csv_any(file_like)
    t = df.iloc[:,0].to_numpy(); v = df.iloc[:,1].to_numpy(); i = df.iloc[:,2].to_numpy()
    v_pp = float(np.ptp(v)); i_pp=float(np.ptp(i))
    v_rms = float(np.sqrt(np.mean(v**2))); i_rms = float(np.sqrt(np.mean(i**2)))
    dvdt = np.gradient(v, t); max_dvdt = float(np.max(np.abs(dvdt)))
    esr_raw, c_raw = _cap_estimate_raw_params(t, v, i)
    return [v_pp, i_pp, v_rms, i_rms, max_dvdt, esr_raw, c_raw]

def _store_cap_model_to_state(model, features, info, version_file):
    st.session_state["cap_mdl"] = model
    st.session_state["cap_features"] = features
    st.session_state["cap_info"] = info
    st.session_state["cap_version_file"] = version_file

def _get_cap_model_from_state():
    return (st.session_state.get("cap_mdl"),
            st.session_state.get("cap_features", []),
            st.session_state.get("cap_info", {}),
            st.session_state.get("cap_version_file"))

def cap_monitor_page():
    st.header("电容状态监测")

    st.subheader("① 加载电容监测模型")
    ver_input = st.text_input("model_version.cap_monitor.json 路径（留空用默认）",
                              value=CAP_VERSION_FILE_DEFAULT)
    ver_path = ver_input if os.path.isabs(ver_input) else os.path.join(os.path.dirname(os.path.abspath(__file__)), ver_input)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("加载/刷新 电容模型", type="primary"):
            if not os.path.exists(ver_path):
                st.error(f"版本文件不存在：{ver_path}")
            else:
                try:
                    info = load_version_json(ver_path)
                    mpath = info.get("model_path","")
                    if not os.path.isabs(mpath):
                        mpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), mpath)
                    model = joblib.load(mpath)
                    feats = info.get("features", [])
                    _store_cap_model_to_state(model, feats, info, ver_path)
                    st.success(f"模型已加载：{os.path.basename(ver_path)}")
                except Exception as e:
                    st.exception(e)
    with c2:
        if st.button("清除电容模型"):
            for k in ["cap_mdl","cap_features","cap_info","cap_version_file"]:
                st.session_state.pop(k, None)
            st.info("已清除电容模型")

    st.subheader("② 上传电容数据 CSV（可多选）")
    files = st.file_uploader("每个CSV需包含前三列：time(s), V, I；其余列忽略。",
                             type=["csv"], accept_multiple_files=True)

    if st.button("开始监测"):
        model, feats, info, vfile = _get_cap_model_from_state()
        if model is None:
            st.error("尚未加载电容模型"); st.stop()
        if not files:
            st.warning("请先上传 CSV 文件"); st.stop()

        rows = []
        for f in files:
            try:
                X = np.array([cap_extract_features_from_csv(f)], dtype=float)
                yhat = model.predict(X)[0]  # [C_pred, R_pred]
                name = f.name
                C_true, R_true = _cap_parse_labels_from_name(name)
                row = {"file": name,
                       "C_pred(uF)": float(yhat[0]), "R_pred(ohm)": float(yhat[1]),
                       "C_true(uF)": C_true, "R_true(ohm)": R_true}
                if C_true is not None:
                    row["C_err(%)"] = abs(yhat[0]-C_true)/max(C_true,1e-12)*100
                if R_true is not None:
                    row["R_err(%)"] = abs(yhat[1]-R_true)/max(R_true,1e-12)*100
                rows.append(row)
            except Exception as e:
                rows.append({"file": f.name, "error": str(e)})

        df_out = pd.DataFrame(rows)
        st.dataframe(df_out, use_container_width=True)

        if "C_err(%)" in df_out or "R_err(%)" in df_out:
            st.subheader("③ 误差汇总（仅当文件名含真值时）")
            if "C_err(%)" in df_out and df_out["C_err(%)"].notna().any():
                st.write("C(uF) 平均相对误差：", f"{df_out['C_err(%)'].dropna().mean():.2f}%")
            if "R_err(%)" in df_out and df_out["R_err(%)"].notna().any():
                st.write("R(Ω) 平均相对误差：", f"{df_out['R_err(%)'].dropna().mean():.2f}%")

        csv_bytes = df_out.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("⬇️ 下载监测结果 CSV", data=csv_bytes,
                           file_name="cap_monitor_results.csv", mime="text/csv")

# ========== 故障诊断 ==========
def diagnosis_page():
    st.title("PFC 故障诊断 — 多算法批量推理与评估")

    # 侧边栏：算法选择 & 版本文件
    st.sidebar.header("算法 / 模型")
    algo = st.sidebar.selectbox("选择算法 / 模型", list(DEFAULT_VERSION_FILE.keys()), index=0)
    custom_ver = st.sidebar.text_input("或自定义 model_version.json 路径（留空使用上方选择）", "")

    # 类别映射规则（可被 class_mapping.json 覆盖）
    filename_rules = FILENAME_RULES
    if os.path.exists("class_mapping.json"):
        try:
            with open("class_mapping.json", "r", encoding="utf-8") as f:
                cm = json.load(f)
            filename_rules = cm.get("filename_rules", filename_rules)
        except Exception:
            pass

    # —— 数据预处理选项：默认收起 —— #
    with st.expander("数据预处理选项", expanded=False):
        drop_dup = st.checkbox("单文件样本去重（按 8 个特征 drop_duplicates）", value=False)
        sigma_mode = st.radio("3σ 异常值过滤", ["关闭","全局 3σ","按类别 3σ"], index=0)
        sigma_mode = {"关闭":"off","全局 3σ":"global","按类别 3σ":"per_class"}[sigma_mode]
        mode = st.radio("运行模式", ["评估（需要标签）","仅预测（不需要标签）"], index=0)

    # —— ① 加载模型 —— #
    st.subheader("① 加载模型")
    version_file = custom_ver.strip() if custom_ver.strip() else DEFAULT_VERSION_FILE[algo]
    if not os.path.isabs(version_file):
        version_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), version_file)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("加载/刷新模型", type="primary"):
            if not os.path.exists(version_file):
                st.error(f"版本文件不存在：{version_file}")
            else:
                try:
                    model, classes_in_model, features_in_model, info_json = load_model(version_file)
                    st.session_state["dx_model"] = model
                    st.session_state["dx_classes"] = classes_in_model
                    st.session_state["dx_features"] = features_in_model
                    st.session_state["dx_info"] = info_json
                    st.success(f"模型已加载： {os.path.basename(version_file)}")
                except Exception as e:
                    st.exception(e)
    with c2:
        if st.button("清除已加载模型"):
            for k in ["dx_model","dx_classes","dx_features","dx_info"]:
                st.session_state.pop(k, None)
            st.session_state["manual_expanded"] = False
            st.info("已清除模型")

    # —— ② 上传 CSV —— #
    st.subheader("② 上传 CSV（可多选）")
    files = st.file_uploader("CSV 列需包含四个基础量（或其别名）；系统会派生比值列。",
                             type=["csv"], accept_multiple_files=True)

    # —— 评估 —— #
    if mode == "评估（需要标签）":
        if st.button("开始评估", type="secondary"):
            model = st.session_state.get("dx_model")
            classes_in_model = st.session_state.get("dx_classes", [])
            if model is None:
                try:
                    model, classes_in_model, features_in_model, info_json = load_model(version_file)
                    st.session_state["dx_model"] = model
                    st.session_state["dx_classes"] = classes_in_model
                    st.session_state["dx_features"] = features_in_model
                    st.session_state["dx_info"] = info_json
                    st.info("已根据选择自动加载模型。")
                except Exception as e:
                    st.error("未能加载模型，请先点击“加载模型”。"); st.stop()
            if not files:
                st.warning("请先上传 CSV 文件。"); st.stop()

            X_test, y_test, classes_used = build_eval_set(files, drop_dup, sigma_mode, filename_rules)
            all_classes = classes_in_model if classes_in_model else sorted(list(set(classes_used)))
            st.write(f"测试样本数：{len(y_test)}；类别数：{len(all_classes)}")
            if len(y_test) == 0:
                st.warning("没有可评估样本，请检查文件名映射是否正确。")
            else:
                eval_and_plot(model, X_test, y_test, all_classes)

    # —— 仅预测（固定后25行）—— #
    else:
        if st.button("开始预测", type="secondary"):
            model = st.session_state.get("dx_model")
            classes_in_model = st.session_state.get("dx_classes", [])
            if model is None:
                try:
                    model, classes_in_model, features_in_model, info_json = load_model(version_file)
                    st.session_state["dx_model"] = model
                    st.session_state["dx_classes"] = classes_in_model
                    st.session_state["dx_features"] = features_in_model
                    st.session_state["dx_info"] = info_json
                    st.info("已根据选择自动加载模型。")
                except Exception as e:
                    st.error("未能加载模型，请先点击“加载模型”。"); st.stop()
            if not files:
                st.warning("请先上传 CSV 文件。"); st.stop()

            TRAIN_ROWS, TOTAL_TAKE = 75, 100
            rows = []
            for f in files:
                df = read_csv_fallback(f).dropna()
                df = coerce_and_derive_features(df)
                if drop_dup: df = df.drop_duplicates(subset=FINAL_FEATURES)
                valid = df[FINAL_FEATURES].dropna().head(TOTAL_TAKE)
                part = valid.iloc[TRAIN_ROWS:TOTAL_TAKE]  # 固定后25行
                if len(part)==0:
                    rows.append({"file": f.name, "error":"有效行数不足"}); continue
                X = part.values
                if sigma_mode == "global":
                    z = np.abs((X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8))
                    keep = np.all(z < 3, axis=1)
                    X = X[keep]
                y_pred = model.predict(X)
                if hasattr(model,"predict_proba"):
                    prob = model.predict_proba(X)
                    avg = np.mean(prob, axis=0)
                    top_idx = int(np.argmax(avg))
                    base_classes = getattr(model,"classes_",None) or classes_in_model
                    top_label = base_classes[top_idx] if base_classes else str(np.unique(y_pred)[0])
                    conf = float(np.max(avg))
                else:
                    vals, cnts = np.unique(y_pred, return_counts=True)
                    top_label = str(vals[int(np.argmax(cnts))]); conf = np.nan
                rows.append({"file": f.name, "samples": len(X),
                             "majority_pred": top_label, "avg_conf": conf})
            out = pd.DataFrame(rows)
            st.dataframe(out, use_container_width=True)

            csv_bytes = out.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button("⬇️ 下载预测结果 CSV", data=csv_bytes,
                               file_name="diagnosis_predictions.csv", mime="text/csv")

    # —— ③ 手动输入单样本预测（二选一显示） —— #
    st.subheader("③ 手动输入单样本预测")
    with st.expander("手动输入并预测", expanded=st.session_state.get("manual_expanded", False)):
        model = st.session_state.get("dx_model")
        if model is None:
            st.info("请先在上方“① 加载模型”中加载模型。")
        else:
            with st.form("manual_predict_form"):
                input_mode = st.radio(
                    "输入方式",
                    ["只填 4 个基础量（自动派生 4 个比值）", "直接填 8 个最终特征"],
                    index=0
                )

                if input_mode == "只填 4 个基础量（自动派生 4 个比值）":
                    v_rms = st.number_input("原始有效值", value=0.0, format="%.6f")
                    v_pk  = st.number_input("原始峰值",   value=0.0, format="%.6f")
                    v_val = st.number_input("原始谷值",   value=0.0, format="%.6f")
                    v_pp  = st.number_input("原始峰峰值", value=0.0, format="%.6f")
                    submitted = st.form_submit_button("预测")

                    if submitted:
                        feats = {
                            "原始有效值": v_rms, "原始峰值": v_pk,
                            "原始谷值": v_val, "原始峰峰值": v_pp,
                            "谷值/峰值": v_val/(v_pk+1e-8),
                            "峰峰值/有效值": v_pp/(v_rms+1e-8),
                            "峰值与有效值的比值": v_pk/(v_rms+1e-8),
                            "谷值与有效值的比值": v_val/(v_rms+1e-8),
                        }
                        X = np.array([[feats[n] for n in FINAL_FEATURES]], dtype=float)
                        y_pred = model.predict(X)[0]
                        st.success(f"预测类别：{y_pred}")

                else:  # 直接填 8 个最终特征
                    feat_vals = {}
                    for n in FINAL_FEATURES:
                        feat_vals[n] = st.number_input(n, value=0.0, format="%.6f", key=f"mf_{n}")
                    submitted = st.form_submit_button("预测")

                    if submitted:
                        X = np.array([[feat_vals[n] for n in FINAL_FEATURES]], dtype=float)
                        y_pred = model.predict(X)[0]
                        st.success(f"预测类别：{y_pred}")

# ========== 主程序 ==========
def main():
    if not login_block():
        return

    st.sidebar.header("功能模块")
    module = st.sidebar.radio("选择模块", ["开关管故障诊断","电容状态监测"], index=0)

    if module == "电容状态监测":
        cap_monitor_page()
    else:
        diagnosis_page()

    st.caption("提示：模型由 `train_export_ml_models_improved.py` 与 `train_cap_monitor.py` 导出，"
               "版本文件（model_version.*.json）记录了模型路径、特征顺序与类别/目标名。")

if __name__ == "__main__":
    main()
