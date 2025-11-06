# rimer_model.py
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

class OptimizedRIMER:
    """
    规则+证据融合的优化版 RIMER。
    重要说明：将类放在独立模块，保证 joblib 在不同进程/脚本中都能定位到类的完全限定路径。
    """
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.rule_base = {}     # {class: {'intervals': [...], 'confidence': float, 'feature_std': [...]}}
        self.classes_ = None
        self.scaler = StandardScaler()  # 训练/预测统一标准化

    # --- 内部度量 ---
    def _calc_dynamic_confidence(self, class_samples):
        feature_stds = np.std(class_samples, axis=0)
        feature_ranges = np.max(class_samples, axis=0) - np.min(class_samples, axis=0)
        avg_std_score = 1 - min(np.mean(feature_stds) / 1.8, 1.0)
        avg_range_score = min(np.mean(feature_ranges) / 4.5, 1.0)
        return round(0.6 + 0.35 * (avg_std_score * 0.7 + avg_range_score * 0.3), 2)

    def _calc_feature_intervals(self, class_samples):
        intervals, key_features = [], [4, 7]
        for i in range(class_samples.shape[1]):
            if i in key_features:
                lower = np.percentile(class_samples[:, i], 15)
                upper = np.percentile(class_samples[:, i], 85)
            else:
                lower = np.percentile(class_samples[:, i], 10)
                upper = np.percentile(class_samples[:, i], 90)
            intervals.append([lower, upper])
        return intervals

    # --- 训练/预测 ---
    def fit(self, X_train, y_train):
        Xs = self.scaler.fit_transform(X_train)
        self.classes_ = np.unique(y_train)
        for cls in tqdm(self.classes_, desc="RIMER 构建规则库"):
            cs = Xs[y_train == cls]
            self.rule_base[cls] = {
                'intervals': self._calc_feature_intervals(cs),
                'confidence': self._calc_dynamic_confidence(cs),
                'feature_std': np.std(cs, axis=0)
            }
        return self

    def _matching(self, x, rule):
        scores, intervals = [], rule['intervals']
        stds = rule['feature_std'] + 1e-8
        key_features = [4, 6, 7]
        coeffs = [1.2 if i in key_features else 0.5 for i in range(len(x))]
        for i, val in enumerate(x):
            lo, hi = intervals[i]
            if lo <= val <= hi:
                sc = 1.1 if i in key_features else 1.0
            else:
                dev = val - hi if val > hi else lo - val
                sc = max(0.0, 1.0 - (dev / stds[i]) * coeffs[i])
            scores.append(sc)
        return np.mean(scores) * rule['confidence']

    def _fuse(self, evidences: dict):
        tot = sum(evidences.values())
        if tot == 0:
            n = len(evidences)
            return {k: 1.0 / n for k in evidences}
        bb = {k: v / tot for k, v in evidences.items()}
        bb = {k: (v if v >= 0.1 else 0.0) for k, v in bb.items()}
        s = sum(bb.values())
        if s == 0:
            n = len(bb)
            bb = {k: 1.0 / n for k in bb}
            s = 1.0
        bb = {k: v / s for k, v in bb.items()}
        # 简化冲突处理：高置信度保持
        return bb

    def predict(self, X_test):
        Xs = self.scaler.transform(X_test)
        preds = []
        for x in Xs:
            ev = {cls: self._matching(x, self.rule_base[cls]) for cls in self.classes_}
            fused = self._fuse(ev)
            preds.append(max(fused.items(), key=lambda kv: kv[1])[0])
        return np.array(preds)

    # 可选：单样本手工预测
    def manual_predict(self, raw4):
        import numpy as np
        x = np.array(raw4, dtype=float).reshape(1, 4)
        f1 = x[:, 2:3] / (x[:, 1:2] + 1e-8)
        f2 = x[:, 3:4] / (x[:, 0:1] + 1e-8)
        f3 = x[:, 1:2] / (x[:, 0:1] + 1e-8)
        f4 = x[:, 2:3] / (x[:, 0:1] + 1e-8)
        X8 = np.hstack([x, f1, f2, f3, f4])
        Xs = self.scaler.transform(X8)
        ev = {cls: self._matching(Xs[0], self.rule_base[cls]) for cls in self.classes_}
        fused = self._fuse(ev)
        pred = max(fused.items(), key=lambda kv: kv[1])[0]
        return pred, dict(sorted(fused.items(), key=lambda kv: kv[1], reverse=True))
