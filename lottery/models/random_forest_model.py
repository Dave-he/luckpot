"""
随机森林彩票预测模型

思路: 对每个红球位置/蓝球位置训练一个随机森林分类器
- 多棵决策树投票，避免过拟合
- 能捕获非线性特征关系
"""
import os
import pickle
import numpy as np
from typing import List, Dict, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class RandomForestPredictor:
    """基于随机森林的彩票号码预测器"""

    def __init__(self, config: Dict):
        self.config = config
        self.red_min, self.red_max = config["red_range"]
        self.blue_min, self.blue_max = config["blue_range"]
        self.red_count = config["red_count"]
        self.blue_count = config["blue_count"]
        self.red_total = self.red_max - self.red_min + 1
        self.blue_total = (self.blue_max - self.blue_min + 1) if self.blue_count > 0 and self.blue_max >= self.blue_min else 0

        self.window_size = 15
        self.red_models = []
        self.blue_models = []
        self.scaler = StandardScaler()
        self.is_trained = False

    def _build_features(self, history: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X, y_red, y_blue = [], [], []

        for i in range(self.window_size, len(history)):
            features = []
            # 历史window_size期号码
            for j in range(i - self.window_size, i):
                for n in history[j]["reds"]:
                    features.append(n)
                features.extend([0] * (self.red_count - len(history[j]["reds"])))
                blues = history[j]["blues"]
                features.extend(blues)
                features.extend([0] * (self.blue_count - len(blues)))

            # 频率特征 (最近10期)
            recent_10 = history[max(0, i - 10):i]
            red_freq = np.zeros(self.red_total)
            for rec in recent_10:
                for n in rec["reds"]:
                    if self.red_min <= n <= self.red_max:
                        red_freq[n - self.red_min] += 1
            features.extend(red_freq.tolist())

            # 和值/跨度
            recent_5 = history[max(0, i - 5):i]
            for rec in recent_5:
                red_sum = sum(rec["reds"])
                red_span = max(rec["reds"]) - min(rec["reds"])
                features.extend([red_sum, red_span])

            X.append(features)
            # 标签: 每个红球位置的号码类别索引
            y_red_pos = [r - self.red_min for r in history[i]["reds"]]
            y_red_pos = [max(0, min(self.red_total - 1, p)) for p in y_red_pos]
            while len(y_red_pos) < self.red_count:
                y_red_pos.append(0)
            y_red.append(y_red_pos[:self.red_count])

            if self.blue_count > 0:
                blues_sorted = sorted(history[i]["blues"])
                y_blue_pos = [b - self.blue_min for b in blues_sorted]
                while len(y_blue_pos) < self.blue_count:
                    y_blue_pos.append(0)
                y_blue.append(y_blue_pos[:self.blue_count])
            else:
                y_blue.append([])

        return np.array(X), np.array(y_red), np.array(y_blue)

    def train(self, history: List[Dict]) -> Dict:
        if len(history) < self.window_size + 10:
            return {"success": False, "error": "数据量不足"}

        X, y_red, y_blue = self._build_features(history)
        if len(X) == 0:
            return {"success": False, "error": "无有效训练数据"}

        X = self.scaler.fit_transform(X)

        # 训练红球模型
        self.red_models = []
        for pos in range(self.red_count):
            model = RandomForestClassifier(
                n_estimators=80, max_depth=8, random_state=42 + pos,
                n_jobs=-1, class_weight="balanced",
            )
            model.fit(X, y_red[:, pos])
            self.red_models.append(model)

        # 训练蓝球模型
        self.blue_models = []
        if self.blue_count > 0 and y_blue.shape[1] > 0:
            for pos in range(self.blue_count):
                model = RandomForestClassifier(
                    n_estimators=80, max_depth=8, random_state=100 + pos,
                    n_jobs=-1, class_weight="balanced",
                )
                model.fit(X, y_blue[:, pos])
                self.blue_models.append(model)

        self.is_trained = True
        # 评估
        train_acc = self._evaluate(X, y_red, y_blue)
        return {"success": True, "samples": len(X), "metrics": train_acc}

    def _evaluate(self, X, y_red, y_blue) -> Dict:
        red_acc = []
        for pos, model in enumerate(self.red_models):
            pred = model.predict(X)
            acc = float(np.mean(pred == y_red[:, pos]))
            red_acc.append(acc)
        metrics = {"red_acc_mean": round(float(np.mean(red_acc)), 4)}

        if self.blue_count > 0 and len(self.blue_models) > 0:
            blue_acc = []
            for pos, model in enumerate(self.blue_models):
                pred = model.predict(X)
                acc = float(np.mean(pred == y_blue[:, pos]))
                blue_acc.append(acc)
            metrics["blue_acc_mean"] = round(float(np.mean(blue_acc)), 4)
        return metrics

    def predict(self, history: List[Dict]) -> Tuple[List[int], List[int], Dict]:
        if not self.is_trained or len(history) < self.window_size:
            return [], [], {}

        recent = history[-self.window_size:]
        features = []
        for rec in recent:
            for n in rec["reds"]:
                features.append(n)
            features.extend([0] * (self.red_count - len(rec["reds"])))
            blues = rec["blues"]
            features.extend(blues)
            features.extend([0] * (self.blue_count - len(blues)))

        recent_10 = history[-10:]
        red_freq = np.zeros(self.red_total)
        for rec in recent_10:
            for n in rec["reds"]:
                if self.red_min <= n <= self.red_max:
                    red_freq[n - self.red_min] += 1
        features.extend(red_freq.tolist())

        recent_5 = history[-5:]
        for rec in recent_5:
            red_sum = sum(rec["reds"])
            red_span = max(rec["reds"]) - min(rec["reds"])
            features.extend([red_sum, red_span])

        X = self.scaler.transform([features])

        # 预测红球概率
        red_probs_all = []
        for model in self.red_models:
            probs = model.predict_proba(X)[0]
            # 补齐类别
            full_probs = np.zeros(self.red_total)
            for cls, p in zip(model.classes_, probs):
                if 0 <= cls < self.red_total:
                    full_probs[cls] = p
            red_probs_all.append(full_probs)

        is_repeatable = (self.red_max - self.red_min + 1) <= 10 and self.red_count >= 3

        if is_repeatable:
            reds = []
            for probs in red_probs_all:
                idx = int(np.argmax(probs))
                reds.append(idx + self.red_min)
            avg_probs = np.mean(red_probs_all, axis=0)
        else:
            avg_probs = np.mean(red_probs_all, axis=0)
            top_indices = np.argsort(avg_probs)[-self.red_count:][::-1]
            reds = sorted([int(idx) + self.red_min for idx in top_indices])

        # 预测蓝球
        blues = []
        if self.blue_count > 0 and len(self.blue_models) > 0:
            blue_probs_all = []
            for model in self.blue_models:
                probs = model.predict_proba(X)[0]
                full_probs = np.zeros(self.blue_total)
                for cls, p in zip(model.classes_, probs):
                    if 0 <= cls < self.blue_total:
                        full_probs[cls] = p
                blue_probs_all.append(full_probs)
            avg_blue = np.mean(blue_probs_all, axis=0)
            top_blue = np.argsort(avg_blue)[-self.blue_count:][::-1]
            blues = sorted([int(idx) + self.blue_min for idx in top_blue])

        info = {
            "red_top_probs": [(int(idx) + self.red_min, round(float(avg_probs[idx]), 4))
                              for idx in np.argsort(avg_probs)[-10:][::-1]],
        }
        return reds, blues, info

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "rf_model.pkl"), "wb") as f:
            pickle.dump({
                "red_models": self.red_models,
                "blue_models": self.blue_models,
                "scaler": self.scaler,
                "is_trained": self.is_trained,
                "window_size": self.window_size,
            }, f)

    def load(self, model_dir: str) -> bool:
        path = os.path.join(model_dir, "rf_model.pkl")
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.red_models = data["red_models"]
        self.blue_models = data["blue_models"]
        self.scaler = data["scaler"]
        self.is_trained = data["is_trained"]
        self.window_size = data["window_size"]
        return True
