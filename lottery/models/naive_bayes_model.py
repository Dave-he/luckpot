"""
朴素贝叶斯彩票预测模型

思路: 基于贝叶斯定理,假设号码间条件独立
- P(号码n | 历史特征) ∝ P(历史特征 | 号码n) × P(号码n)
- 特征: 最近N期是否出现该号码、和值范围、奇偶比等
"""
import os
import pickle
import numpy as np
from typing import List, Dict, Tuple
from sklearn.naive_bayes import MultinomialNB


class NaiveBayesPredictor:
    """基于朴素贝叶斯的彩票号码预测器"""

    def __init__(self, config: Dict):
        self.config = config
        self.red_min, self.red_max = config["red_range"]
        self.blue_min, self.blue_max = config["blue_range"]
        self.red_count = config["red_count"]
        self.blue_count = config["blue_count"]
        self.red_total = self.red_max - self.red_min + 1
        self.blue_total = (self.blue_max - self.blue_min + 1) if self.blue_count > 0 and self.blue_max >= self.blue_min else 0

        self.window_size = 10
        self.red_models = []  # 每个号码一个二分类器: 出现/不出现
        self.blue_models = []
        self.is_trained = False

    def _build_features(self, history: List[Dict], idx: int) -> List[float]:
        """构建单期特征 (最近window_size期)"""
        features = []
        for j in range(max(0, idx - self.window_size), idx):
            for n in history[j]["reds"]:
                features.append(n)
            features.extend([0] * (self.red_count - len(history[j]["reds"])))
            blues = history[j]["blues"]
            features.extend(blues)
            features.extend([0] * (self.blue_count - len(blues)))
        # 补齐
        expected = self.window_size * (self.red_count + self.blue_count)
        while len(features) < expected:
            features.append(0)
        return features[:expected]

    def train(self, history: List[Dict]) -> Dict:
        if len(history) < self.window_size + 10:
            return {"success": False, "error": "数据量不足"}

        # 构建所有样本特征
        X = []
        for i in range(self.window_size, len(history)):
            X.append(self._build_features(history, i))
        X = np.array(X, dtype=np.float32)

        # 为每个红球号码训练二分类器 (是否出现)
        self.red_models = []
        red_labels = np.zeros((len(X), self.red_total))
        for i, idx in enumerate(range(self.window_size, len(history))):
            for n in history[idx]["reds"]:
                if self.red_min <= n <= self.red_max:
                    red_labels[i][n - self.red_min] = 1

        for n in range(self.red_total):
            model = MultinomialNB(alpha=1.0)
            model.fit(X, red_labels[:, n])
            self.red_models.append(model)

        # 蓝球
        self.blue_models = []
        if self.blue_count > 0:
            blue_labels = np.zeros((len(X), self.blue_total))
            for i, idx in enumerate(range(self.window_size, len(history))):
                for b in history[idx]["blues"]:
                    if self.blue_min <= b <= self.blue_max:
                        blue_labels[i][b - self.blue_min] = 1
            for n in range(self.blue_total):
                model = MultinomialNB(alpha=1.0)
                model.fit(X, blue_labels[:, n])
                self.blue_models.append(model)

        self.is_trained = True
        return {"success": True, "samples": len(X), "metrics": {}}

    def predict(self, history: List[Dict]) -> Tuple[List[int], List[int], Dict]:
        if not self.is_trained or len(history) < self.window_size:
            return [], [], {}

        features = self._build_features(history, len(history))
        X = np.array([features], dtype=np.float32)

        # 计算每个红球号码出现的概率
        red_probs = np.zeros(self.red_total)
        for n, model in enumerate(self.red_models):
            probs = model.predict_proba(X)[0]
            # 第二类是"出现"
            if len(model.classes_) == 2:
                red_probs[n] = probs[1]
            else:
                red_probs[n] = 0.5

        is_repeatable = (self.red_max - self.red_min + 1) <= 10 and self.red_count >= 3

        if is_repeatable:
            # 按位置选top1 - 但朴素贝叶斯不区分位置，这里取top red_count
            # 简化: 直接取前red_count个最高概率号码
            top_indices = np.argsort(red_probs)[-self.red_count:][::-1]
            reds = [int(idx) + self.red_min for idx in top_indices]
        else:
            top_indices = np.argsort(red_probs)[-self.red_count:][::-1]
            reds = sorted([int(idx) + self.red_min for idx in top_indices])

        # 蓝球
        blues = []
        blue_probs = np.zeros(max(self.blue_total, 1))
        if self.blue_count > 0 and len(self.blue_models) > 0:
            for n, model in enumerate(self.blue_models):
                probs = model.predict_proba(X)[0]
                if len(model.classes_) == 2:
                    blue_probs[n] = probs[1]
                else:
                    blue_probs[n] = 0.5
            top_blue = np.argsort(blue_probs)[-self.blue_count:][::-1]
            blues = sorted([int(idx) + self.blue_min for idx in top_blue])

        info = {
            "red_top_probs": [(int(idx) + self.red_min, round(float(red_probs[idx]), 4))
                              for idx in np.argsort(red_probs)[-10:][::-1]],
        }
        if self.blue_count > 0:
            info["blue_top_probs"] = [(int(idx) + self.blue_min, round(float(blue_probs[idx]), 4))
                                      for idx in np.argsort(blue_probs)[-5:][::-1]]
        return reds, blues, info

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "nb_model.pkl"), "wb") as f:
            pickle.dump({
                "red_models": self.red_models,
                "blue_models": self.blue_models,
                "is_trained": self.is_trained,
                "window_size": self.window_size,
            }, f)

    def load(self, model_dir: str) -> bool:
        path = os.path.join(model_dir, "nb_model.pkl")
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.red_models = data["red_models"]
        self.blue_models = data["blue_models"]
        self.is_trained = data["is_trained"]
        self.window_size = data["window_size"]
        return True
