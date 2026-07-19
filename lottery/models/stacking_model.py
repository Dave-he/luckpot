"""
Stacking 元学习器 - 用逻辑回归组合所有基础模型

思路:
- 第一层: 各基础模型预测每个号码的概率
- 第二层: 逻辑回归学习如何组合这些概率 (元特征)
- 输出: 综合所有基础模型的最优组合

简化版:
- 不实际训练元学习器 (数据量限制)
- 而是基于历史回测表现动态调整组合权重
- 类似 weighted_predict 但元学习器自动选择最优组合
"""
import os
import json
import numpy as np
from typing import List, Dict, Tuple
from sklearn.linear_model import LogisticRegression


class StackingPredictor:
    """Stacking 元学习预测器

    基于多个基础模型的预测概率，训练一个逻辑回归元学习器
    """

    def __init__(self, config: Dict):
        self.config = config
        self.red_min, self.red_max = config["red_range"]
        self.blue_min, self.blue_max = config["blue_range"]
        self.red_count = config["red_count"]
        self.blue_count = config["blue_count"]
        self.red_total = self.red_max - self.red_min + 1
        self.blue_total = (self.blue_max - self.blue_min + 1) if self.blue_count > 0 and self.blue_max >= self.blue_min else 0

        self.meta_red = None  # 红球元学习器
        self.meta_blue = None  # 蓝球元学习器
        self.is_trained = False
        self.base_predictors = []  # 基础预测器列表

    def set_base_predictors(self, predictors: List):
        """设置基础预测器列表"""
        self.base_predictors = predictors

    def _get_base_predictions(self, history: List[Dict]) -> Dict:
        """获取所有基础模型对每个号码的预测概率"""
        red_probs_per_model = []  # (n_models, red_total)
        blue_probs_per_model = []  # (n_models, blue_total)

        for predictor in self.base_predictors:
            try:
                reds, blues, info = predictor.predict(history)
                # 从info提取每个号码的概率
                red_probs = np.zeros(self.red_total)
                for n, p in info.get("red_top_probs", []):
                    if self.red_min <= n <= self.red_max:
                        red_probs[n - self.red_min] = p
                # 归一化
                if red_probs.sum() > 0:
                    red_probs = red_probs / red_probs.sum()
                else:
                    red_probs = np.ones(self.red_total) / self.red_total
                red_probs_per_model.append(red_probs)

                if self.blue_count > 0:
                    blue_probs = np.zeros(self.blue_total)
                    for n, p in info.get("blue_top_probs", []):
                        if self.blue_min <= n <= self.blue_max:
                            blue_probs[n - self.blue_min] = p
                    if blue_probs.sum() > 0:
                        blue_probs = blue_probs / blue_probs.sum()
                    else:
                        blue_probs = np.ones(self.blue_total) / self.blue_total
                    blue_probs_per_model.append(blue_probs)
            except Exception:
                red_probs_per_model.append(np.ones(self.red_total) / self.red_total)
                if self.blue_count > 0:
                    blue_probs_per_model.append(np.ones(self.blue_total) / self.blue_total)

        return {
            "red": np.array(red_probs_per_model) if red_probs_per_model else None,
            "blue": np.array(blue_probs_per_model) if blue_probs_per_model else None,
        }

    def train_meta(self, history: List[Dict], n_backtests: int = 20) -> Dict:
        """训练元学习器

        用历史回测获取基础模型预测, 用实际开奖作为标签
        """
        if len(history) < 50 or not self.base_predictors:
            return {"success": False, "error": "数据不足或无基础模型"}

        # 构建元特征: 每个基础模型对每个号码的概率
        # 标签: 该号码是否在实际开奖中出现 (二分类)
        X_meta_red = []
        y_meta_red = []
        X_meta_blue = []
        y_meta_blue = []

        total = len(history)
        n_bt = min(n_backtests, total - 50)
        backtest_points = list(range(total - n_bt, total))

        for idx in backtest_points:
            train_data = history[:idx]
            actual = history[idx]

            # 重训所有基础模型 (太慢,跳过 - 用当前已训练的模型)
            # 获取基础模型预测
            preds = self._get_base_predictions(train_data)
            if preds["red"] is None:
                continue

            # 每个号码一个样本: 特征=[各模型对该号码的概率]
            actual_red_set = set(actual["reds"])
            for n in range(self.red_total):
                num = n + self.red_min
                features = preds["red"][:, n]  # (n_models,)
                X_meta_red.append(features)
                y_meta_red.append(1 if num in actual_red_set else 0)

            if self.blue_count > 0 and preds["blue"] is not None:
                actual_blue_set = set(actual["blues"])
                for n in range(self.blue_total):
                    num = n + self.blue_min
                    features = preds["blue"][:, n]
                    X_meta_blue.append(features)
                    y_meta_blue.append(1 if num in actual_blue_set else 0)

        if len(X_meta_red) == 0:
            return {"success": False, "error": "无法构建元特征"}

        X_meta_red = np.array(X_meta_red)
        y_meta_red = np.array(y_meta_red)

        # 训练逻辑回归元学习器
        try:
            self.meta_red = LogisticRegression(
                C=1.0, max_iter=200, random_state=42,
                class_weight="balanced",
            )
            self.meta_red.fit(X_meta_red, y_meta_red)

            if self.blue_count > 0 and len(X_meta_blue) > 0:
                X_meta_blue = np.array(X_meta_blue)
                y_meta_blue = np.array(y_meta_blue)
                self.meta_blue = LogisticRegression(
                    C=1.0, max_iter=200, random_state=42,
                    class_weight="balanced",
                )
                self.meta_blue.fit(X_meta_blue, y_meta_blue)

            self.is_trained = True
            # 评估
            train_acc = float(self.meta_red.score(X_meta_red, y_meta_red))
            return {"success": True, "samples": len(X_meta_red),
                    "metrics": {"meta_train_acc": round(train_acc, 4)}}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def predict(self, history: List[Dict]) -> Tuple[List[int], List[int], Dict]:
        if not self.is_trained or not self.base_predictors:
            return [], [], {}

        # 获取基础模型预测
        preds = self._get_base_predictions(history)
        if preds["red"] is None:
            return [], [], {}

        # 用元学习器预测每个号码出现概率
        X_meta_red = preds["red"].T  # (red_total, n_models)
        red_proba = self.meta_red.predict_proba(X_meta_red)[:, 1]  # (red_total,)

        is_repeatable = (self.red_max - self.red_min + 1) <= 10 and self.red_count >= 3
        if is_repeatable:
            top_indices = np.argsort(red_proba)[-self.red_count:][::-1]
            reds = [int(idx) + self.red_min for idx in top_indices]
        else:
            top_indices = np.argsort(red_proba)[-self.red_count:][::-1]
            reds = sorted([int(idx) + self.red_min for idx in top_indices])

        blues = []
        blue_proba = np.zeros(max(self.blue_total, 1))
        if self.blue_count > 0 and self.meta_blue is not None and preds["blue"] is not None:
            X_meta_blue = preds["blue"].T
            blue_proba = self.meta_blue.predict_proba(X_meta_blue)[:, 1]
            top_blue = np.argsort(blue_proba)[-self.blue_count:][::-1]
            blues = sorted([int(idx) + self.blue_min for idx in top_blue])

        info = {
            "red_top_probs": [(int(idx) + self.red_min, round(float(red_proba[idx]), 4))
                              for idx in np.argsort(red_proba)[-10:][::-1]],
            "n_base_models": len(self.base_predictors),
        }
        if self.blue_count > 0:
            info["blue_top_probs"] = [(int(idx) + self.blue_min, round(float(blue_proba[idx]), 4))
                                      for idx in np.argsort(blue_proba)[-5:][::-1]]
        return reds, blues, info

    def save(self, model_dir: str):
        import pickle
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "stacking_model.pkl"), "wb") as f:
            pickle.dump({
                "meta_red": self.meta_red,
                "meta_blue": self.meta_blue,
                "is_trained": self.is_trained,
            }, f)

    def load(self, model_dir: str) -> bool:
        import pickle
        path = os.path.join(model_dir, "stacking_model.pkl")
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.meta_red = data["meta_red"]
        self.meta_blue = data["meta_blue"]
        self.is_trained = data["is_trained"]
        return True
