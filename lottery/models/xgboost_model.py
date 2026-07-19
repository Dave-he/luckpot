"""
XGBoost彩票号码预测模型

思路: 将每个号码位置作为一个独立的回归/分类问题
- 红球: 对每个号码位置训练一个XGBoost回归模型，预测该位置的号码值
- 蓝球: 同理
- 特征: 历史N期的号码、和值、奇偶比、遗漏值等统计特征
"""
import os
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from sklearn.preprocessing import StandardScaler
import xgboost as xgb


class XGBoostPredictor:
    """基于XGBoost的彩票号码预测器"""

    def __init__(self, config: Dict):
        self.config = config
        self.red_min, self.red_max = config["red_range"]
        self.blue_min, self.blue_max = config["blue_range"]
        self.red_count = config["red_count"]
        self.blue_count = config["blue_count"]
        self.red_total = self.red_max - self.red_min + 1
        self.blue_total = (self.blue_max - self.blue_min + 1) if self.blue_count > 0 and self.blue_max >= self.blue_min else 0

        self.window_size = 20
        self.red_models = []  # 每个红球位置一个模型
        self.blue_models = []  # 每个蓝球位置一个模型
        self.scaler = StandardScaler()
        self.is_trained = False

    def _build_features(self, history: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """构建特征和标签"""
        X, y_red, y_blue = [], [], []

        for i in range(self.window_size, len(history)):
            features = []
            # 1. 历史window_size期的红球号码
            for j in range(i - self.window_size, i):
                for n in history[j]["reds"]:
                    features.append(n)
                # 补齐到固定长度
                features.extend([0] * (self.red_count - len(history[j]["reds"])))

            # 2. 历史window_size期的蓝球号码
            for j in range(i - self.window_size, i):
                blues = history[j]["blues"]
                features.extend(blues)
                features.extend([0] * (self.blue_count - len(blues)))

            # 3. 统计特征: 最近10期每个号码出现次数
            recent_10 = history[max(0, i - 10):i]
            red_freq = np.zeros(self.red_total)
            blue_freq = np.zeros(max(self.blue_total, 1))
            for rec in recent_10:
                for n in rec["reds"]:
                    if self.red_min <= n <= self.red_max:
                        red_freq[n - self.red_min] += 1
                for n in rec["blues"]:
                    if self.blue_min <= n <= self.blue_max:
                        blue_freq[n - self.blue_min] += 1
            features.extend(red_freq.tolist())
            if self.blue_count > 0:
                features.extend(blue_freq.tolist())

            # 4. 和值、奇偶比、跨度
            recent_5 = history[max(0, i - 5):i]
            for rec in recent_5:
                red_sum = sum(rec["reds"])
                red_odd = sum(1 for n in rec["reds"] if n % 2 == 1)
                red_span = max(rec["reds"]) - min(rec["reds"])
                features.extend([red_sum, red_odd, red_span])

            X.append(features)
            y_red.append(sorted(history[i]["reds"]))
            y_blue.append(sorted(history[i]["blues"]))

        return np.array(X), np.array(y_red), np.array(y_blue)

    def train(self, history: List[Dict]) -> Dict:
        """训练XGBoost模型"""
        if len(history) < self.window_size + 10:
            return {"success": False, "error": "数据量不足"}

        X, y_red, y_blue = self._build_features(history)

        if len(X) == 0:
            return {"success": False, "error": "无有效训练数据"}

        # 标准化特征
        X = self.scaler.fit_transform(X)

        # 训练红球模型 (每个位置一个模型)
        self.red_models = []
        for pos in range(self.red_count):
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
            )
            model.fit(X, y_red[:, pos])
            self.red_models.append(model)

        # 训练蓝球模型
        self.blue_models = []
        if self.blue_count > 0 and y_blue.shape[1] > 0:
            for pos in range(self.blue_count):
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbosity=0,
                )
                model.fit(X, y_blue[:, pos])
                self.blue_models.append(model)

        self.is_trained = True

        # 计算训练集上的准确率
        train_metrics = self._evaluate(X, y_red, y_blue)
        return {"success": True, "samples": len(X), "metrics": train_metrics}

    def _evaluate(self, X: np.ndarray, y_red: np.ndarray, y_blue: np.ndarray) -> Dict:
        """评估训练效果"""
        red_preds = []
        for pos, model in enumerate(self.red_models):
            pred = model.predict(X)
            red_preds.append(pred)
        red_preds = np.array(red_preds).T

        red_mse = float(np.mean((red_preds - y_red) ** 2))
        red_mae = float(np.mean(np.abs(red_preds - y_red)))

        metrics = {"red_mse": round(red_mse, 4), "red_mae": round(red_mae, 4)}

        if self.blue_count > 0 and len(self.blue_models) > 0:
            blue_preds = []
            for pos, model in enumerate(self.blue_models):
                pred = model.predict(X)
                blue_preds.append(pred)
            blue_preds = np.array(blue_preds).T
            blue_mse = float(np.mean((blue_preds - y_blue) ** 2))
            blue_mae = float(np.mean(np.abs(blue_preds - y_blue)))
            metrics["blue_mse"] = round(blue_mse, 4)
            metrics["blue_mae"] = round(blue_mae, 4)

        return metrics

    def predict(self, history: List[Dict]) -> Tuple[List[int], List[int], Dict]:
        """预测下一期号码"""
        if not self.is_trained or len(history) < self.window_size:
            return [], [], {}

        # 构建最新一期的特征
        recent = history[-self.window_size:]
        features = []
        for rec in recent:
            for n in rec["reds"]:
                features.append(n)
            features.extend([0] * (self.red_count - len(rec["reds"])))
        for rec in recent:
            blues = rec["blues"]
            features.extend(blues)
            features.extend([0] * (self.blue_count - len(blues)))

        # 统计特征
        recent_10 = history[-10:]
        red_freq = np.zeros(self.red_total)
        blue_freq = np.zeros(max(self.blue_total, 1))
        for rec in recent_10:
            for n in rec["reds"]:
                if self.red_min <= n <= self.red_max:
                    red_freq[n - self.red_min] += 1
            for n in rec["blues"]:
                if self.blue_min <= n <= self.blue_max:
                    blue_freq[n - self.blue_min] += 1
        features.extend(red_freq.tolist())
        if self.blue_count > 0:
            features.extend(blue_freq.tolist())

        recent_5 = history[-5:]
        for rec in recent_5:
            red_sum = sum(rec["reds"])
            red_odd = sum(1 for n in rec["reds"] if n % 2 == 1)
            red_span = max(rec["reds"]) - min(rec["reds"])
            features.extend([red_sum, red_odd, red_span])

        X = np.array([features])
        X = self.scaler.transform(X)

        # 预测红球
        red_preds = []
        for pos, model in enumerate(self.red_models):
            pred = model.predict(X)[0]
            red_preds.append(pred)

        # 判断是否为可重复号码彩种(七星彩/排列3/排列5等数字型)
        is_repeatable = (self.red_max - self.red_min + 1) <= 10 and self.red_count >= 3

        # 将预测值映射到合法范围
        red_candidates = []
        for pred in red_preds:
            n = int(round(pred))
            n = max(self.red_min, min(self.red_max, n))
            red_candidates.append(n)

        if is_repeatable:
            # 可重复号码: 不去重, 按位置返回
            reds = red_candidates
        else:
            # 不可重复号码: 去重
            reds = self._deduplicate_reds(red_preds, red_candidates)

        # 预测蓝球
        blues = []
        if self.blue_count > 0 and len(self.blue_models) > 0:
            blue_preds = []
            for pos, model in enumerate(self.blue_models):
                pred = model.predict(X)[0]
                blue_preds.append(pred)
            for pred in blue_preds:
                n = int(round(pred))
                n = max(self.blue_min, min(self.blue_max, n))
                blues.append(n)
            blues = sorted(list(set(blues)))
            # 如果去重后不足，补充
            while len(blues) < self.blue_count:
                for pred in sorted(blue_preds, key=lambda x: abs(x - round(x))):
                    n = int(round(pred))
                    if n not in blues and self.blue_min <= n <= self.blue_max:
                        blues.append(n)
                        blues.sort()
                        if len(blues) >= self.blue_count:
                            break
                break

        info = {
            "raw_red_preds": [round(float(p), 2) for p in red_preds],
            "raw_blue_preds": [round(float(p), 2) for p in (blue_preds if self.blue_count > 0 else [])],
        }
        return sorted(reds), sorted(blues[:self.blue_count]), info

    def _deduplicate_reds(self, raw_preds: List[float], candidates: List[int]) -> List[int]:
        """红球去重，保持预测概率最大的号码"""
        reds = []
        used = set()
        # 按原始预测值排序，优先选择预测值最接近整数的
        indexed = [(i, p, candidates[i]) for i, p in enumerate(raw_preds)]
        indexed.sort(key=lambda x: abs(x[1] - round(x[1])))

        for _, _, n in indexed:
            if n not in used and self.red_min <= n <= self.red_max:
                reds.append(n)
                used.add(n)
            if len(reds) >= self.red_count:
                break

        # 如果不足，从附近号码补充
        while len(reds) < self.red_count:
            for n in range(self.red_min, self.red_max + 1):
                if n not in used:
                    reds.append(n)
                    used.add(n)
                    break
        return reds

    def save(self, model_dir: str):
        """保存模型"""
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "xgb_model.pkl"), "wb") as f:
            pickle.dump({
                "red_models": self.red_models,
                "blue_models": self.blue_models,
                "scaler": self.scaler,
                "is_trained": self.is_trained,
                "window_size": self.window_size,
            }, f)

    def load(self, model_dir: str) -> bool:
        """加载模型"""
        path = os.path.join(model_dir, "xgb_model.pkl")
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
