"""
微型神经网络彩票预测模型 (基于numpy实现MLP)

避免依赖重量级框架(tf/torch)，使用纯numpy实现一个三层MLP
- 输入: 历史window_size期的号码特征
- 输出: 每个号码位置的概率分布
- 损失: 多分类交叉熵
"""
import os
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional


class MLPModel:
    """纯numpy实现的三层MLP"""

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 lr: float = 0.01, epochs: int = 200, seed: int = 42):
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        rng = np.random.RandomState(seed)

        # He初始化
        self.W1 = rng.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = rng.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(hidden_size)
        self.W3 = rng.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b3 = np.zeros(output_size)

    def _relu(self, x):
        return np.maximum(0, x)

    def _softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self._relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self._relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        return self._softmax(self.z3)

    def predict_proba(self, X):
        return self.forward(X)

    def train(self, X, y, verbose: bool = False):
        """训练, y为类别索引(0~output_size-1)"""
        n = X.shape[0]
        y_onehot = np.zeros((n, self.W3.shape[1]))
        y_onehot[np.arange(n), y] = 1

        for epoch in range(self.epochs):
            # 前向
            probs = self.forward(X)

            # 反向传播
            dz3 = (probs - y_onehot) / n
            dW3 = self.a2.T @ dz3
            db3 = np.sum(dz3, axis=0)

            da2 = dz3 @ self.W3.T
            dz2 = da2 * (self.z2 > 0)
            dW2 = self.a1.T @ dz2
            db2 = np.sum(dz2, axis=0)

            da1 = dz2 @ self.W2.T
            dz1 = da1 * (self.z1 > 0)
            dW1 = X.T @ dz1
            db1 = np.sum(dz1, axis=0)

            # 参数更新
            self.W3 -= self.lr * dW3
            self.b3 -= self.lr * db3
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

            if verbose and (epoch + 1) % 50 == 0:
                loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-8), axis=1))
                acc = np.mean(np.argmax(probs, axis=1) == y)
                print(f"    epoch {epoch+1}/{self.epochs}, loss={loss:.4f}, acc={acc:.4f}")


class MLPredictor:
    """基于MLP的彩票号码预测器"""

    def __init__(self, config: Dict):
        self.config = config
        self.red_min, self.red_max = config["red_range"]
        self.blue_min, self.blue_max = config["blue_range"]
        self.red_count = config["red_count"]
        self.blue_count = config["blue_count"]
        self.red_total = self.red_max - self.red_min + 1
        self.blue_total = (self.blue_max - self.blue_min + 1) if self.blue_count > 0 and self.blue_max >= self.blue_min else 0

        self.window_size = 15
        self.red_models = []  # 每个红球位置一个分类模型
        self.blue_models = []
        self.feature_size = 0
        self.is_trained = False

    def _build_features(self, history: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """构建特征(用one-hot)和标签(类别索引)"""
        X, y_red, y_blue = [], [], []

        for i in range(self.window_size, len(history)):
            features = []
            # 历史window_size期的红球one-hot
            for j in range(i - self.window_size, i):
                red_vec = np.zeros(self.red_total)
                for n in history[j]["reds"]:
                    if self.red_min <= n <= self.red_max:
                        red_vec[n - self.red_min] = 1
                features.extend(red_vec.tolist())

                if self.blue_count > 0:
                    blue_vec = np.zeros(max(self.blue_total, 1))
                    for n in history[j]["blues"]:
                        if self.blue_min <= n <= self.blue_max:
                            blue_vec[n - self.blue_min] = 1
                    features.extend(blue_vec.tolist())

            X.append(features)

            # 标签: 每个红球位置的号码类别索引
            # 注意: 对于可重复号码的彩种(七星彩/排列3/排列5), 号码按位置存储, 不排序
            # 对于不可重复号码的彩种(双色球/大乐透等), 号码已排序存储
            y_red_pos = [r - self.red_min for r in history[i]["reds"]]
            # 防止索引越界
            y_red_pos = [max(0, min(self.red_total - 1, p)) for p in y_red_pos]
            # 补齐到red_count长度
            while len(y_red_pos) < self.red_count:
                y_red_pos.append(0)
            y_red.append(y_red_pos[:self.red_count])

            blues_sorted = sorted(history[i]["blues"])
            y_blue_pos = [b - self.blue_min for b in blues_sorted] if self.blue_count > 0 else []
            y_blue.append(y_blue_pos)

        return np.array(X, dtype=np.float32), np.array(y_red), np.array(y_blue)

    def train(self, history: List[Dict]) -> Dict:
        """训练MLP模型"""
        if len(history) < self.window_size + 10:
            return {"success": False, "error": "数据量不足"}

        X, y_red, y_blue = self._build_features(history)
        if len(X) == 0:
            return {"success": False, "error": "无有效训练数据"}

        self.feature_size = X.shape[1]
        hidden_size = 128

        # 训练红球模型(每个位置一个)
        self.red_models = []
        for pos in range(self.red_count):
            model = MLPModel(
                input_size=self.feature_size,
                hidden_size=hidden_size,
                output_size=self.red_total,
                lr=0.01,
                epochs=150,
                seed=42 + pos,
            )
            y_pos = y_red[:, pos]
            model.train(X, y_pos, verbose=False)
            self.red_models.append(model)

        # 训练蓝球模型
        self.blue_models = []
        if self.blue_count > 0 and y_blue.shape[1] > 0:
            for pos in range(self.blue_count):
                model = MLPModel(
                    input_size=self.feature_size,
                    hidden_size=64,
                    output_size=self.blue_total,
                    lr=0.01,
                    epochs=150,
                    seed=100 + pos,
                )
                y_pos = y_blue[:, pos]
                model.train(X, y_pos, verbose=False)
                self.blue_models.append(model)

        self.is_trained = True

        # 评估
        train_acc = self._evaluate(X, y_red, y_blue)
        return {"success": True, "samples": len(X), "metrics": train_acc}

    def _evaluate(self, X, y_red, y_blue) -> Dict:
        """评估训练集准确率"""
        red_acc = []
        for pos, model in enumerate(self.red_models):
            probs = model.predict_proba(X)
            pred = np.argmax(probs, axis=1)
            acc = float(np.mean(pred == y_red[:, pos]))
            red_acc.append(acc)

        metrics = {"red_acc_mean": round(float(np.mean(red_acc)), 4),
                    "red_acc_per_pos": [round(a, 4) for a in red_acc]}

        if self.blue_count > 0 and len(self.blue_models) > 0:
            blue_acc = []
            for pos, model in enumerate(self.blue_models):
                probs = model.predict_proba(X)
                pred = np.argmax(probs, axis=1)
                acc = float(np.mean(pred == y_blue[:, pos]))
                blue_acc.append(acc)
            metrics["blue_acc_mean"] = round(float(np.mean(blue_acc)), 4)

        return metrics

    def predict(self, history: List[Dict]) -> Tuple[List[int], List[int], Dict]:
        """预测下一期号码"""
        if not self.is_trained or len(history) < self.window_size:
            return [], [], {}

        recent = history[-self.window_size:]
        features = []
        for rec in recent:
            red_vec = np.zeros(self.red_total)
            for n in rec["reds"]:
                if self.red_min <= n <= self.red_max:
                    red_vec[n - self.red_min] = 1
            features.extend(red_vec.tolist())

            if self.blue_count > 0:
                blue_vec = np.zeros(max(self.blue_total, 1))
                for n in rec["blues"]:
                    if self.blue_min <= n <= self.blue_max:
                        blue_vec[n - self.blue_min] = 1
                features.extend(blue_vec.tolist())

        X = np.array([features], dtype=np.float32)

        # 红球: 每个位置预测概率
        red_probs_all = []
        for model in self.red_models:
            probs = model.predict_proba(X)[0]
            red_probs_all.append(probs)

        # 判断是否为可重复号码彩种(七星彩/排列3/排列5等数字型)
        # 特征: 红球范围<=10且红球数量>=3
        is_repeatable = (self.red_max - self.red_min + 1) <= 10 and self.red_count >= 3

        if is_repeatable:
            # 可重复号码: 每个位置独立预测(argmax)
            reds = []
            for probs in red_probs_all:
                idx = int(np.argmax(probs))
                reds.append(idx + self.red_min)
            avg_probs = np.mean(red_probs_all, axis=0)
            top_indices = np.argsort(avg_probs)[-self.red_count:][::-1]
        else:
            # 不可重复号码: 综合所有位置的预测概率，选择red_count个号码
            avg_probs = np.mean(red_probs_all, axis=0)
            top_indices = np.argsort(avg_probs)[-self.red_count:][::-1]
            reds = sorted([int(idx) + self.red_min for idx in top_indices])

        # 蓝球
        blues = []
        if self.blue_count > 0 and len(self.blue_models) > 0:
            blue_probs_all = []
            for model in self.blue_models:
                probs = model.predict_proba(X)[0]
                blue_probs_all.append(probs)
            avg_blue = np.mean(blue_probs_all, axis=0)
            top_blue = np.argsort(avg_blue)[-self.blue_count:][::-1]
            blues = sorted([int(idx) + self.blue_min for idx in top_blue])

        info = {
            "red_top_probs": [(int(idx) + self.red_min, round(float(avg_probs[idx]), 4))
                              for idx in top_indices[:10]],
        }
        if self.blue_count > 0:
            info["blue_top_probs"] = [(int(idx) + self.blue_min, round(float(avg_blue[idx]), 4))
                                      for idx in top_blue[:5]]
        return reds, blues, info

    def save(self, model_dir: str):
        """保存模型参数"""
        os.makedirs(model_dir, exist_ok=True)
        params = {
            "window_size": self.window_size,
            "feature_size": self.feature_size,
            "is_trained": self.is_trained,
            "red_models": [],
            "blue_models": [],
        }
        for m in self.red_models:
            params["red_models"].append({
                "W1": m.W1, "b1": m.b1, "W2": m.W2, "b2": m.b2,
                "W3": m.W3, "b3": m.b3, "lr": m.lr, "epochs": m.epochs, "seed": m.seed,
            })
        for m in self.blue_models:
            params["blue_models"].append({
                "W1": m.W1, "b1": m.b1, "W2": m.W2, "b2": m.b2,
                "W3": m.W3, "b3": m.b3, "lr": m.lr, "epochs": m.epochs, "seed": m.seed,
            })

        with open(os.path.join(model_dir, "mlp_model.pkl"), "wb") as f:
            pickle.dump(params, f)

    def load(self, model_dir: str) -> bool:
        """加载模型参数"""
        path = os.path.join(model_dir, "mlp_model.pkl")
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            params = pickle.load(f)

        self.window_size = params["window_size"]
        self.feature_size = params["feature_size"]
        self.is_trained = params["is_trained"]

        self.red_models = []
        for p in params["red_models"]:
            m = MLPModel(self.feature_size, p["W2"].shape[0], p["W3"].shape[1],
                         p["lr"], p["epochs"], p["seed"])
            m.W1, m.b1, m.W2, m.b2, m.W3, m.b3 = p["W1"], p["b1"], p["W2"], p["b2"], p["W3"], p["b3"]
            self.red_models.append(m)

        self.blue_models = []
        for p in params["blue_models"]:
            m = MLPModel(self.feature_size, p["W2"].shape[0], p["W3"].shape[1],
                         p["lr"], p["epochs"], p["seed"])
            m.W1, m.b1, m.W2, m.b2, m.W3, m.b3 = p["W1"], p["b1"], p["W2"], p["b2"], p["W3"], p["b3"]
            self.blue_models.append(m)
        return True
