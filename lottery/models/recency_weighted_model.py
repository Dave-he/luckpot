"""
LotteryAi 思想 - Recency-Weighted 近期加权预测模型

思路来源: LotteryAi (CorvusCodex/LotteryAi)
- Recency-Weighted Sampling: 近期数据采样权重最高可达20倍
- 核心思想: 彩票号码分布有微弱的近期趋势，越近期的数据权重越高

实现方式:
- 构建指数衰减权重: weight = exp(-i / tau)，其中 i 是距离当期的期数
- 加权统计每个号码的出现频次
- 按加权频次从高到低选出 top_k
- 支持多窗口混合 (近10期/近30期/近100期 不同衰减系数)
"""
import os
import pickle
import numpy as np
from typing import List, Dict, Tuple


class RecencyWeightedPredictor:
    """LotteryAi 近期加权频率预测器"""

    def __init__(self, config: Dict):
        self.config = config
        self.red_min, self.red_max = config["red_range"]
        self.blue_min, self.blue_max = config["blue_range"]
        self.red_count = config["red_count"]
        self.blue_count = config["blue_count"]
        self.red_total = self.red_max - self.red_min + 1
        self.blue_total = (self.blue_max - self.blue_min + 1) if self.blue_count > 0 and self.blue_max >= self.blue_min else 0

        self.red_weights = None  # 每个红球号码的加权得分
        self.blue_weights = None
        self.is_trained = False

        # 多窗口参数
        self.windows = [10, 30, 100]       # 三个时间窗口
        self.window_weights = [0.5, 0.3, 0.2]  # 每个窗口的权重
        self.decay_tau = 20  # 指数衰减半衰期 (期数)

    def _weighted_freq(self, history: List[Dict], color: str = "red") -> np.ndarray:
        """计算加权频率

        Args:
            history: 历史数据 (从旧到新)
            color: 'red' 或 'blue'
        Returns:
            每个号码的加权得分数组
        """
        if color == "red":
            total = self.red_total
            min_val = self.red_min
            count_pos = self.red_count
        else:
            total = self.blue_total
            min_val = self.blue_min
            count_pos = self.blue_count

        if total == 0:
            return np.array([])

        final_score = np.zeros(total)
        total_w = 0.0

        for win_size, win_w in zip(self.windows, self.window_weights):
            # 取最近 win_size 期
            recent = history[-win_size:] if len(history) >= win_size else history
            n = len(recent)

            # 指数衰减权重: 越近权重越高
            # w_i = exp(-(n-1-i) / tau), i 从 0 (最旧) 到 n-1 (最新)
            indices = np.arange(n)
            weights = np.exp(-(n - 1 - indices) / self.decay_tau)
            weights = weights / weights.sum()  # 归一化

            freq = np.zeros(total)
            for i, h in enumerate(recent):
                nums = h["reds"] if color == "red" else h["blues"]
                if not nums:
                    continue
                # 去重 (非重复彩种)
                for num in nums:
                    idx = num - min_val
                    if 0 <= idx < total:
                        freq[idx] += weights[i]

            final_score += freq * win_w
            total_w += win_w

        if total_w > 0:
            final_score /= total_w

        return final_score

    def train(self, history: List[Dict]) -> Dict:
        """训练 (计算加权频率)"""
        if len(history) < 10:
            return {"success": False, "error": "数据量不足"}

        self.red_weights = self._weighted_freq(history, "red")
        self.blue_weights = self._weighted_freq(history, "blue")
        self.is_trained = True

        metrics = {}
        if len(self.red_weights) > 0:
            metrics["red_max_prob"] = round(float(self.red_weights.max()), 4)
            metrics["red_top5"] = [
                int(i + self.red_min)
                for i in np.argsort(-self.red_weights)[:5]
            ]
        if len(self.blue_weights) > 0:
            metrics["blue_max_prob"] = round(float(self.blue_weights.max()), 4)

        return {"success": True, "metrics": metrics}

    def predict(self, history: List[Dict]) -> Tuple[List[int], List[int], Dict]:
        """预测下一期号码"""
        if not self.is_trained:
            self.train(history)

        info = {}

        # 红球: 按加权得分从高到低选 top red_count
        red_scores = self.red_weights.copy()
        red_indices = np.argsort(-red_scores)[:self.red_count]
        red_pred = sorted([int(i + self.red_min) for i in red_indices])

        info["red_top_probs"] = [float(red_scores[i]) for i in np.argsort(-red_scores)]
        info["red_top_numbers"] = [int(i + self.red_min) for i in np.argsort(-red_scores)[:10]]

        # 蓝球
        blue_pred = []
        if self.blue_total > 0 and self.blue_count > 0 and len(self.blue_weights) > 0:
            blue_scores = self.blue_weights.copy()
            blue_indices = np.argsort(-blue_scores)[:self.blue_count]
            blue_pred = sorted([int(i + self.blue_min) for i in blue_indices])
            info["blue_top_probs"] = [float(blue_scores[i]) for i in np.argsort(-blue_scores)]
            info["blue_top_numbers"] = [int(i + self.blue_min) for i in np.argsort(-blue_scores)[:8]]

        return red_pred, blue_pred, info

    def save(self, save_dir: str) -> bool:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "rw_model.pkl")
        try:
            with open(path, "wb") as f:
                pickle.dump({
                    "red_weights": self.red_weights,
                    "blue_weights": self.blue_weights,
                    "config": self.config,
                    "is_trained": self.is_trained,
                }, f)
            return True
        except Exception:
            return False

    def load(self, save_dir: str) -> bool:
        path = os.path.join(save_dir, "rw_model.pkl")
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.red_weights = data["red_weights"]
            self.blue_weights = data["blue_weights"]
            self.is_trained = data.get("is_trained", True)
            return True
        except Exception:
            return False
