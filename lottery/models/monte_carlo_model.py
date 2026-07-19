"""
蒙特卡洛模拟彩票预测模型

思路: 基于历史频率分布进行大量随机抽样
- 用最近N期号码构建经验概率分布
- 大量(10000次)模拟下一期号码
- 统计每个号码在模拟结果中出现的频率
- 选出现频率最高的号码作为预测
"""
import os
import pickle
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter


class MonteCarloPredictor:
    """基于蒙特卡洛模拟的彩票预测器"""

    def __init__(self, config: Dict):
        self.config = config
        self.red_min, self.red_max = config["red_range"]
        self.blue_min, self.blue_max = config["blue_range"]
        self.red_count = config["red_count"]
        self.blue_count = config["blue_count"]
        self.red_total = self.red_max - self.red_min + 1
        self.blue_total = (self.blue_max - self.blue_min + 1) if self.blue_count > 0 and self.blue_max >= self.blue_min else 0

        self.n_simulations = 5000
        self.recent_window = 50  # 用最近50期构建概率
        # 存储历史频率
        self.red_freq = None
        self.blue_freq = None
        self.is_trained = False

    def train(self, history: List[Dict]) -> Dict:
        if len(history) < 20:
            return {"success": False, "error": "数据量不足"}

        # 用最近recent_window期构建频率
        recent = history[-self.recent_window:]
        self.red_freq = np.zeros(self.red_total)
        for rec in recent:
            for n in rec["reds"]:
                if self.red_min <= n <= self.red_max:
                    self.red_freq[n - self.red_min] += 1
        # 归一化为概率
        if self.red_freq.sum() > 0:
            self.red_freq = self.red_freq / self.red_freq.sum()
        else:
            self.red_freq = np.ones(self.red_total) / self.red_total

        if self.blue_count > 0:
            self.blue_freq = np.zeros(self.blue_total)
            for rec in recent:
                for b in rec["blues"]:
                    if self.blue_min <= b <= self.blue_max:
                        self.blue_freq[b - self.blue_min] += 1
            if self.blue_freq.sum() > 0:
                self.blue_freq = self.blue_freq / self.blue_freq.sum()
            else:
                self.blue_freq = np.ones(self.blue_total) / self.blue_total
        else:
            self.blue_freq = np.zeros(1)

        self.is_trained = True
        return {"success": True, "samples": len(recent),
                "metrics": {"red_max_prob": round(float(self.red_freq.max()), 4)}}

    def predict(self, history: List[Dict]) -> Tuple[List[int], List[int], Dict]:
        if not self.is_trained:
            return [], [], {}

        is_repeatable = (self.red_max - self.red_min + 1) <= 10 and self.red_count >= 3

        # 蒙特卡洛模拟
        red_sim_counts = np.zeros(self.red_total)
        blue_sim_counts = np.zeros(max(self.blue_total, 1))

        rng = np.random.default_rng(42)

        for _ in range(self.n_simulations):
            # 按频率随机抽样
            if is_repeatable:
                # 可重复: 每个位置独立采样
                sim_reds = rng.choice(self.red_total, size=self.red_count, p=self.red_freq, replace=True)
            else:
                # 不可重复: 按频率无放回采样
                sim_reds = rng.choice(self.red_total, size=self.red_count, p=self.red_freq, replace=False)
            for n in sim_reds:
                red_sim_counts[n] += 1

            if self.blue_count > 0:
                sim_blues = rng.choice(self.blue_total, size=self.blue_count, p=self.blue_freq, replace=False)
                for b in sim_blues:
                    blue_sim_counts[b] += 1

        # 模拟频率作为预测概率
        red_probs = red_sim_counts / red_sim_counts.sum() if red_sim_counts.sum() > 0 else self.red_freq
        blue_probs = blue_sim_counts / blue_sim_counts.sum() if self.blue_count > 0 and blue_sim_counts.sum() > 0 else self.blue_freq

        # 选号
        if is_repeatable:
            # 可重复: 取top red_count (允许重复)
            top_indices = np.argsort(red_probs)[-self.red_count:][::-1]
            reds = [int(idx) + self.red_min for idx in top_indices]
        else:
            top_indices = np.argsort(red_probs)[-self.red_count:][::-1]
            reds = sorted([int(idx) + self.red_min for idx in top_indices])

        blues = []
        if self.blue_count > 0:
            top_blue = np.argsort(blue_probs)[-self.blue_count:][::-1]
            blues = sorted([int(idx) + self.blue_min for idx in top_blue])

        info = {
            "n_simulations": self.n_simulations,
            "red_top_probs": [(int(idx) + self.red_min, round(float(red_probs[idx]), 4))
                              for idx in np.argsort(red_probs)[-10:][::-1]],
        }
        if self.blue_count > 0:
            info["blue_top_probs"] = [(int(idx) + self.blue_min, round(float(blue_probs[idx]), 4))
                                      for idx in np.argsort(blue_probs)[-5:][::-1]]
        return reds, blues, info

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "mc_model.pkl"), "wb") as f:
            pickle.dump({
                "red_freq": self.red_freq,
                "blue_freq": self.blue_freq,
                "is_trained": self.is_trained,
                "n_simulations": self.n_simulations,
                "recent_window": self.recent_window,
            }, f)

    def load(self, model_dir: str) -> bool:
        path = os.path.join(model_dir, "mc_model.pkl")
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.red_freq = data["red_freq"]
        self.blue_freq = data["blue_freq"]
        self.is_trained = data["is_trained"]
        self.n_simulations = data["n_simulations"]
        self.recent_window = data["recent_window"]
        return True
