"""
马尔可夫链彩票预测模型

思路: 对每个号码位置建模为状态转移过程
- 统计每个位置号码的转移概率矩阵 P(i->j)
- 给定最近一期号码，预测下一期最可能的号码
- 适用于按位置的彩种 (七星彩/排列3/排列5等)
"""
import os
import pickle
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


class MarkovPredictor:
    """基于马尔可夫链的彩票号码预测器"""

    def __init__(self, config: Dict):
        self.config = config
        self.red_min, self.red_max = config["red_range"]
        self.blue_min, self.blue_max = config["blue_range"]
        self.red_count = config["red_count"]
        self.blue_count = config["blue_count"]
        self.red_total = self.red_max - self.red_min + 1
        self.blue_total = (self.blue_max - self.blue_min + 1) if self.blue_count > 0 and self.blue_max >= self.blue_min else 0

        # 每个红球位置一个转移矩阵
        self.red_transitions = []  # list of (red_total, red_total) numpy arrays
        self.blue_transitions = []
        self.is_trained = False

        # 一阶+二阶混合 (考虑前两期)
        self.red_transitions_2 = []  # 二阶转移
        self.blue_transitions_2 = []

    def train(self, history: List[Dict]) -> Dict:
        if len(history) < 10:
            return {"success": False, "error": "数据量不足"}

        # 一阶转移矩阵 (每个位置)
        self.red_transitions = []
        for pos in range(self.red_count):
            trans = np.ones((self.red_total, self.red_total)) * 0.01  # 拉普拉斯平滑
            for i in range(1, len(history)):
                prev_reds = history[i - 1]["reds"]
                curr_reds = history[i]["reds"]
                if pos < len(prev_reds) and pos < len(curr_reds):
                    p = prev_reds[pos] - self.red_min
                    c = curr_reds[pos] - self.red_min
                    if 0 <= p < self.red_total and 0 <= c < self.red_total:
                        trans[p][c] += 1
            # 归一化
            row_sums = trans.sum(axis=1, keepdims=True)
            trans = trans / row_sums
            self.red_transitions.append(trans)

        # 蓝球一阶转移
        self.blue_transitions = []
        if self.blue_count > 0:
            for pos in range(self.blue_count):
                trans = np.ones((self.blue_total, self.blue_total)) * 0.01
                for i in range(1, len(history)):
                    prev_blues = history[i - 1]["blues"]
                    curr_blues = history[i]["blues"]
                    if pos < len(prev_blues) and pos < len(curr_blues):
                        p = prev_blues[pos] - self.blue_min
                        c = curr_blues[pos] - self.blue_min
                        if 0 <= p < self.blue_total and 0 <= c < self.blue_total:
                            trans[p][c] += 1
                row_sums = trans.sum(axis=1, keepdims=True)
                trans = trans / row_sums
                self.blue_transitions.append(trans)

        # 二阶转移 (前两期到当前)
        self.red_transitions_2 = []
        for pos in range(self.red_count):
            # 状态: (prev_prev, prev) -> curr
            # 用一阶矩阵的乘积近似
            if len(self.red_transitions) > pos:
                self.red_transitions_2.append(self.red_transitions[pos] @ self.red_transitions[pos])

        self.is_trained = True
        # 评估: 在历史数据上计算平均转移概率
        avg_red_prob = float(np.mean([np.mean(np.diag(t)) for t in self.red_transitions]))
        return {"success": True, "samples": len(history),
                "metrics": {"avg_diag_prob": round(avg_red_prob, 4)}}

    def predict(self, history: List[Dict]) -> Tuple[List[int], List[int], Dict]:
        if not self.is_trained or len(history) < 2:
            return [], [], {}

        last = history[-1]
        prev = history[-2] if len(history) >= 2 else last

        # 红球: 每个位置基于一阶+二阶混合预测
        reds = []
        red_probs_all = []
        for pos, trans in enumerate(self.red_transitions):
            if pos < len(last["reds"]):
                last_n = last["reds"][pos] - self.red_min
                if 0 <= last_n < self.red_total:
                    # 混合一阶和二阶
                    p1 = trans[last_n]
                    if pos < len(self.red_transitions_2):
                        p2 = self.red_transitions_2[pos][last_n]
                        p = 0.7 * p1 + 0.3 * p2
                    else:
                        p = p1
                    red_probs_all.append(p)
                    idx = int(np.argmax(p))
                    reds.append(idx + self.red_min)
                else:
                    red_probs_all.append(np.ones(self.red_total) / self.red_total)
                    reds.append(self.red_min)
            else:
                red_probs_all.append(np.ones(self.red_total) / self.red_total)
                reds.append(self.red_min)

        is_repeatable = (self.red_max - self.red_min + 1) <= 10 and self.red_count >= 3

        if not is_repeatable:
            # 不可重复: 综合所有位置概率取top
            avg_probs = np.mean(red_probs_all, axis=0)
            top_indices = np.argsort(avg_probs)[-self.red_count:][::-1]
            reds = sorted([int(idx) + self.red_min for idx in top_indices])
            avg_red = avg_probs
        else:
            avg_red = np.mean(red_probs_all, axis=0)

        # 蓝球
        blues = []
        avg_blue = np.zeros(max(self.blue_total, 1))
        if self.blue_count > 0 and len(self.blue_transitions) > 0:
            blue_probs_all = []
            for pos, trans in enumerate(self.blue_transitions):
                if pos < len(last["blues"]):
                    last_b = last["blues"][pos] - self.blue_min
                    if 0 <= last_b < self.blue_total:
                        p = trans[last_b]
                    else:
                        p = np.ones(self.blue_total) / self.blue_total
                else:
                    p = np.ones(self.blue_total) / self.blue_total
                blue_probs_all.append(p)
            avg_blue = np.mean(blue_probs_all, axis=0)
            top_blue = np.argsort(avg_blue)[-self.blue_count:][::-1]
            blues = sorted([int(idx) + self.blue_min for idx in top_blue])

        info = {
            "red_top_probs": [(int(idx) + self.red_min, round(float(avg_red[idx]), 4))
                              for idx in np.argsort(avg_red)[-10:][::-1]],
        }
        if self.blue_count > 0:
            info["blue_top_probs"] = [(int(idx) + self.blue_min, round(float(avg_blue[idx]), 4))
                                      for idx in np.argsort(avg_blue)[-5:][::-1]]
        return reds, blues, info

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "markov_model.pkl"), "wb") as f:
            pickle.dump({
                "red_transitions": self.red_transitions,
                "blue_transitions": self.blue_transitions,
                "red_transitions_2": self.red_transitions_2,
                "is_trained": self.is_trained,
            }, f)

    def load(self, model_dir: str) -> bool:
        path = os.path.join(model_dir, "markov_model.pkl")
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.red_transitions = data["red_transitions"]
        self.blue_transitions = data["blue_transitions"]
        self.red_transitions_2 = data.get("red_transitions_2", [])
        self.is_trained = data["is_trained"]
        return True
