"""
LotteryAi 思想 - Triggers & Followers 触发跟随预测模型

思路来源: LotteryAi (CorvusCodex/LotteryAi)
- Triggers & Followers: 当号码 X 出现时，下一期哪些号码出现概率显著提升
- 本质是条件概率 P(Y|X)，比马尔可夫1阶转移更细粒度（针对每对号码）
- Wolfpacks: 重复一起出现的三连号组/四连号组（群体共现模式）

实现方式:
- 统计每个号码 X 出现后，下一期各号码 Y 的出现次数
- 计算条件概率 P(Y|X) = count(X->Y) / count(X)
- 结合上一期开出的所有号码作为 triggers，加权预测下一期
- 同时统计 Wolfpacks (2码/3码组合共现概率)
"""
import os
import pickle
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


class TriggersFollowersPredictor:
    """LotteryAi 触发跟随预测器"""

    def __init__(self, config: Dict):
        self.config = config
        self.red_min, self.red_max = config["red_range"]
        self.blue_min, self.blue_max = config["blue_range"]
        self.red_count = config["red_count"]
        self.blue_count = config["blue_count"]
        self.red_total = self.red_max - self.red_min + 1
        self.blue_total = (self.blue_max - self.blue_min + 1) if self.blue_count > 0 and self.blue_max >= self.blue_min else 0

        # 触发跟随矩阵: trigger_matrix[X][Y] = P(Y|X) X出现后Y下一期出现的条件概率
        self.red_trigger_matrix = None  # (red_total, red_total)
        self.blue_trigger_matrix = None
        self.is_trained = False

        # Wolfpacks: 高频2码组合 (简化版，只做2码)
        self.red_wolfpack_2 = None  # (red_total, red_total) 共现次数矩阵
        self.blue_wolfpack_2 = None

        # 平滑参数
        self.laplace_alpha = 0.1
        # 触发器权重 vs 基础频率权重
        self.trigger_weight = 0.6
        self.base_weight = 0.4
        # Wolfpack 权重
        self.wolfpack_weight = 0.3

    def _build_trigger_matrix(self, history: List[Dict], color: str = "red") -> Tuple[np.ndarray, np.ndarray]:
        """构建触发跟随矩阵和 Wolfpack 矩阵"""
        if color == "red":
            total = self.red_total
            min_val = self.red_min
        else:
            total = self.blue_total
            min_val = self.blue_min

        if total == 0:
            return np.array([]), np.array([])

        # 拉普拉斯平滑初始化
        trigger_count = np.ones((total, total)) * self.laplace_alpha
        x_count = np.ones(total) * self.laplace_alpha * total  # 每个X的出现次数（平滑）
        wolfpack_count = np.ones((total, total)) * self.laplace_alpha

        for i in range(1, len(history)):
            prev_nums = history[i - 1]["reds"] if color == "red" else history[i - 1]["blues"]
            curr_nums = history[i]["reds"] if color == "red" else history[i]["blues"]

            if not prev_nums or not curr_nums:
                continue

            # 去重 (非重复彩种)
            prev_set = set(prev_nums)
            curr_set = set(curr_nums)

            # 更新触发跟随: prev 中的每个 X -> curr 中的每个 Y
            for x in prev_set:
                xi = x - min_val
                if 0 <= xi < total:
                    x_count[xi] += 1
                    for y in curr_set:
                        yi = y - min_val
                        if 0 <= yi < total:
                            trigger_count[xi][yi] += 1

            # 更新 Wolfpack 2码共现 (同一期内一起出现的号码对)
            curr_list = list(curr_set)
            for a_idx in range(len(curr_list)):
                for b_idx in range(a_idx + 1, len(curr_list)):
                    ai = curr_list[a_idx] - min_val
                    bi = curr_list[b_idx] - min_val
                    if 0 <= ai < total and 0 <= bi < total:
                        wolfpack_count[ai][bi] += 1
                        wolfpack_count[bi][ai] += 1

        # 条件概率 P(Y|X) = count(X->Y) / count(X)
        trigger_matrix = np.zeros((total, total))
        for x in range(total):
            if x_count[x] > 0:
                trigger_matrix[x] = trigger_count[x] / x_count[x]

        return trigger_matrix, wolfpack_count

    def train(self, history: List[Dict]) -> Dict:
        """训练触发跟随模型"""
        if len(history) < 20:
            return {"success": False, "error": "数据量不足"}

        self.red_trigger_matrix, self.red_wolfpack_2 = self._build_trigger_matrix(history, "red")
        self.blue_trigger_matrix, self.blue_wolfpack_2 = self._build_trigger_matrix(history, "blue")
        self.is_trained = True

        metrics = {}
        if len(self.red_trigger_matrix) > 0:
            metrics["red_avg_trigger_prob"] = round(float(self.red_trigger_matrix.mean()), 4)
            metrics["red_max_trigger_prob"] = round(float(self.red_trigger_matrix.max()), 4)
        if len(self.blue_trigger_matrix) > 0:
            metrics["blue_avg_trigger_prob"] = round(float(self.blue_trigger_matrix.mean()), 4)

        return {"success": True, "metrics": metrics}

    def _score_numbers(self, history: List[Dict], color: str = "red") -> Tuple[List[float], List[int]]:
        """基于触发器+Wolfpack+基础频率计算每个号码的得分"""
        if color == "red":
            trigger_matrix = self.red_trigger_matrix
            wolfpack = self.red_wolfpack_2
            total = self.red_total
            min_val = self.red_min
            count = self.red_count
        else:
            trigger_matrix = self.blue_trigger_matrix
            wolfpack = self.blue_wolfpack_2
            total = self.blue_total
            min_val = self.blue_min
            count = self.blue_count

        if total == 0 or trigger_matrix is None or len(trigger_matrix) == 0:
            return [], []

        # 最近一期号码作为 triggers
        last_nums = history[-1]["reds"] if color == "red" else history[-1]["blues"]
        last_set = set(last_nums) if last_nums else set()

        # 1. 触发器得分: sum over triggers X of P(Y|X) * weight
        trigger_scores = np.zeros(total)
        for x in last_set:
            xi = x - min_val
            if 0 <= xi < total:
                trigger_scores += trigger_matrix[xi]
        if len(last_set) > 0:
            trigger_scores /= len(last_set)

        # 2. 基础频率得分 (最近30期频率)
        base_freq = np.zeros(total)
        recent = history[-30:] if len(history) >= 30 else history
        for h in recent:
            nums = h["reds"] if color == "red" else h["blues"]
            if not nums:
                continue
            for n in set(nums):
                ni = n - min_val
                if 0 <= ni < total:
                    base_freq[ni] += 1
        base_freq = base_freq / base_freq.sum() if base_freq.sum() > 0 else base_freq

        # 3. Wolfpack 得分: 如果上一期有号码，找与它们共现频率最高的号码
        wolfpack_scores = np.zeros(total)
        if last_set and wolfpack is not None and len(wolfpack) > 0:
            for x in last_set:
                xi = x - min_val
                if 0 <= xi < total:
                    wolfpack_scores += wolfpack[xi]
            if len(last_set) > 0:
                wolfpack_scores /= len(last_set)
            wolfpack_scores = wolfpack_scores / wolfpack_scores.sum() if wolfpack_scores.sum() > 0 else wolfpack_scores

        # 综合得分
        final_scores = (self.trigger_weight * trigger_scores
                       + self.base_weight * base_freq
                       + self.wolfpack_weight * wolfpack_scores)

        # 归一化
        if final_scores.sum() > 0:
            final_scores = final_scores / final_scores.sum()

        sorted_indices = np.argsort(-final_scores)
        sorted_scores = [float(final_scores[i]) for i in sorted_indices]
        sorted_numbers = [int(i + min_val) for i in sorted_indices]

        return sorted_scores, sorted_numbers

    def predict(self, history: List[Dict]) -> Tuple[List[int], List[int], Dict]:
        """预测下一期号码"""
        if not self.is_trained:
            self.train(history)

        info = {}

        # 红球
        red_scores, red_numbers = self._score_numbers(history, "red")
        red_pred = sorted(red_numbers[:self.red_count])
        info["red_top_probs"] = red_scores
        info["red_top_numbers"] = red_numbers[:10]

        # 蓝球
        blue_pred = []
        if self.blue_total > 0 and self.blue_count > 0:
            blue_scores, blue_numbers = self._score_numbers(history, "blue")
            blue_pred = sorted(blue_numbers[:self.blue_count])
            info["blue_top_probs"] = blue_scores
            info["blue_top_numbers"] = blue_numbers[:8]

        return red_pred, blue_pred, info

    def save(self, save_dir: str) -> bool:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "tf_model.pkl")
        try:
            with open(path, "wb") as f:
                pickle.dump({
                    "red_trigger_matrix": self.red_trigger_matrix,
                    "blue_trigger_matrix": self.blue_trigger_matrix,
                    "red_wolfpack_2": self.red_wolfpack_2,
                    "blue_wolfpack_2": self.blue_wolfpack_2,
                    "is_trained": self.is_trained,
                }, f)
            return True
        except Exception:
            return False

    def load(self, save_dir: str) -> bool:
        path = os.path.join(save_dir, "tf_model.pkl")
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.red_trigger_matrix = data["red_trigger_matrix"]
            self.blue_trigger_matrix = data["blue_trigger_matrix"]
            self.red_wolfpack_2 = data.get("red_wolfpack_2")
            self.blue_wolfpack_2 = data.get("blue_wolfpack_2")
            self.is_trained = data.get("is_trained", True)
            return True
        except Exception:
            return False
