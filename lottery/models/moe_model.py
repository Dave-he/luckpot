"""
LotteryAi 思想 - Mixture of Experts (MoE) 动态路由预测模型

思路来源: LotteryAi (CorvusCodex/LotteryAi)
- Mixture of Experts: 动态路由输入到最相关专家模型
- 9 种神经网络架构集成，通过 Gate Network 动态加权
- 核心思想: 不同的市场状态（号码分布模式）下，不同专家的表现不同

简化实现:
- 专家模型 = 本项目已有的各种预测器 (热号/冷号/马尔可夫/近期加权/...)
- Gating Network = 根据当前数据特征（和值、跨度、奇偶比等）计算各专家权重
- 不是静态权重，而是根据"当前号码画像"动态调整

实现方式:
1. 提取当前期的特征画像: 和值、跨度、奇偶比、大小比、三区比、AC值
2. 滑窗遍历历史，找到特征相似的历史时期
3. 统计各专家在相似时期的历史表现
4. 用表现加权作为当前各专家的动态权重
5. 加权融合各专家预测
"""
import os
import pickle
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


class MoEPredictor:
    """LotteryAi MoE 动态路由预测器（简化版）"""

    def __init__(self, config: Dict):
        self.config = config
        self.red_min, self.red_max = config["red_range"]
        self.blue_min, self.blue_max = config["blue_range"]
        self.red_count = config["red_count"]
        self.blue_count = config["blue_count"]
        self.red_total = self.red_max - self.red_min + 1
        self.blue_total = (self.blue_max - self.blue_min + 1) if self.blue_count > 0 and self.blue_max >= self.blue_min else 0

        self.is_trained = False
        self.expert_names = [
            "hot", "cold", "balanced",
            "markov_like", "recency_weighted",
            "triggers_followers", "ghost_variances",
        ]

        # 各专家历史表现存储
        self.expert_history = None  # {expert_name: [list of hit_rates]}
        self.expert_weights = None  # 当前动态权重

    def _extract_features(self, reds: List[int]) -> np.ndarray:
        """提取一期红球的特征画像

        Returns:
            特征向量: [和值, 跨度, 奇数个数, 偶数个数, 小号个数, 大号个数,
                      一区个数, 二区个数, 三区个数, 连号对数, AC值近似]
        """
        if not reds:
            return np.zeros(11)

        nums = sorted(reds)
        n = len(nums)

        # 和值
        sum_val = sum(nums)

        # 跨度
        span = nums[-1] - nums[0] if n > 1 else 0

        # 奇偶比
        odd_count = sum(1 for x in nums if x % 2 == 1)
        even_count = n - odd_count

        # 大小比 (以中间值为界)
        mid = (self.red_min + self.red_max) / 2
        small_count = sum(1 for x in nums if x < mid)
        big_count = n - small_count

        # 三区间分布
        range_size = (self.red_max - self.red_min + 1) / 3
        zone1 = sum(1 for x in nums if x < self.red_min + range_size)
        zone2 = sum(1 for x in nums if self.red_min + range_size <= x < self.red_min + 2 * range_size)
        zone3 = n - zone1 - zone2

        # 连号对数
        consec = 0
        for i in range(1, n):
            if nums[i] - nums[i - 1] == 1:
                consec += 1

        # AC 值近似 (两两差值去重数 - n + 1)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(nums[j] - nums[i])
        ac_approx = len(diffs) - n + 1

        return np.array([sum_val, span, odd_count, even_count,
                         small_count, big_count, zone1, zone2, zone3,
                         consec, ac_approx], dtype=float)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """余弦相似度"""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _expert_predict(self, expert_name: str, history: List[Dict],
                       red_total: int, red_min: int, red_count: int,
                       blue_total: int, blue_min: int, blue_count: int,
                       is_repeatable: bool):
        """各专家的预测逻辑（简化版，不依赖外部模型）"""

        recent_30 = history[-30:] if len(history) >= 30 else history
        recent_10 = history[-10:] if len(history) >= 10 else history

        # 统计频率
        freq = np.zeros(red_total)
        for h in recent_30:
            for n in set(h["reds"]):
                ni = n - red_min
                if 0 <= ni < red_total:
                    freq[ni] += 1

        if expert_name == "hot":
            # 热号: 最近10期高频
            hot_freq = np.zeros(red_total)
            for h in recent_10:
                for n in set(h["reds"]):
                    ni = n - red_min
                    if 0 <= ni < red_total:
                        hot_freq[ni] += 1
            return hot_freq

        elif expert_name == "cold":
            # 冷号回补: 最近10期低频 + 总频率不太低
            cold_score = np.zeros(red_total)
            recent_freq = np.zeros(red_total)
            for h in recent_10:
                for n in set(h["reds"]):
                    ni = n - red_min
                    if 0 <= ni < red_total:
                        recent_freq[ni] += 1
            for i in range(red_total):
                if freq[i] > 0:
                    cold_score[i] = (freq[i] / 30) * (1.0 - recent_freq[i] / 10)
            return cold_score

        elif expert_name == "balanced":
            # 均衡分布: 考虑和值、奇偶、大小均衡
            base = freq / freq.sum() if freq.sum() > 0 else np.ones(red_total) / red_total
            balanced = base.copy()
            mid_idx = red_total // 2
            # 倾向于中区间号码
            for i in range(red_total):
                distance_from_mid = abs(i - mid_idx) / mid_idx
                balanced[i] *= (1.0 - distance_from_mid * 0.3)
            return balanced

        elif expert_name == "markov_like":
            # 简化版马尔可夫: 上一期号码邻居加权
            last_reds = history[-1]["reds"] if history else []
            markov_score = np.zeros(red_total)
            for n in set(last_reds):
                ni = n - red_min
                if 0 <= ni < red_total:
                    # 邻居号码有更高概率
                    for j in range(max(0, ni - 3), min(red_total, ni + 4)):
                        dist = abs(j - ni)
                        markov_score[j] += 1.0 / (dist + 1)
            return markov_score

        elif expert_name == "recency_weighted":
            # 近期加权: 越近权重越高
            rw_score = np.zeros(red_total)
            n = len(recent_30)
            for i, h in enumerate(recent_30):
                weight = np.exp(-(n - 1 - i) / 15)  # 指数衰减
                for num in set(h["reds"]):
                    ni = num - red_min
                    if 0 <= ni < red_total:
                        rw_score[ni] += weight
            return rw_score

        elif expert_name == "triggers_followers":
            # 触发跟随: 上一期号码触发的跟随号码
            last_reds = history[-1]["reds"] if history else []
            tf_score = np.zeros(red_total)
            # 用完整历史计算条件概率
            if len(history) >= 20:
                trigger_count = defaultdict(lambda: defaultdict(int))
                x_count = defaultdict(int)
                for i in range(1, len(history)):
                    prev_set = set(history[i - 1]["reds"])
                    curr_set = set(history[i]["reds"])
                    for x in prev_set:
                        xi = x - red_min
                        if 0 <= xi < red_total:
                            x_count[xi] += 1
                            for y in curr_set:
                                yi = y - red_min
                                if 0 <= yi < red_total:
                                    trigger_count[xi][yi] += 1
                for x in set(last_reds):
                    xi = x - red_min
                    if 0 <= xi < red_total and xi in x_count:
                        for yi in range(red_total):
                            if yi in trigger_count[xi]:
                                tf_score[yi] += trigger_count[xi][yi] / x_count[xi]
            return tf_score

        elif expert_name == "ghost_variances":
            # 幽灵方差: 高遗漏+回归趋势的号码
            gv_score = np.zeros(red_total)
            # 计算每个号码的遗漏和趋势
            last_appear = {}
            gaps_list = defaultdict(list)
            prev_period = {i: -1 for i in range(red_total)}
            for period_idx, h in enumerate(history):
                for n in set(h["reds"]):
                    ni = n - red_min
                    if 0 <= ni < red_total:
                        if prev_period[ni] >= 0:
                            gaps_list[ni].append(period_idx - prev_period[ni])
                        prev_period[ni] = period_idx
                        last_appear[ni] = period_idx

            for i in range(red_total):
                last_gap = len(history) - 1 - last_appear.get(i, -1)
                gaps = gaps_list.get(i, [])
                if gaps:
                    mean_gap = np.mean(gaps)
                    std_gap = np.std(gaps)
                    z = (last_gap - mean_gap) / max(std_gap, 0.5)
                    # 高遗漏 + 近期可能回补
                    if z > 1.0:
                        gv_score[i] = z * 0.5
                    else:
                        gv_score[i] = 1.0 / (mean_gap + 1)
                else:
                    gv_score[i] = 0.01
            return gv_score

        else:
            return np.ones(red_total) / red_total

    def _compute_dynamic_weights(self, history: List[Dict], n_similar: int = 50) -> Dict[str, float]:
        """计算当前状态下各专家的动态权重

        方法: 找到与当前特征最相似的 n_similar 个历史时期，
             统计各专家在那些时期的表现，表现好的权重高
        """
        if len(history) < 30:
            # 数据不足，用均匀权重
            return {name: 1.0 / len(self.expert_names) for name in self.expert_names}

        # 当前特征画像
        current_reds = history[-1]["reds"]
        current_feat = self._extract_features(current_reds)

        # 计算每个历史时期与当前的相似度
        similarities = []
        for i in range(len(history) - 1):  # 不包含最后一期（当前期）
            feat = self._extract_features(history[i]["reds"])
            sim = self._cosine_similarity(current_feat, feat)
            similarities.append((i, sim))

        # 取最相似的 n_similar 个时期
        similarities.sort(key=lambda x: -x[1])
        top_indices = [idx for idx, _ in similarities[:n_similar]]

        # 对每个专家，计算在相似时期的平均表现
        expert_scores = {}
        red_min = self.red_min
        red_total = self.red_total

        for expert_name in self.expert_names:
            total_hit = 0
            total_count = 0
            for idx in top_indices:
                if idx + 1 >= len(history):
                    continue
                # 用 idx 之前的数据预测 idx+1
                train_hist = history[:idx + 1]
                actual_reds = set(history[idx + 1]["reds"])
                # 专家预测得分
                scores = self._expert_predict(
                    expert_name, train_hist, red_total, red_min, self.red_count,
                    self.blue_total, self.blue_min, self.blue_count, False
                )
                # 取 top red_count 个号码
                top_indices_pred = np.argsort(-scores)[:self.red_count]
                pred_set = set(int(i + red_min) for i in top_indices_pred)
                hits = len(pred_set & actual_reds)
                total_hit += hits
                total_count += self.red_count

            expert_scores[expert_name] = total_hit / max(total_count, 1)

        # Softmax 归一化为权重
        scores_arr = np.array([expert_scores.get(n, 0.0) for n in self.expert_names])
        scores_arr = scores_arr - scores_arr.max()  # 数值稳定
        exp_scores = np.exp(scores_arr / 0.02)  # temperature = 0.02
        weights = exp_scores / exp_scores.sum()

        return {name: float(w) for name, w in zip(self.expert_names, weights)}

    def train(self, history: List[Dict]) -> Dict:
        """训练 MoE 模型（计算初始动态权重）"""
        if len(history) < 30:
            return {"success": False, "error": "数据量不足"}

        self.expert_weights = self._compute_dynamic_weights(history)
        self.is_trained = True

        # 计算各专家基础频率（加速预测）
        self._base_freq = np.zeros(self.red_total)
        for h in history[-30:]:
            for n in set(h["reds"]):
                ni = n - self.red_min
                if 0 <= ni < self.red_total:
                    self._base_freq[ni] += 1

        metrics = {
            "expert_weights": {k: round(v, 4) for k, v in self.expert_weights.items()},
            "top_expert": max(self.expert_weights, key=self.expert_weights.get),
        }

        return {"success": True, "metrics": metrics}

    def predict(self, history: List[Dict]) -> Tuple[List[int], List[int], Dict]:
        """预测下一期号码"""
        if not self.is_trained:
            self.train(history)

        info = {}

        # 动态计算当前权重 (每次预测都重新计算，因为最新一期可能改变相似历史)
        dynamic_weights = self._compute_dynamic_weights(history, n_similar=50)
        info["dynamic_weights"] = {k: round(v, 4) for k, v in dynamic_weights.items()}
        info["top_expert"] = max(dynamic_weights, key=dynamic_weights.get)

        # 红球: 各专家加权融合
        red_final = np.zeros(self.red_total)
        for expert_name, weight in dynamic_weights.items():
            scores = self._expert_predict(
                expert_name, history, self.red_total, self.red_min, self.red_count,
                self.blue_total, self.blue_min, self.blue_count, False
            )
            if scores.sum() > 0:
                scores = scores / scores.sum()
            red_final += scores * weight

        red_sorted_indices = np.argsort(-red_final)
        red_pred = sorted([int(i + self.red_min) for i in red_sorted_indices[:self.red_count]])
        info["red_top_probs"] = [float(red_final[i]) for i in red_sorted_indices]
        info["red_top_numbers"] = [int(i + self.red_min) for i in red_sorted_indices[:10]]

        # 蓝球: 简单频率 (MoE 主要用于红球)
        blue_pred = []
        if self.blue_total > 0 and self.blue_count > 0:
            blue_freq = np.zeros(self.blue_total)
            for h in history[-30:]:
                for n in set(h.get("blues", [])):
                    ni = n - self.blue_min
                    if 0 <= ni < self.blue_total:
                        blue_freq[ni] += 1
            blue_sorted = np.argsort(-blue_freq)
            blue_pred = sorted([int(i + self.blue_min) for i in blue_sorted[:self.blue_count]])
            info["blue_top_probs"] = [float(blue_freq[i] / blue_freq.sum() if blue_freq.sum() > 0 else 0)
                                       for i in blue_sorted]
            info["blue_top_numbers"] = [int(i + self.blue_min) for i in blue_sorted[:8]]

        return red_pred, blue_pred, info

    def save(self, save_dir: str) -> bool:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "moe_model.pkl")
        try:
            with open(path, "wb") as f:
                pickle.dump({
                    "expert_weights": self.expert_weights,
                    "expert_names": self.expert_names,
                    "is_trained": self.is_trained,
                }, f)
            return True
        except Exception:
            return False

    def load(self, save_dir: str) -> bool:
        path = os.path.join(save_dir, "moe_model.pkl")
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.expert_weights = data.get("expert_weights")
            self.expert_names = data.get("expert_names", self.expert_names)
            self.is_trained = data.get("is_trained", True)
            return True
        except Exception:
            return False
