"""
LotteryAi 思想 - Ghost Variances 幽灵方差预测模型

思路来源: LotteryAi (CorvusCodex/LotteryAi)
- Ghost Variances: 识别波动性异常的号码
- Death Zones: 统计上长期低表现的号码区间
- Skip Velocity Rhythm: 号码返回间隔的加速度

核心思想:
1. 计算每个号码出现间隔的标准差 (波动性)
2. 与历史基线对比，方差显著偏离的号码标记为"幽灵号码"
3. 高波动号码可能即将爆发或沉寂，是预测的重要信号
4. 结合 Skip Velocity (间隔加速度) 判断趋势方向

实现方式:
- 计算每个号码的出现间隔序列
- 计算间隔均值 (基线)、标准差 (波动性)
- 计算最近间隔 vs 历史平均的偏离度 (Z-score)
- 计算 Skip Velocity: 最近3个间隔的变化趋势 (加速/减速)
- 综合评分: 即将回补的号码（间隔过长+减速）得分高
"""
import os
import pickle
import numpy as np
from typing import List, Dict, Tuple


class GhostVariancesPredictor:
    """LotteryAi 幽灵方差预测器"""

    def __init__(self, config: Dict):
        self.config = config
        self.red_min, self.red_max = config["red_range"]
        self.blue_min, self.blue_max = config["blue_range"]
        self.red_count = config["red_count"]
        self.blue_count = config["blue_count"]
        self.red_total = self.red_max - self.red_min + 1
        self.blue_total = (self.blue_max - self.blue_min + 1) if self.blue_count > 0 and self.blue_max >= self.blue_min else 0

        # 每个号码的统计特征
        self.red_stats = None  # dict: {number_idx: {mean_gap, std_gap, last_gap, z_score, velocity, ...}}
        self.blue_stats = None
        self.is_trained = False

        # 参数
        self.z_score_threshold = 1.5  # Z-score 阈值，超过视为异常
        self.velocity_weight = 0.3   # 间隔加速度权重
        self.frequency_weight = 0.4  # 基础频率权重
        self.anomaly_weight = 0.3    # 异常检测权重

    def _calc_gap_stats(self, history: List[Dict], color: str = "red") -> Dict[int, Dict]:
        """计算每个号码的间隔统计特征

        Returns:
            dict: {number_idx: stats_dict}
        """
        if color == "red":
            total = self.red_total
            min_val = self.red_min
        else:
            total = self.blue_total
            min_val = self.blue_min

        if total == 0:
            return {}

        # 每个号码出现的期数索引 (从0开始)
        appearances = {i: [] for i in range(total)}

        for period_idx, h in enumerate(history):
            nums = h["reds"] if color == "red" else h["blues"]
            if not nums:
                continue
            for n in set(nums):
                ni = n - min_val
                if 0 <= ni < total:
                    appearances[ni].append(period_idx)

        stats = {}
        for ni in range(total):
            app = appearances[ni]
            if len(app) < 3:
                # 出现太少，给默认值
                stats[ni] = {
                    "mean_gap": len(history) / max(len(app), 1),
                    "std_gap": len(history) / max(len(app), 1),
                    "last_gap": len(history),
                    "z_score": 0.0,
                    "velocity": 0.0,
                    "frequency": len(app) / len(history),
                    "count": len(app),
                }
                continue

            # 间隔序列
            gaps = np.diff(app)
            mean_gap = float(np.mean(gaps))
            std_gap = float(np.std(gaps))

            # 距上次出现的间隔 (当前遗漏)
            last_gap = len(history) - 1 - app[-1]

            # Z-score: 当前遗漏相对于历史的偏离程度
            # z > 0 表示遗漏大于平均（冷号），z < 0 表示遗漏小于平均（热号）
            z_score = (last_gap - mean_gap) / max(std_gap, 0.1)

            # Skip Velocity: 最近3个间隔的变化趋势
            # 正: 间隔在变大（减速远离）；负: 间隔在变小（加速回归）
            if len(gaps) >= 3:
                recent_3 = gaps[-3:]
                # 线性回归斜率
                x = np.arange(3)
                velocity = float(np.polyfit(x, recent_3, 1)[0])
            else:
                velocity = 0.0

            stats[ni] = {
                "mean_gap": mean_gap,
                "std_gap": std_gap,
                "last_gap": last_gap,
                "z_score": z_score,
                "velocity": velocity,
                "frequency": len(app) / len(history),
                "count": len(app),
            }

        return stats

    def _score_from_stats(self, stats: Dict[int, Dict], total: int) -> Tuple[List[float], List[int]]:
        """根据统计特征计算每个号码的预测得分

        得分逻辑:
        - 频率高 (热号) + 分
        - Z-score 高但 velocity 负 (冷号即将回补) + 分
        - 波动性适中 (既不过于规律也不过于随机) + 分
        """
        scores = np.zeros(total)

        for ni in range(total):
            s = stats[ni]

            # 1. 基础频率得分
            freq_score = s["frequency"] * self.frequency_weight

            # 2. 异常检测得分 (Ghost Variances)
            # 如果号码长期未出 (z_score > threshold) 且间隔在加速回补 (velocity < 0)
            # 说明可能即将开出，给高分
            if s["z_score"] > self.z_score_threshold and s["velocity"] < 0:
                anomaly_score = abs(s["z_score"]) * self.anomaly_weight * 0.5
            elif s["z_score"] < -self.z_score_threshold:
                # 刚开出不久，短期可能不回
                anomaly_score = -abs(s["z_score"]) * self.anomaly_weight * 0.3
            else:
                anomaly_score = 0.0

            # 3. Velocity 得分
            # velocity 为负（间隔缩小）说明有回归趋势
            velocity_score = -s["velocity"] * self.velocity_weight * 0.1
            if velocity_score > 0:
                velocity_score = min(velocity_score, 0.1)

            scores[ni] = freq_score + anomaly_score + velocity_score

        # 归一化到 [0, 1]
        s_min = scores.min()
        s_max = scores.max()
        if s_max - s_min > 1e-8:
            scores = (scores - s_min) / (s_max - s_min)
        else:
            scores = np.ones(total) / total

        sorted_indices = np.argsort(-scores)
        sorted_scores = [float(scores[i]) for i in sorted_indices]
        sorted_numbers = [int(i) for i in sorted_indices]

        return sorted_scores, sorted_numbers

    def train(self, history: List[Dict]) -> Dict:
        """训练幽灵方差模型"""
        if len(history) < 50:
            return {"success": False, "error": "数据量不足"}

        self.red_stats = self._calc_gap_stats(history, "red")
        self.blue_stats = self._calc_gap_stats(history, "blue")
        self.is_trained = True

        metrics = {}
        if self.red_stats:
            high_z = [i + self.red_min for i, s in self.red_stats.items()
                     if s["z_score"] > self.z_score_threshold]
            low_z = [i + self.red_min for i, s in self.red_stats.items()
                    if s["z_score"] < -self.z_score_threshold]
            metrics["red_ghost_count"] = len(high_z)
            metrics["red_ghost_numbers"] = high_z[:10]
            metrics["red_hot_count"] = len(low_z)
            metrics["red_hot_numbers"] = low_z[:10]

        return {"success": True, "metrics": metrics}

    def predict(self, history: List[Dict]) -> Tuple[List[int], List[int], Dict]:
        """预测下一期号码"""
        if not self.is_trained:
            self.train(history)

        info = {}

        # 红球
        if self.red_stats and self.red_total > 0:
            red_scores, red_indices = self._score_from_stats(self.red_stats, self.red_total)
            red_pred = sorted([int(i + self.red_min) for i in red_indices[:self.red_count]])
            info["red_top_probs"] = red_scores
            info["red_top_numbers"] = [int(i + self.red_min) for i in red_indices[:10]]
            info["red_ghost_numbers"] = [
                int(i + self.red_min) for i, s in self.red_stats.items()
                if s["z_score"] > self.z_score_threshold
            ]
        else:
            red_pred = []

        # 蓝球
        blue_pred = []
        if self.blue_stats and self.blue_total > 0 and self.blue_count > 0:
            blue_scores, blue_indices = self._score_from_stats(self.blue_stats, self.blue_total)
            blue_pred = sorted([int(i + self.blue_min) for i in blue_indices[:self.blue_count]])
            info["blue_top_probs"] = blue_scores
            info["blue_top_numbers"] = [int(i + self.blue_min) for i in blue_indices[:8]]

        return red_pred, blue_pred, info

    def save(self, save_dir: str) -> bool:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "gv_model.pkl")
        try:
            with open(path, "wb") as f:
                pickle.dump({
                    "red_stats": self.red_stats,
                    "blue_stats": self.blue_stats,
                    "is_trained": self.is_trained,
                }, f)
            return True
        except Exception:
            return False

    def load(self, save_dir: str) -> bool:
        path = os.path.join(save_dir, "gv_model.pkl")
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.red_stats = data["red_stats"]
            self.blue_stats = data["blue_stats"]
            self.is_trained = data.get("is_trained", True)
            return True
        except Exception:
            return False
