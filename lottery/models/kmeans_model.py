"""
K-Means聚类彩票预测模型

思路: 把历史开奖号码作为高维向量聚类
- 找出K个号码聚集中心 (代表历史"热号模式")
- 用最近期号码找最近的聚类中心
- 从该中心附近+该类样本频率预测下一期
"""
import os
import pickle
import numpy as np
from typing import List, Dict, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class KMeansPredictor:
    """基于K-Means聚类的彩票预测器"""

    def __init__(self, config: Dict):
        self.config = config
        self.red_min, self.red_max = config["red_range"]
        self.blue_min, self.blue_max = config["blue_range"]
        self.red_count = config["red_count"]
        self.blue_count = config["blue_count"]
        self.red_total = self.red_max - self.red_min + 1
        self.blue_total = (self.blue_max - self.blue_min + 1) if self.blue_count > 0 and self.blue_max >= self.blue_min else 0

        self.n_clusters = 8
        self.kmeans = None
        self.cluster_centers = None
        self.cluster_red_freq = []  # 每个聚类的红球频率
        self.cluster_blue_freq = []
        self.scaler = StandardScaler()
        self.is_trained = False

    def _to_feature(self, rec: Dict) -> List[int]:
        """将一期开奖转为one-hot特征向量"""
        feat = [0] * self.red_total
        for n in rec["reds"]:
            if self.red_min <= n <= self.red_max:
                feat[n - self.red_min] = 1
        if self.blue_count > 0:
            feat_blues = [0] * self.blue_total
            for b in rec["blues"]:
                if self.blue_min <= b <= self.blue_max:
                    feat_blues[b - self.blue_min] = 1
            feat.extend(feat_blues)
        return feat

    def train(self, history: List[Dict]) -> Dict:
        if len(history) < 30:
            return {"success": False, "error": "数据量不足"}

        # 构建所有样本特征
        X = np.array([self._to_feature(rec) for rec in history], dtype=np.float32)

        # 聚类
        n_clust = min(self.n_clusters, len(X) // 5)
        n_clust = max(2, n_clust)
        self.kmeans = KMeans(n_clusters=n_clust, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(X)
        self.cluster_centers = self.kmeans.cluster_centers_

        # 每个聚类的红球频率
        self.cluster_red_freq = []
        self.cluster_blue_freq = []
        for c in range(n_clust):
            mask = (labels == c)
            cluster_recs = [history[i] for i in range(len(history)) if mask[i]]
            red_freq = np.zeros(self.red_total)
            for rec in cluster_recs:
                for n in rec["reds"]:
                    if self.red_min <= n <= self.red_max:
                        red_freq[n - self.red_min] += 1
            if red_freq.sum() > 0:
                red_freq = red_freq / red_freq.sum()
            else:
                red_freq = np.ones(self.red_total) / self.red_total
            self.cluster_red_freq.append(red_freq)

            if self.blue_count > 0:
                blue_freq = np.zeros(self.blue_total)
                for rec in cluster_recs:
                    for b in rec["blues"]:
                        if self.blue_min <= b <= self.blue_max:
                            blue_freq[b - self.blue_min] += 1
                if blue_freq.sum() > 0:
                    blue_freq = blue_freq / blue_freq.sum()
                else:
                    blue_freq = np.ones(self.blue_total) / self.blue_total
                self.cluster_blue_freq.append(blue_freq)

        self.is_trained = True
        # 各聚类大小
        cluster_sizes = [int((labels == c).sum()) for c in range(n_clust)]
        return {"success": True, "samples": len(X),
                "metrics": {"n_clusters": n_clust, "cluster_sizes": cluster_sizes}}

    def predict(self, history: List[Dict]) -> Tuple[List[int], List[int], Dict]:
        if not self.is_trained or not history:
            return [], [], {}

        # 用最近一期找聚类
        last_feat = np.array([self._to_feature(history[-1])], dtype=np.float32)
        cluster = int(self.kmeans.predict(last_feat)[0])
        red_freq = self.cluster_red_freq[cluster]
        blue_freq = self.cluster_blue_freq[cluster] if self.blue_count > 0 and cluster < len(self.cluster_blue_freq) else np.zeros(1)

        is_repeatable = (self.red_max - self.red_min + 1) <= 10 and self.red_count >= 3

        # 综合聚类中心 + 频率
        # 直接用频率选号
        if is_repeatable:
            top_indices = np.argsort(red_freq)[-self.red_count:][::-1]
            reds = [int(idx) + self.red_min for idx in top_indices]
        else:
            top_indices = np.argsort(red_freq)[-self.red_count:][::-1]
            reds = sorted([int(idx) + self.red_min for idx in top_indices])

        blues = []
        if self.blue_count > 0:
            top_blue = np.argsort(blue_freq)[-self.blue_count:][::-1]
            blues = sorted([int(idx) + self.blue_min for idx in top_blue])

        info = {
            "cluster": cluster,
            "red_top_probs": [(int(idx) + self.red_min, round(float(red_freq[idx]), 4))
                              for idx in np.argsort(red_freq)[-10:][::-1]],
        }
        if self.blue_count > 0:
            info["blue_top_probs"] = [(int(idx) + self.blue_min, round(float(blue_freq[idx]), 4))
                                      for idx in np.argsort(blue_freq)[-5:][::-1]]
        return reds, blues, info

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "kmeans_model.pkl"), "wb") as f:
            pickle.dump({
                "kmeans": self.kmeans,
                "cluster_centers": self.cluster_centers,
                "cluster_red_freq": self.cluster_red_freq,
                "cluster_blue_freq": self.cluster_blue_freq,
                "is_trained": self.is_trained,
                "n_clusters": self.n_clusters,
            }, f)

    def load(self, model_dir: str) -> bool:
        path = os.path.join(model_dir, "kmeans_model.pkl")
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.kmeans = data["kmeans"]
        self.cluster_centers = data["cluster_centers"]
        self.cluster_red_freq = data["cluster_red_freq"]
        self.cluster_blue_freq = data["cluster_blue_freq"]
        self.is_trained = data["is_trained"]
        self.n_clusters = data["n_clusters"]
        return True
