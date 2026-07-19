import random
from typing import List, Dict, Tuple
from collections import defaultdict

from ..data.processor import DataProcessor
from ..analysis.frequency import FrequencyAnalyzer


class LotteryPredictor:
    def __init__(self, config: Dict):
        self.config = config
        self.processor = DataProcessor(config)
        self.freq_analyzer = FrequencyAnalyzer(config)
        self.red_min, self.red_max = config["red_range"]
        self.blue_min, self.blue_max = config["blue_range"]
        self.red_count = config["red_count"]
        self.blue_count = config["blue_count"]
        self.red_range = list(range(self.red_min, self.red_max + 1))
        self.blue_range = list(range(self.blue_min, self.blue_max + 1))

    def _score_numbers(self, history: List[Dict], method: str = "combined") -> Tuple[Dict[int, float], Dict[int, float]]:
        red_scores = {n: 0.0 for n in self.red_range}
        # 无蓝球彩种: blue_scores直接置空, 跳过所有蓝球相关计算
        blue_scores = {n: 0.0 for n in self.blue_range} if self.blue_count > 0 else {}

        if method in ("frequency", "combined"):
            hot = self.freq_analyzer.get_hot_numbers(history, recent_n=30, top_n=self.red_max)
            red_max_count = max((c for _, c in hot["red"]), default=1) or 1
            for n, c in hot["red"]:
                red_scores[n] += (c / red_max_count) * 1.0
            if self.blue_count > 0:
                blue_max_count = max((c for _, c in hot["blue"]), default=1) or 1
                for n, c in hot["blue"]:
                    blue_scores[n] += (c / blue_max_count) * 1.0

        if method in ("missing", "combined"):
            missing = self.processor.get_missing_values(history)
            red_max_miss = max(missing["red"].values(), default=1) or 1
            for n, m in missing["red"].items():
                red_scores[n] += (m / red_max_miss) * 0.8
            if self.blue_count > 0:
                blue_max_miss = max(missing["blue"].values(), default=1) or 1
                for n, m in missing["blue"].items():
                    blue_scores[n] += (m / blue_max_miss) * 0.8

        if method in ("ratio", "combined"):
            ratio = self.freq_analyzer.get_frequency_ratio(history, recent_n=50)
            for n, r in ratio["red"].items():
                if r > 1.2:
                    red_scores[n] += 0.5
                elif r < 0.8:
                    red_scores[n] += 0.3
            if self.blue_count > 0:
                for n, r in ratio["blue"].items():
                    if r > 1.2:
                        blue_scores[n] += 0.5
                    elif r < 0.8:
                        blue_scores[n] += 0.3

        if method in ("transition", "combined"):
            if len(history) >= 2:
                red_trans = defaultdict(lambda: defaultdict(int))
                blue_trans = defaultdict(lambda: defaultdict(int))
                for i in range(1, len(history)):
                    for prev in history[i - 1]["reds"]:
                        for curr in history[i]["reds"]:
                            red_trans[prev][curr] += 1
                    if self.blue_count > 0:
                        for prev in history[i - 1]["blues"]:
                            for curr in history[i]["blues"]:
                                blue_trans[prev][curr] += 1

                last_reds = history[-1]["reds"]
                last_blues = history[-1]["blues"]
                red_next_scores = defaultdict(float)
                blue_next_scores = defaultdict(float)
                for lr in last_reds:
                    for nxt, cnt in red_trans[lr].items():
                        total = sum(red_trans[lr].values())
                        if total > 0:
                            red_next_scores[nxt] += cnt / total
                if self.blue_count > 0:
                    for lb in last_blues:
                        for nxt, cnt in blue_trans[lb].items():
                            total = sum(blue_trans[lb].values())
                            if total > 0:
                                blue_next_scores[nxt] += cnt / total

                red_max_ts = max(red_next_scores.values(), default=1) or 1
                for n, s in red_next_scores.items():
                    if n in red_scores:
                        red_scores[n] += (s / red_max_ts) * 0.6
                if self.blue_count > 0:
                    blue_max_ts = max(blue_next_scores.values(), default=1) or 1
                    for n, s in blue_next_scores.items():
                        if n in blue_scores:
                            blue_scores[n] += (s / blue_max_ts) * 0.6

        return red_scores, blue_scores

    def predict_by_frequency(self, history: List[Dict]) -> Tuple[List[int], List[int]]:
        return self._select_by_scores(*self._score_numbers(history, "frequency"))

    def predict_by_missing(self, history: List[Dict]) -> Tuple[List[int], List[int]]:
        return self._select_by_scores(*self._score_numbers(history, "missing"))

    def predict_combined(self, history: List[Dict]) -> Tuple[List[int], List[int]]:
        return self._select_by_scores(*self._score_numbers(history, "combined"))

    def predict_random(self) -> Tuple[List[int], List[int]]:
        reds = sorted(random.sample(self.red_range, self.red_count))
        blues = sorted(random.sample(self.blue_range, self.blue_count)) if self.blue_count > 0 else []
        return reds, blues

    def predict_weighted_random(self, history: List[Dict], n_sets: int = 5) -> List[Tuple[List[int], List[int]]]:
        red_scores, blue_scores = self._score_numbers(history, "combined")
        results = []
        for _ in range(n_sets):
            reds = self._weighted_sample(red_scores, self.red_count)
            blues = self._weighted_sample(blue_scores, self.blue_count)
            results.append((sorted(reds), sorted(blues)))
        return results

    def _weighted_sample(self, scores: Dict[int, float], count: int) -> List[int]:
        numbers = list(scores.keys())
        weights = [max(scores[n], 0.01) for n in numbers]
        selected = []
        remaining_numbers = list(numbers)
        remaining_weights = list(weights)

        for _ in range(count):
            total = sum(remaining_weights)
            if total <= 0:
                pick = random.choice(remaining_numbers)
            else:
                probs = [w / total for w in remaining_weights]
                pick = random.choices(remaining_numbers, weights=probs, k=1)[0]
            idx = remaining_numbers.index(pick)
            selected.append(pick)
            remaining_numbers.pop(idx)
            remaining_weights.pop(idx)

        return selected

    def _select_by_scores(self, red_scores: Dict[int, float], blue_scores: Dict[int, float]) -> Tuple[List[int], List[int]]:
        reds = sorted(red_scores.items(), key=lambda x: x[1], reverse=True)
        blues = sorted(blue_scores.items(), key=lambda x: x[1], reverse=True)
        selected_reds = sorted([n for n, _ in reds[:self.red_count]])
        selected_blues = sorted([n for n, _ in blues[:self.blue_count]])
        return selected_reds, selected_blues

    def predict_multi_strategy(self, history: List[Dict]) -> Dict[str, Tuple[List[int], List[int]]]:
        return {
            "热号推荐": self.predict_by_frequency(history),
            "冷号回补": self.predict_by_missing(history),
            "综合推荐": self.predict_combined(history),
            "随机机选": self.predict_random(),
        }
