from typing import List, Dict, Tuple
from collections import Counter
from ..data.processor import DataProcessor


class FrequencyAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.processor = DataProcessor(config)
        self.red_min, self.red_max = config["red_range"]
        self.blue_min, self.blue_max = config["blue_range"]
        self.red_count = config["red_count"]
        self.blue_count = config["blue_count"]

    def count_all(self, history: List[Dict]) -> Dict[str, Counter]:
        red_counter = Counter()
        blue_counter = Counter()

        for record in history:
            red_counter.update(record["reds"])
            blue_counter.update(record["blues"])

        for n in range(self.red_min, self.red_max + 1):
            red_counter.setdefault(n, 0)
        for n in range(self.blue_min, self.blue_max + 1):
            blue_counter.setdefault(n, 0)

        return {"red": red_counter, "blue": blue_counter}

    def get_hot_numbers(self, history: List[Dict], recent_n: int = 50, top_n: int = 10) -> Dict[str, List[Tuple[int, int]]]:
        recent = history[-recent_n:] if len(history) >= recent_n else history
        counts = self.count_all(recent)
        red_sorted = sorted(counts["red"].items(), key=lambda x: x[1], reverse=True)[:top_n]
        blue_sorted = sorted(counts["blue"].items(), key=lambda x: x[1], reverse=True)[:top_n]
        return {"red": red_sorted, "blue": blue_sorted}

    def get_cold_numbers(self, history: List[Dict], recent_n: int = 50, bottom_n: int = 10) -> Dict[str, List[Tuple[int, int]]]:
        recent = history[-recent_n:] if len(history) >= recent_n else history
        counts = self.count_all(recent)
        red_sorted = sorted(counts["red"].items(), key=lambda x: x[1])[:bottom_n]
        blue_sorted = sorted(counts["blue"].items(), key=lambda x: x[1])[:bottom_n]
        return {"red": red_sorted, "blue": blue_sorted}

    def get_overdue_numbers(self, history: List[Dict]) -> Dict[str, List[Tuple[int, int]]]:
        missing = self.processor.get_missing_values(history)
        red_sorted = sorted(missing["red"].items(), key=lambda x: x[1], reverse=True)
        blue_sorted = sorted(missing["blue"].items(), key=lambda x: x[1], reverse=True)
        return {"red": red_sorted, "blue": blue_sorted}

    def get_frequency_ratio(self, history: List[Dict], recent_n: int = 100) -> Dict[str, Dict[int, float]]:
        all_counts = self.count_all(history)
        recent = history[-recent_n:] if len(history) >= recent_n else history
        recent_counts = self.count_all(recent)

        total_draws = len(history)
        recent_draws = len(recent)

        red_ratio = {}
        blue_ratio = {}

        for n in range(self.red_min, self.red_max + 1):
            avg = all_counts["red"][n] / total_draws if total_draws > 0 else 0
            recent_avg = recent_counts["red"][n] / recent_draws if recent_draws > 0 else 0
            red_ratio[n] = recent_avg / avg if avg > 0 else 1.0

        for n in range(self.blue_min, self.blue_max + 1):
            avg = all_counts["blue"][n] / total_draws if total_draws > 0 else 0
            recent_avg = recent_counts["blue"][n] / recent_draws if recent_draws > 0 else 0
            blue_ratio[n] = recent_avg / avg if avg > 0 else 1.0

        return {"red": red_ratio, "blue": blue_ratio}
