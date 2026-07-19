from typing import List, Dict, Tuple
import math
from collections import defaultdict


class StatisticsAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.red_min, self.red_max = config["red_range"]
        self.blue_min, self.blue_max = config["blue_range"]
        self.red_count = config["red_count"]
        self.blue_count = config["blue_count"]

    def sum_analysis(self, history: List[Dict]) -> Dict[str, Dict]:
        red_sums = []
        blue_sums = []
        for record in history:
            red_sums.append(sum(record["reds"]))
            blue_sums.append(sum(record["blues"]))

        def calc_stats(values: List[int]) -> Dict:
            if not values:
                return {}
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            mean = sum(values) / n
            variance = sum((x - mean) ** 2 for x in values) / n
            std = math.sqrt(variance)
            return {
                "mean": round(mean, 2),
                "std": round(std, 2),
                "min": min(values),
                "max": max(values),
                "median": sorted_vals[n // 2],
                "most_common": max(set(values), key=values.count),
            }

        return {
            "red_sum": calc_stats(red_sums),
            "blue_sum": calc_stats(blue_sums),
        }

    def odd_even_analysis(self, history: List[Dict]) -> Dict[str, Dict]:
        red_odd_even = defaultdict(int)
        blue_odd_even = defaultdict(int)

        for record in history:
            red_odd = sum(1 for n in record["reds"] if n % 2 == 1)
            red_even = self.red_count - red_odd
            red_odd_even[f"{red_odd}:{red_even}"] += 1

            blue_odd = sum(1 for n in record["blues"] if n % 2 == 1)
            blue_even = self.blue_count - blue_odd
            blue_odd_even[f"{blue_odd}:{blue_even}"] += 1

        total = len(history)
        return {
            "red": {k: {"count": v, "ratio": round(v / total * 100, 2)} for k, v in sorted(red_odd_even.items(), key=lambda x: x[1], reverse=True)},
            "blue": {k: {"count": v, "ratio": round(v / total * 100, 2)} for k, v in sorted(blue_odd_even.items(), key=lambda x: x[1], reverse=True)},
        }

    def region_analysis(self, history: List[Dict]) -> Dict[str, Dict]:
        red_regions = defaultdict(int)
        blue_regions = defaultdict(int)
        total = len(history)

        red_range = self.red_max - self.red_min + 1
        red_region_size = red_range // 3
        red_r1 = self.red_min + red_region_size
        red_r2 = red_r1 + red_region_size

        blue_range = self.blue_max - self.blue_min + 1
        blue_region_size = blue_range // 2
        blue_b = self.blue_min + blue_region_size

        for record in history:
            r1_count = sum(1 for n in record["reds"] if n <= red_r1)
            r2_count = sum(1 for n in record["reds"] if red_r1 < n <= red_r2)
            r3_count = self.red_count - r1_count - r2_count
            red_regions[f"{r1_count}:{r2_count}:{r3_count}"] += 1

            b1_count = sum(1 for n in record["blues"] if n <= blue_b)
            b2_count = self.blue_count - b1_count
            blue_regions[f"{b1_count}:{b2_count}"] += 1

        return {
            "red": {k: {"count": v, "ratio": round(v / total * 100, 2)} for k, v in sorted(red_regions.items(), key=lambda x: x[1], reverse=True)[:10]},
            "blue": {k: {"count": v, "ratio": round(v / total * 100, 2)} for k, v in sorted(blue_regions.items(), key=lambda x: x[1], reverse=True)},
        }

    def consecutive_numbers(self, history: List[Dict]) -> Dict[str, Dict]:
        red_consec = defaultdict(int)
        total = len(history)

        for record in history:
            reds = sorted(record["reds"])
            consec_count = 0
            for i in range(len(reds) - 1):
                if reds[i + 1] - reds[i] == 1:
                    consec_count += 1
            red_consec[consec_count] += 1

        return {
            "red": {str(k): {"count": v, "ratio": round(v / total * 100, 2)} for k, v in sorted(red_consec.items())},
        }

    def repeat_last_draw(self, history: List[Dict]) -> Dict[str, float]:
        if len(history) < 2:
            return {"red_avg_repeat": 0, "blue_avg_repeat": 0}

        red_repeats = []
        blue_repeats = []

        for i in range(1, len(history)):
            prev_red = set(history[i - 1]["reds"])
            curr_red = set(history[i]["reds"])
            red_repeats.append(len(prev_red & curr_red))

            prev_blue = set(history[i - 1]["blues"])
            curr_blue = set(history[i]["blues"])
            blue_repeats.append(len(prev_blue & curr_blue))

        return {
            "red_avg_repeat": round(sum(red_repeats) / len(red_repeats), 2),
            "blue_avg_repeat": round(sum(blue_repeats) / len(blue_repeats), 2),
        }
