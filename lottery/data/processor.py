from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict


class DataProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.red_min, self.red_max = config["red_range"]
        self.blue_min, self.blue_max = config["blue_range"]
        self.red_count = config["red_count"]
        self.blue_count = config["blue_count"]
        self.red_total = self.red_max - self.red_min + 1
        self.blue_total = self.blue_max - self.blue_min + 1

    def numbers_to_onehot(self, reds: List[int], blues: List[int]) -> np.ndarray:
        red_vec = np.zeros(self.red_total, dtype=np.float32)
        blue_vec = np.zeros(self.blue_total, dtype=np.float32)

        for n in reds:
            if self.red_min <= n <= self.red_max:
                red_vec[n - self.red_min] = 1.0
        for n in blues:
            if self.blue_min <= n <= self.blue_max:
                blue_vec[n - self.blue_min] = 1.0

        return np.concatenate([red_vec, blue_vec])

    def onehot_to_numbers(self, vec: np.ndarray) -> Tuple[List[int], List[int]]:
        red_vec = vec[:self.red_total]
        blue_vec = vec[self.red_total:]

        red_indices = np.argsort(red_vec)[-self.red_count:]
        blue_indices = np.argsort(blue_vec)[-self.blue_count:]

        reds = sorted([int(i) + self.red_min for i in red_indices])
        blues = sorted([int(i) + self.blue_min for i in blue_indices])
        return reds, blues

    def build_sequences(self, history: List[Dict], window_size: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        red_vectors = []
        blue_vectors = []

        for record in history:
            vec = self.numbers_to_onehot(record["reds"], record["blues"])
            red_vec = vec[:self.red_total]
            blue_vec = vec[self.red_total:]
            red_vectors.append(red_vec)
            blue_vectors.append(blue_vec)

        red_vectors = np.array(red_vectors)
        blue_vectors = np.array(blue_vectors)

        for i in range(window_size, len(history)):
            red_window = red_vectors[i - window_size:i].flatten()
            blue_window = blue_vectors[i - window_size:i].flatten()
            X.append(np.concatenate([red_window, blue_window]))
            y.append(np.concatenate([red_vectors[i], blue_vectors[i]]))

        return np.array(X), np.array(y)

    def get_recent_frequency(self, history: List[Dict], recent_n: int = 30) -> Dict[str, Dict[int, int]]:
        recent = history[-recent_n:] if len(history) >= recent_n else history
        red_freq = defaultdict(int)
        blue_freq = defaultdict(int)

        for record in recent:
            for n in record["reds"]:
                red_freq[n] += 1
            for n in record["blues"]:
                blue_freq[n] += 1

        return {"red": dict(red_freq), "blue": dict(blue_freq)}

    def get_missing_values(self, history: List[Dict]) -> Dict[str, Dict[int, int]]:
        red_missing = {n: 0 for n in range(self.red_min, self.red_max + 1)}
        blue_missing = {n: 0 for n in range(self.blue_min, self.blue_max + 1)}

        red_found = {n: False for n in red_missing}
        blue_found = {n: False for n in blue_missing}

        for record in reversed(history):
            all_reds_found = all(red_found.values())
            all_blues_found = all(blue_found.values())
            if all_reds_found and all_blues_found:
                break

            for n in red_missing:
                if not red_found[n]:
                    if n in record["reds"]:
                        red_found[n] = True
                    else:
                        red_missing[n] += 1
            for n in blue_missing:
                if not blue_found[n]:
                    if n in record["blues"]:
                        blue_found[n] = True
                    else:
                        blue_missing[n] += 1

        return {"red": red_missing, "blue": blue_missing}

    def get_consecutive_pairs(self, history: List[Dict], top_n: int = 10) -> Dict[str, List[Tuple[Tuple[int, int], int]]]:
        red_pairs = defaultdict(int)
        blue_pairs = defaultdict(int)

        for record in history:
            reds = sorted(record["reds"])
            for i in range(len(reds) - 1):
                for j in range(i + 1, len(reds)):
                    pair = (reds[i], reds[j])
                    red_pairs[pair] += 1
            blues = sorted(record["blues"])
            for i in range(len(blues) - 1):
                for j in range(i + 1, len(blues)):
                    pair = (blues[i], blues[j])
                    blue_pairs[pair] += 1

        sorted_red = sorted(red_pairs.items(), key=lambda x: x[1], reverse=True)[:top_n]
        sorted_blue = sorted(blue_pairs.items(), key=lambda x: x[1], reverse=True)[:top_n]

        return {"red": sorted_red, "blue": sorted_blue}
