import csv
import os
from typing import List, Dict, Optional, Tuple


def issue_sort_key(issue: str) -> Tuple[int, int]:
    issue = issue.strip()
    try:
        num = int(issue)
    except ValueError:
        return (0, 0)

    if len(issue) == 5:
        year = 2000 + (num // 1000)
        seq = num % 1000
        return (year, seq)
    elif len(issue) == 7:
        year = num // 1000
        seq = num % 1000
        return (year, seq)
    else:
        return (num, 0)


class DataLoader:
    def __init__(self, config: Dict):
        self.config = config
        self.data_file = config["data_file"]
        self.red_count = config["red_count"]
        self.blue_count = config["blue_count"]

    def load_history(self) -> List[Dict]:
        if not os.path.exists(self.data_file):
            return []

        results = []
        with open(self.data_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    issue = row.get("issue", "").strip()
                    date = row.get("date", "").strip()
                    reds = []
                    for i in range(1, self.red_count + 1):
                        val = row.get(f"red{i}", "").strip()
                        if val:
                            reds.append(int(val))
                    blues = []
                    for i in range(1, self.blue_count + 1):
                        val = row.get(f"blue{i}", "").strip()
                        if val:
                            blues.append(int(val))

                    if reds and blues and len(reds) == self.red_count and len(blues) == self.blue_count:
                        results.append({
                            "issue": issue,
                            "date": date,
                            "reds": sorted(reds),
                            "blues": sorted(blues),
                        })
                except (ValueError, KeyError):
                    continue

        results.sort(key=lambda x: issue_sort_key(x["issue"]))
        return results

    def get_latest_issue(self) -> Optional[str]:
        data = self.load_history()
        if not data:
            return None
        return data[-1]["issue"]

    def save_data(self, records: List[Dict], append: bool = True):
        mode = "a" if append and os.path.exists(self.data_file) else "w"
        file_exists = os.path.exists(self.data_file)

        if not append:
            existing = []
        else:
            existing = {r["issue"] for r in self.load_history()}

        with open(self.data_file, mode, newline="", encoding="utf-8") as f:
            fieldnames = ["issue", "date"]
            fieldnames.extend([f"red{i}" for i in range(1, self.red_count + 1)])
            fieldnames.extend([f"blue{i}" for i in range(1, self.blue_count + 1)])

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists or mode == "w":
                writer.writeheader()

            for record in records:
                if append and record["issue"] in existing:
                    continue
                row = {"issue": record["issue"], "date": record.get("date", "")}
                for i, n in enumerate(record["reds"], 1):
                    row[f"red{i}"] = f"{n:02d}"
                for i, n in enumerate(record["blues"], 1):
                    row[f"blue{i}"] = f"{n:02d}"
                writer.writerow(row)

    def merge_existing_csv(self, old_files: List[str]):
        all_records = {}

        for old_file in old_files:
            if not os.path.exists(old_file):
                continue
            with open(old_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if not row or len(row) < 1 + self.red_count + self.blue_count:
                        continue
                    try:
                        issue = row[0].strip()
                        if not issue:
                            continue
                        reds = []
                        for i in range(self.red_count):
                            if 1 + i >= len(row):
                                break
                            val = row[1 + i].strip()
                            if val:
                                reds.append(int(val))
                        blues = []
                        for i in range(self.blue_count):
                            idx = 1 + self.red_count + i
                            if idx >= len(row):
                                break
                            val = row[idx].strip()
                            if val:
                                blues.append(int(val))
                        if len(reds) == self.red_count and len(blues) == self.blue_count:
                            all_records[issue] = {
                                "issue": issue,
                                "date": "",
                                "reds": sorted(reds),
                                "blues": sorted(blues),
                            }
                    except (ValueError, IndexError):
                        continue

        records = sorted(all_records.values(), key=lambda x: issue_sort_key(x["issue"]))
        self.save_data(records, append=False)
        return len(records)
