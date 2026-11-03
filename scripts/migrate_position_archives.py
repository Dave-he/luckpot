#!/usr/bin/env python3
"""隔离旧排序预测记录，保留审计副本，禁止与修复后的统计混算。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lottery.config import LOTTERY_CONFIGS


def main():
    marker = Path('data/legacy-position-v1/migration.json')
    if marker.exists():
        return
    marker.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name in ('prediction_hits.json', 'prediction_history.json', 'predictions.json',
                 'weighted_predictions.json'):
        path = Path('data', name)
        if not path.exists():
            continue
        rows = json.loads(path.read_text())
        legacy = [r for r in rows if LOTTERY_CONFIGS[r['lottery']].get('ordered')]
        keep = [r for r in rows if not LOTTERY_CONFIGS[r['lottery']].get('ordered')]
        (marker.parent / name).write_text(json.dumps(legacy, ensure_ascii=False, indent=2))
        path.write_text(json.dumps(keep, ensure_ascii=False, indent=2))
        counts[name] = len(legacy)
    marker.write_text(json.dumps({'reason': '旧记录丢失原始位置，不能作为中奖证据',
                                 'isolated_records': counts}, ensure_ascii=False, indent=2))
    print('已隔离旧排序预测记录:', counts)


if __name__ == '__main__':
    main()
