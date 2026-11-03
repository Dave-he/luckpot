#!/usr/bin/env python3
"""发布前校验七个彩种均完成本轮更新、训练及合法预测。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lottery.config import LOTTERY_CONFIGS
from lottery.data import DataLoader


def validate_numbers(config, prediction):
    reds, blues = prediction.get('reds', []), prediction.get('blues', [])
    assert len(reds) == config['red_count'], '主号码数量错误'
    assert len(blues) == config['blue_count'], '副号码数量错误'
    assert all(type(n) is int and lo <= n <= hi
               for n, (lo, hi) in zip(reds, config['red_position_ranges'])), '主号码越界'
    assert all(type(n) is int and config['blue_range'][0] <= n <= config['blue_range'][1]
               for n in blues), '副号码越界'
    if not config.get('ordered'):
        assert len(set(reds)) == len(reds), '主号码重复'
    assert len(set(blues)) == len(blues), '副号码重复'


def read_index(name):
    rows = json.loads(Path('data', name).read_text())
    assert len(rows) == len(LOTTERY_CONFIGS), f'{name}: 彩种缺失或重复'
    result = {r['lottery']: r for r in rows}
    assert set(result) == set(LOTTERY_CONFIGS), f'{name}: 彩种不完整'
    return result


def main():
    updates = read_index('update_report.json')
    training = read_index('training_report.json')
    predictions = read_index('predictions.json')
    weighted = read_index('weighted_predictions.json')
    for key, config in LOTTERY_CONFIGS.items():
        assert updates[key]['status'] == 'success', f'{key}: 更新失败'
        assert training[key].get('models', {}).get('xgboost', {}).get('success'), f'{key}: XGBoost训练失败'
        history = DataLoader(config).load_history()
        assert len(history) >= 50, f'{key}: 训练数据不足'
        assert training[key]['latest_issue'] == history[-1]['issue'], f'{key}: 模型期号过时'
        if config.get('ordered'):
            assert Path(config['data_file'] + '.schema.json').exists(), f'{key}: 未重建原始位置数据'
        for record in history:
            validate_numbers(config, record)
        pred = predictions[key]
        assert pred['latest_issue'] == history[-1]['issue'], f'{key}: 预测期号过时'
        assert weighted[key]['latest_issue'] == history[-1]['issue'], f'{key}: 融合期号过时'
        validate_numbers(config, pred['predictions']['xgboost'])
        for name, value in pred['predictions'].items():
            if name == 'traditional':
                for pair in value.values():
                    validate_numbers(config, pair)
            else:
                assert 'error' not in value, f'{key}/{name}: 预测失败'
                validate_numbers(config, value)
        validate_numbers(config, weighted[key]['weighted_prediction'])
        print(f'{key}: 数据/训练/预测校验通过（最新期 {history[-1]["issue"]}）')


if __name__ == '__main__':
    main()
