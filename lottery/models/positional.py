"""按原始位置融合及滚动回测，不把数字彩当作无序号码集合。"""
from collections import Counter, defaultdict
from datetime import datetime

from .predictor import LotteryPredictor
from .xgboost_model import XGBoostPredictor


def blend(config, candidates, weights):
    reds = []
    for pos, (lo, hi) in enumerate(config['red_position_ranges']):
        scores = defaultdict(float)
        for name, numbers in candidates.items():
            if len(numbers) == config['red_count'] and lo <= numbers[pos] <= hi:
                scores[numbers[pos]] += weights.get(name, 1.0)
        if not scores:
            raise ValueError(f'第 {pos + 1} 位无有效候选')
        reds.append(max(scores, key=scores.get))
    return reds


def prize_level(key, predicted, actual):
    hits = sum(p == a for p, a in zip(predicted, actual))
    if hits == len(actual):
        return 1
    if key != 'qxc':
        return 0  # 3D/排列3 只评估单选/直选；排列5只有全中中奖。
    first = sum(p == a for p, a in zip(predicted[:6], actual[:6]))
    last = predicted[6] == actual[6]
    if first == 6:
        return 2
    if first == 5 and last:
        return 3
    if hits == 5:
        return 4
    if hits == 4:
        return 5
    if hits == 3 or (first in (0, 1) and last):
        return 6
    return 0


def backtest(key, config, history, n):
    n = min(n, len(history) - 50)
    stats = defaultdict(Counter)
    model = None
    for step, idx in enumerate(range(len(history) - n, len(history))):
        train = history[:idx]
        if model is None or step % 5 == 0:
            model = XGBoostPredictor(config)
            if not model.train(train).get('success'):
                raise RuntimeError(f'{key} XGBoost 回测训练失败')
        candidates = {'xgboost': model.predict(train)[0]}
        trad = LotteryPredictor(config).predict_multi_strategy(train)
        candidates.update({f'trad_{name}': pair[0] for name, pair in trad.items()})
        candidates['weighted_ensemble'] = blend(config, candidates, {})
        for name, numbers in candidates.items():
            s = stats[name]
            hits = sum(p == a for p, a in zip(numbers, history[idx]['reds']))
            full = hits == config['red_count']
            s.update({'count': 1, 'red_hits': hits, 'red_total': config['red_count'],
                      'full_red': int(full), 'full_match': int(full),
                      'at_least_1_red': int(hits >= 1), 'at_least_2_red': int(hits >= 2)})
            level = prize_level(key, numbers, history[idx]['reds'])
            s[f'prize_{level}' if level else 'no_prize'] += 1
    results = {}
    for name, s in stats.items():
        r = {field: s[field] for field in ('count', 'red_hits', 'red_total', 'blue_hits',
             'blue_total', 'full_red', 'full_blue', 'full_match', 'at_least_1_red',
             'at_least_2_red', 'at_least_1_blue', 'no_prize')}
        r.update({f'prize_{i}': s[f'prize_{i}'] for i in range(1, 7)})
        r['red_rate'] = round(s['red_hits'] / s['red_total'], 4)
        r['blue_rate'] = 0
        for field in ('full_match', 'at_least_1_red', 'at_least_2_red', 'at_least_1_blue'):
            r[f'{field}_rate'] = round(s[field] / s['count'], 4)
        r['total_prize'] = s['count'] - s['no_prize']
        r['total_prize_rate'] = round(r['total_prize'] / s['count'], 4)
        r['high_prize'] = s['prize_1'] + s['prize_2']
        for i in range(1, 7):
            r[f'prize_{i}_rate'] = round(s[f'prize_{i}'] / s['count'], 4)
        results[name] = r
    return {'lottery': key, 'name': config['name'], 'backtest_count': n,
            'is_repeatable': True, 'blue_count': 0, 'results': results,
            'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
