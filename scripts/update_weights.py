#!/usr/bin/env python3
"""
动态权重更新系统 - 基于历史命中率调整各算法权重

权重计算原理:
- 对每个彩种，使用滑动窗口回测最近 N 期
- 每次回测: 用 [0:i] 的数据训练模型, 预测第 i 期, 对比实际开奖
- 算法得分 = 红球命中率 * 0.6 + 蓝球命中率 * 0.4 (无蓝球则只看红球)
- 权重 = exp(score / temperature) (softmax归一化)
- 历史命中权重指数衰减 (近期权重更高)

用法: python3 scripts/update_weights.py [--backtest N]
输出: data/algorithm_weights.json
"""
import sys
import os
import json
import time
import argparse
import math
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lottery.config import LOTTERY_CONFIGS
from lottery.data import DataLoader
from lottery.models import XGBoostPredictor, MLPredictor
from lottery.models.predictor import LotteryPredictor

WEIGHTS_FILE = os.path.join("data", "algorithm_weights.json")
HITS_FILE = os.path.join("data", "prediction_hits.json")


def load_json(path, default=None):
    if not os.path.exists(path):
        return default if default is not None else []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def count_hits(predicted, actual, is_repeatable=False):
    """计算命中数"""
    if is_repeatable:
        return sum(1 for p, a in zip(predicted, actual) if p == a)
    return len(set(predicted) & set(actual))


def backtest_lottery(lottery_key, config, n_backtests=20):
    """对单个彩种进行回测

    策略: 取最近 n_backtests 期作为验证集，
    每次用之前所有历史数据训练模型并预测，对比实际开奖
    """
    name = config["name"]
    loader = DataLoader(config)
    history = loader.load_history()
    if len(history) < 100:
        return None

    red_min, red_max = config["red_range"]
    red_count = config["red_count"]
    is_repeatable = (red_max - red_min + 1) <= 10 and red_count >= 3
    blue_count = config["blue_count"]

    # 回测点 (最近 n_backtests 期)
    total = len(history)
    n_bt = min(n_backtests, total - 50)
    if n_bt < 3:
        return None

    backtest_points = list(range(total - n_bt, total))

    # 算法得分累积
    algo_stats = defaultdict(lambda: {
        "red_hits": 0, "red_total": 0,
        "blue_hits": 0, "blue_total": 0,
        "full_red": 0, "full_blue": 0, "full_match": 0,
        "count": 0,
        "recent_scores": [],  # 最近的得分序列
    })

    print(f"  [{name}] 回测 {n_bt} 期 (从第 {backtest_points[0]} 到 {backtest_points[-1]})...")

    # 为了加速: XGBoost和MLP训练较慢，回测时减少训练频率
    # 每5期重新训练一次模型
    xgb_pred = None
    mlp_pred = None
    last_train_idx = -100

    for idx in backtest_points:
        train_data = history[:idx]
        actual = history[idx]
        actual_reds = actual["reds"]
        actual_blues = actual["blues"]

        # 距离上次训练超过5期才重新训练 (加速)
        should_retrain = (idx - last_train_idx) >= 5

        # 1. XGBoost
        try:
            if should_retrain or xgb_pred is None:
                xgb_pred = XGBoostPredictor(config)
                train_result = xgb_pred.train(train_data)
                if not train_result.get("success"):
                    xgb_pred = None
                last_train_idx = idx
            if xgb_pred is not None and xgb_pred.is_trained:
                reds, blues, _ = xgb_pred.predict(train_data)
                if reds:
                    rh = count_hits(reds, actual_reds, is_repeatable)
                    bh = count_hits(blues, actual_blues, False) if blues and blue_count > 0 else 0
                    _update_stats(algo_stats["xgboost"], rh, len(actual_reds),
                                  bh, len(actual_blues))
        except Exception as e:
            pass

        # 2. MLP (回测时只在前2个回测点用, 太慢)
        if idx - backtest_points[0] < 2:
            try:
                mlp_pred = MLPredictor(config)
                train_result = mlp_pred.train(train_data)
                if train_result.get("success"):
                    reds, blues, _ = mlp_pred.predict(train_data)
                    if reds:
                        rh = count_hits(reds, actual_reds, is_repeatable)
                        bh = count_hits(blues, actual_blues, False) if blues and blue_count > 0 else 0
                        _update_stats(algo_stats["mlp"], rh, len(actual_reds),
                                      bh, len(actual_blues))
            except Exception as e:
                pass

        # 3. 传统策略 (快)
        try:
            trad = LotteryPredictor(config)
            strategies = trad.predict_multi_strategy(train_data)
            for sname, (reds, blues) in strategies.items():
                rh = count_hits(reds, actual_reds, is_repeatable)
                bh = count_hits(blues, actual_blues, False) if blues and blue_count > 0 else 0
                _update_stats(algo_stats[f"trad_{sname}"], rh, len(actual_reds),
                              bh, len(actual_blues))
        except Exception as e:
            pass

    # 计算每个算法的综合得分
    algo_scores = {}
    for algo, stats in algo_stats.items():
        if stats["count"] == 0:
            continue
        red_rate = stats["red_hits"] / max(stats["red_total"], 1)
        blue_rate = stats["blue_hits"] / max(stats["blue_total"], 1) if stats["blue_total"] > 0 else 0

        # 综合得分: 红球命中率为主，蓝球为辅
        if blue_count > 0:
            score = red_rate * 0.6 + blue_rate * 0.4
        else:
            score = red_rate

        # 完全命中额外加分
        full_bonus = (stats["full_match"] * 0.3 +
                      stats["full_red"] * 0.15 +
                      stats["full_blue"] * 0.1) / max(stats["count"], 1)
        score += full_bonus

        algo_scores[algo] = {
            "score": round(score, 4),
            "red_rate": round(red_rate, 4),
            "blue_rate": round(blue_rate, 4) if blue_count > 0 else 0,
            "red_hits": stats["red_hits"],
            "red_total": stats["red_total"],
            "blue_hits": stats["blue_hits"],
            "blue_total": stats["blue_total"],
            "full_match": stats["full_match"],
            "full_red": stats["full_red"],
            "full_blue": stats["full_blue"],
            "count": stats["count"],
        }

    # 计算softmax权重 (让得分高的算法权重更大)
    temperature = 0.05  # 温度越低，差异越大
    scores = {a: s["score"] for a, s in algo_scores.items()}
    if scores:
        max_score = max(scores.values())
        exp_scores = {a: math.exp((s - max_score) / temperature) for a, s in scores.items()}
        total_exp = sum(exp_scores.values())
        weights = {a: round(e / total_exp, 4) for a, e in exp_scores.items()}
    else:
        weights = {}

    return {
        "lottery": lottery_key,
        "name": name,
        "backtest_count": n_bt,
        "is_repeatable": is_repeatable,
        "blue_count": blue_count,
        "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm_scores": algo_scores,
        "weights": weights,
    }


def _update_stats(stats, red_hits, red_total, blue_hits, blue_total):
    """更新算法统计"""
    stats["red_hits"] += red_hits
    stats["red_total"] += red_total
    stats["blue_hits"] += blue_hits
    stats["blue_total"] += blue_total
    stats["count"] += 1
    if red_hits == red_total:
        stats["full_red"] += 1
    if blue_total > 0 and blue_hits == blue_total:
        stats["full_blue"] += 1
    if red_hits == red_total and (blue_total == 0 or blue_hits == blue_total):
        stats["full_match"] += 1


def merge_with_history_weights(new_weights, history_weights):
    """与历史权重平滑合并 (EMA: 指数移动平均)"""
    alpha = 0.3  # 新权重占比
    merged = {}
    for lottery_key, lot_data in new_weights.items():
        old_lot = history_weights.get(lottery_key, {})
        merged_w = {}
        all_algos = set(lot_data["weights"].keys()) | set(old_lot.get("weights", {}).keys())
        for algo in all_algos:
            new_w = lot_data["weights"].get(algo, 0)
            old_w = old_lot.get("weights", {}).get(algo, 0)
            # EMA 平滑
            merged_w[algo] = round(alpha * new_w + (1 - alpha) * old_w, 4)
        # 重新归一化
        total_w = sum(merged_w.values())
        if total_w > 0:
            merged_w = {a: round(w / total_w, 4) for a, w in merged_w.items()}

        merged[lottery_key] = {
            **lot_data,
            "weights": merged_w,
            "previous_weights": old_lot.get("weights", {}),
        }
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest", type=int, default=20,
                        help="回测期数 (默认20)")
    parser.add_argument("--no-merge", action="store_true",
                        help="不与历史权重合并")
    args = parser.parse_args()

    print(f"动态权重更新 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"回测期数: {args.backtest}")

    # 加载历史权重
    history_weights = load_json(WEIGHTS_FILE, default={})
    if not isinstance(history_weights, dict):
        history_weights = {}

    # 计算每个彩种的新权重
    new_weights = {}
    for key, config in LOTTERY_CONFIGS.items():
        print(f"\n处理 {config['name']} ({key}) ...")
        t0 = time.time()
        result = backtest_lottery(key, config, n_backtests=args.backtest)
        if result is None:
            print(f"  数据不足，跳过")
            continue
        elapsed = time.time() - t0
        new_weights[key] = result
        print(f"  完成 ({elapsed:.1f}s)")
        print(f"  权重: {result['weights']}")
        # 打印 Top3 算法
        sorted_algos = sorted(result["algorithm_scores"].items(),
                              key=lambda x: -x[1]["score"])[:3]
        for algo, s in sorted_algos:
            print(f"    {algo}: 得分={s['score']} 红球命中率={s['red_rate']}"
                  f" ({s['red_hits']}/{s['red_total']})", end="")
            if result["blue_count"] > 0:
                print(f" 蓝球命中率={s['blue_rate']} ({s['blue_hits']}/{s['blue_total']})", end="")
            if s["full_match"] > 0:
                print(f" ★完全命中={s['full_match']}", end="")
            print()

    # 与历史权重合并 (EMA平滑)
    if args.no_merge or not history_weights:
        final_weights = new_weights
    else:
        print(f"\n与历史权重合并 (EMA α=0.3) ...")
        final_weights = merge_with_history_weights(new_weights, history_weights)

    # 保存元信息
    output = {
        "_meta": {
            "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "backtest_count": args.backtest,
            "merge_alpha": 0.3 if not args.no_merge else 0,
            "description": "动态算法权重 - 基于历史回测命中率计算, EMA平滑更新",
        },
        "lotteries": final_weights,
    }

    save_json(WEIGHTS_FILE, output)
    print(f"\n权重已保存: {WEIGHTS_FILE}")

    # 汇总
    print(f"\n{'='*60}")
    print("权重更新汇总:")
    print(f"{'='*60}")
    for key, data in final_weights.items():
        name = data["name"]
        weights = data["weights"]
        top_algo = max(weights.items(), key=lambda x: x[1]) if weights else ("-", 0)
        # 找最高准确率
        scores = data.get("algorithm_scores", {})
        if scores:
            top_score_algo = max(scores.items(), key=lambda x: x[1]["score"])
            top_score = top_score_algo[1]["score"]
            top_red_rate = top_score_algo[1]["red_rate"]
        else:
            top_score = 0
            top_red_rate = 0
        print(f"  {name:<8}: Top算法={top_algo[0]} (权重={top_algo[1]:.3f})"
              f" 最高得分={top_score:.3f} 红球命中率={top_red_rate:.3f}")


if __name__ == "__main__":
    main()
