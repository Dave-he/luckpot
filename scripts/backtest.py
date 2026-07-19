#!/usr/bin/env python3
"""
回测验证脚本 - 用历史数据评估各算法和权重融合的长期表现

策略:
- 从历史数据倒数 N 期开始, 每次用之前所有数据预测下一期
- 统计每个算法 (含 weighted_ensemble) 的命中率
- 找出最佳算法和是否达到 50% 准确率

用法: python3 scripts/backtest.py [--n N] [--lottery KEY]
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


def count_hits(predicted, actual, is_repeatable=False):
    if is_repeatable:
        return sum(1 for p, a in zip(predicted, actual) if p == a)
    return len(set(predicted) & set(actual))


def weighted_vote_predict(config, history, xgb_pred, mlp_pred, trad_pred,
                          weights, last_train_idx):
    """权重融合预测 - 简化版用于回测"""
    red_min, red_max = config["red_range"]
    red_count = config["red_count"]
    blue_count = config["blue_count"]
    is_repeatable = (red_max - red_min + 1) <= 10 and red_count >= 3

    # 收集每个算法对每个号码的分数
    final_red = defaultdict(float)
    final_blue = defaultdict(float)
    active_algos = []

    # XGBoost
    if xgb_pred is not None and xgb_pred.is_trained:
        try:
            reds, blues, info = xgb_pred.predict(history)
            raw_preds = info.get("raw_red_preds", [])
            for i, pred in enumerate(raw_preds):
                if i < red_count:
                    center = int(round(pred))
                    for n in range(red_min, red_max + 1):
                        dist = abs(n - center)
                        s = math.exp(-dist * dist / 2)
                        final_red[n] += weights.get("xgboost", 0) * s
            active_algos.append("xgboost")
        except Exception:
            pass

    # MLP
    if mlp_pred is not None and mlp_pred.is_trained:
        try:
            reds, blues, info = mlp_pred.predict(history)
            top_probs = info.get("red_top_probs", [])
            max_p = max((p for _, p in top_probs), default=1)
            for n, p in top_probs:
                if red_min <= n <= red_max:
                    final_red[n] += weights.get("mlp", 0) * (p / max_p if max_p > 0 else 0)
            active_algos.append("mlp")
        except Exception:
            pass

    # 传统策略 - 综合推荐
    if trad_pred is not None:
        try:
            red_scores, blue_scores = trad_pred._score_numbers(history, "combined")
            if red_scores:
                max_s = max(red_scores.values()) or 1
                for n, s in red_scores.items():
                    final_red[n] += weights.get("trad_综合推荐", 0) * (s / max_s)
            active_algos.append("trad_综合推荐")
        except Exception:
            pass

    if not active_algos:
        return [], []

    # 归一化权重
    total_w = sum(weights.get(a, 0) for a in active_algos)
    if total_w == 0:
        return [], []

    # 选号
    if is_repeatable:
        if xgb_pred is not None and xgb_pred.is_trained:
            reds, _, _ = xgb_pred.predict(history)
            if len(reds) == red_count:
                return reds, []
        sorted_reds = sorted(final_red.items(), key=lambda x: -x[1])
        reds = [n for n, _ in sorted_reds[:red_count]]
    else:
        sorted_reds = sorted(final_red.items(), key=lambda x: -x[1])
        reds = sorted([n for n, _ in sorted_reds[:red_count]])

    blues = []
    if blue_count > 0:
        blue_min, blue_max = config["blue_range"]
        sorted_blues = sorted(final_blue.items(), key=lambda x: -x[1])
        blues = sorted([n for n, _ in sorted_blues[:blue_count]])

    return reds, blues


def backtest_lottery(lottery_key, config, n_backtests, use_mlp=False):
    """回测单个彩种"""
    name = config["name"]
    loader = DataLoader(config)
    history = loader.load_history()
    if len(history) < 100:
        return None

    red_min, red_max = config["red_range"]
    red_count = config["red_count"]
    is_repeatable = (red_max - red_min + 1) <= 10 and red_count >= 3
    blue_count = config["blue_count"]

    total = len(history)
    n_bt = min(n_backtests, total - 50)
    backtest_points = list(range(total - n_bt, total))

    algo_stats = defaultdict(lambda: {
        "red_hits": 0, "red_total": 0,
        "blue_hits": 0, "blue_total": 0,
        "full_red": 0, "full_blue": 0, "full_match": 0,
        "at_least_1_red": 0, "at_least_2_red": 0, "at_least_1_blue": 0,
        "count": 0,
    })

    print(f"\n  [{name}] 回测 {n_bt} 期...")

    xgb_pred = None
    mlp_pred = None
    last_train_idx = -100
    # 训练时计算的权重 (回测期间用同一个)
    backtest_weights = {
        "xgboost": 0.25, "mlp": 0.25, "trad_综合推荐": 0.5
    }

    for i, idx in enumerate(backtest_points):
        train_data = history[:idx]
        actual = history[idx]
        actual_reds = actual["reds"]
        actual_blues = actual["blues"]

        # 每5期重新训练一次
        should_retrain = (idx - last_train_idx) >= 5

        if should_retrain:
            try:
                xgb_pred = XGBoostPredictor(config)
                r = xgb_pred.train(train_data)
                if not r.get("success"):
                    xgb_pred = None
            except Exception:
                xgb_pred = None
            last_train_idx = idx

        # MLP只在前3个回测点用 (太慢)
        if use_mlp and i < 3:
            try:
                mlp_pred = MLPredictor(config)
                r = mlp_pred.train(train_data)
                if not r.get("success"):
                    mlp_pred = None
            except Exception:
                mlp_pred = None
        elif i >= 3:
            mlp_pred = None

        trad_pred = LotteryPredictor(config)

        # 1. XGBoost
        if xgb_pred is not None and xgb_pred.is_trained:
            try:
                reds, blues, _ = xgb_pred.predict(train_data)
                if reds:
                    rh = count_hits(reds, actual_reds, is_repeatable)
                    bh = count_hits(blues, actual_blues, False) if blues and blue_count > 0 else 0
                    _update(algo_stats["xgboost"], rh, len(actual_reds), bh, len(actual_blues))
            except Exception:
                pass

        # 2. MLP
        if mlp_pred is not None and mlp_pred.is_trained:
            try:
                reds, blues, _ = mlp_pred.predict(train_data)
                if reds:
                    rh = count_hits(reds, actual_reds, is_repeatable)
                    bh = count_hits(blues, actual_blues, False) if blues and blue_count > 0 else 0
                    _update(algo_stats["mlp"], rh, len(actual_reds), bh, len(actual_blues))
            except Exception:
                pass

        # 3. 传统策略
        try:
            strategies = trad_pred.predict_multi_strategy(train_data)
            for sname, (reds, blues) in strategies.items():
                rh = count_hits(reds, actual_reds, is_repeatable)
                bh = count_hits(blues, actual_blues, False) if blues and blue_count > 0 else 0
                _update(algo_stats[f"trad_{sname}"], rh, len(actual_reds), bh, len(actual_blues))
        except Exception:
            pass

        # 4. 权重融合
        try:
            reds, blues = weighted_vote_predict(config, train_data, xgb_pred, mlp_pred,
                                                trad_pred, backtest_weights, idx)
            if reds:
                rh = count_hits(reds, actual_reds, is_repeatable)
                bh = count_hits(blues, actual_blues, False) if blues and blue_count > 0 else 0
                _update(algo_stats["weighted_ensemble"], rh, len(actual_reds), bh, len(actual_blues))
        except Exception:
            pass

        if (i + 1) % 10 == 0:
            print(f"    进度: {i+1}/{n_bt}")

    # 计算命中率
    results = {}
    for algo, stats in algo_stats.items():
        if stats["count"] == 0:
            continue
        red_rate = stats["red_hits"] / max(stats["red_total"], 1)
        blue_rate = stats["blue_hits"] / max(stats["blue_total"], 1) if stats["blue_total"] > 0 else 0
        results[algo] = {
            "red_rate": round(red_rate, 4),
            "blue_rate": round(blue_rate, 4),
            "red_hits": stats["red_hits"],
            "red_total": stats["red_total"],
            "blue_hits": stats["blue_hits"],
            "blue_total": stats["blue_total"],
            "full_match": stats["full_match"],
            "full_red": stats["full_red"],
            "full_blue": stats["full_blue"],
            "at_least_1_red": stats["at_least_1_red"],
            "at_least_2_red": stats["at_least_2_red"],
            "at_least_1_blue": stats["at_least_1_blue"],
            "at_least_1_red_rate": round(stats["at_least_1_red"] / max(stats["count"], 1), 4),
            "at_least_2_red_rate": round(stats["at_least_2_red"] / max(stats["count"], 1), 4),
            "at_least_1_blue_rate": round(stats["at_least_1_blue"] / max(stats["count"], 1), 4),
            "count": stats["count"],
        }

    return {
        "lottery": lottery_key,
        "name": name,
        "backtest_count": n_bt,
        "is_repeatable": is_repeatable,
        "results": results,
    }


def _update(stats, rh, rt, bh, bt):
    stats["red_hits"] += rh
    stats["red_total"] += rt
    stats["blue_hits"] += bh
    stats["blue_total"] += bt
    stats["count"] += 1
    if rh == rt:
        stats["full_red"] += 1
    if bt > 0 and bh == bt:
        stats["full_blue"] += 1
    if rh == rt and (bt == 0 or bh == bt):
        stats["full_match"] += 1
    # 新指标: 至少命中1个红球
    if rh >= 1:
        stats["at_least_1_red"] += 1
    # 至少命中2个红球
    if rh >= 2:
        stats["at_least_2_red"] += 1
    # 至少命中1个蓝球
    if bh >= 1:
        stats["at_least_1_blue"] += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="回测期数")
    parser.add_argument("--lottery", type=str, default="", help="只回测某个彩种")
    parser.add_argument("--use-mlp", action="store_true", help="回测MLP (慢)")
    args = parser.parse_args()

    print(f"回测验证 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"回测期数: {args.n}")

    all_results = []
    for key, config in LOTTERY_CONFIGS.items():
        if args.lottery and key != args.lottery:
            continue
        t0 = time.time()
        r = backtest_lottery(key, config, args.n, use_mlp=args.use_mlp)
        if r:
            elapsed = time.time() - t0
            print(f"  完成 ({elapsed:.1f}s)")
            all_results.append(r)

    # 打印汇总
    print(f"\n{'='*80}")
    print("回测结果汇总:")
    print(f"{'='*80}")
    print(f"\n各彩种各算法命中率:")
    print(f"{'彩种':<10} {'算法':<22} {'红球命中率':<14} {'至少1红':<14} {'至少2红':<14} {'完全命中':<10} {'达标≥50%':<10}")
    print("-" * 100)

    reached_50 = []
    for r in all_results:
        name = r["name"]
        for algo, s in r["results"].items():
            red_pct = f"{s['red_rate']*100:.1f}% ({s['red_hits']}/{s['red_total']})"
            at_least_1 = f"{s['at_least_1_red_rate']*100:.1f}% ({s['at_least_1_red']}/{s['count']})"
            at_least_2 = f"{s['at_least_2_red_rate']*100:.1f}% ({s['at_least_2_red']}/{s['count']})"
            full = f"{s['full_match']}/{s['count']}"
            # 达标条件: "至少命中1个红球" 比例 >= 50%
            reached = "✓" if s["at_least_1_red_rate"] >= 0.5 else ""
            if reached:
                reached_50.append((name, algo, s["at_least_1_red_rate"], s["count"], "at_least_1_red"))
            print(f"{name:<10} {algo:<22} {red_pct:<14} {at_least_1:<14} {at_least_2:<14} {full:<10} {reached:<10}")

    print(f"\n{'='*100}")
    if reached_50:
        print(f"★ 达到 50% 准确率的算法 ({len(reached_50)} 个) [指标: 至少命中1个红球]:")
        for name, algo, rate, count, metric in reached_50:
            print(f"  - {name} / {algo}: {rate*100:.1f}% (基于 {count} 期回测)")
    else:
        print("目前还没有算法在'至少命中1红'上达到 50%。")
        all_algos = []
        for r in all_results:
            for algo, s in r["results"].items():
                all_algos.append((r["name"], algo, s["at_least_1_red_rate"], s["count"]))
        all_algos.sort(key=lambda x: -x[2])
        print("\nTop 5 '至少命中1红' 准确率最高的算法:")
        for name, algo, rate, count in all_algos[:5]:
            print(f"  - {name} / {algo}: {rate*100:.1f}% (基于 {count} 期回测)")

    # 保存结果
    output_path = "data/backtest_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "backtest_count": args.n,
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n回测报告已保存: {output_path}")


if __name__ == "__main__":
    main()
