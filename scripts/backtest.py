#!/usr/bin/env python3
"""
回测验证脚本 - 用历史数据评估各算法和权重融合的长期表现

策略:
- 从历史数据倒数 N 期开始, 每次用之前所有数据预测下一期
- 统计每个算法 (含 weighted_ensemble) 的命中率
- 找出最佳算法和是否达到 50% 准确率

支持算法: XGBoost, MLP, RandomForest, Markov, NaiveBayes,
         MonteCarlo, KMeans, LSTM, 传统策略, weighted_ensemble

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
from lottery.models import (
    XGBoostPredictor, MLPredictor,
    RandomForestPredictor, MarkovPredictor,
    NaiveBayesPredictor, MonteCarloPredictor,
    KMeansPredictor, LSTMPredictor,
)
from lottery.models.predictor import LotteryPredictor


def count_hits(predicted, actual, is_repeatable=False):
    if is_repeatable:
        return sum(1 for p, a in zip(predicted, actual) if p == a)
    return len(set(predicted) & set(actual))


def _model_red_scores(model, config, history):
    """从模型预测结果提取红球分数 (归一化0-1)"""
    red_min, red_max = config["red_range"]
    scores = {n: 0.0 for n in range(red_min, red_max + 1)}
    try:
        reds, blues, info = model.predict(history)
        # 优先使用 red_top_probs
        top_probs = info.get("red_top_probs", []) if isinstance(info, dict) else []
        if top_probs:
            max_p = max((p for _, p in top_probs), default=1)
            for n, p in top_probs:
                if red_min <= n <= red_max:
                    scores[n] = p / max_p if max_p > 0 else 0
        else:
            # 回退: 用预测的红球号码给满分
            for n in reds:
                if red_min <= n <= red_max:
                    scores[n] = 1.0
    except Exception:
        pass
    return scores


def _model_blue_scores(model, config, history):
    """从模型预测结果提取蓝球分数"""
    blue_count = config["blue_count"]
    if blue_count == 0:
        return {}
    blue_min, blue_max = config["blue_range"]
    scores = {n: 0.0 for n in range(blue_min, blue_max + 1)}
    try:
        reds, blues, info = model.predict(history)
        top_probs = info.get("blue_top_probs", []) if isinstance(info, dict) else []
        if top_probs:
            max_p = max((p for _, p in top_probs), default=1)
            for n, p in top_probs:
                if blue_min <= n <= blue_max:
                    scores[n] = p / max_p if max_p > 0 else 0
        else:
            for n in blues:
                if blue_min <= n <= blue_max:
                    scores[n] = 1.0
    except Exception:
        pass
    return scores


def _xgb_red_scores(xgb_pred, config, history):
    """XGBoost 专用: 用 raw_red_preds 高斯衰减"""
    red_min, red_max = config["red_range"]
    red_count = config["red_count"]
    scores = {n: 0.0 for n in range(red_min, red_max + 1)}
    try:
        reds, blues, info = xgb_pred.predict(history)
        raw_preds = info.get("raw_red_preds", [])
        for i, pred in enumerate(raw_preds):
            if i < red_count:
                center = int(round(pred))
                for n in range(red_min, red_max + 1):
                    dist = abs(n - center)
                    s = math.exp(-dist * dist / 2)
                    scores[n] = max(scores[n], s)
    except Exception:
        pass
    return scores


def _xgb_blue_scores(xgb_pred, config, history):
    """XGBoost 专用: 用 raw_blue_preds 高斯衰减"""
    blue_count = config["blue_count"]
    if blue_count == 0:
        return {}
    blue_min, blue_max = config["blue_range"]
    scores = {n: 0.0 for n in range(blue_min, blue_max + 1)}
    try:
        reds, blues, info = xgb_pred.predict(history)
        raw_preds = info.get("raw_blue_preds", [])
        for i, pred in enumerate(raw_preds):
            if i < blue_count:
                center = int(round(pred))
                for n in range(blue_min, blue_max + 1):
                    dist = abs(n - center)
                    s = math.exp(-dist * dist / 2)
                    scores[n] = max(scores[n], s)
    except Exception:
        pass
    return scores


def _trad_scores(trad_pred, config, history):
    """传统策略的分数"""
    red_min, red_max = config["red_range"]
    blue_count = config["blue_count"]
    try:
        red_scores, blue_scores = trad_pred._score_numbers(history, "combined")
        if red_scores:
            max_s = max(red_scores.values()) or 1
            red_norm = {n: red_scores.get(n, 0) / max_s for n in range(red_min, red_max + 1)}
        else:
            red_norm = {}
        blue_norm = {}
        if blue_count > 0 and blue_scores:
            blue_min, blue_max = config["blue_range"]
            max_s = max(blue_scores.values()) or 1
            blue_norm = {n: blue_scores.get(n, 0) / max_s for n in range(blue_min, blue_max + 1)}
        return red_norm, blue_norm
    except Exception:
        return {}, {}


def weighted_vote_predict(config, history, models, trad_pred, weights):
    """权重融合预测 - 用所有算法投票

    models: dict of {algo_name: model_instance_or_None}
    weights: dict of {algo_name: weight}
    """
    red_min, red_max = config["red_range"]
    red_count = config["red_count"]
    blue_count = config["blue_count"]
    is_repeatable = (red_max - red_min + 1) <= 10 and red_count >= 3

    final_red = defaultdict(float)
    final_blue = defaultdict(float)
    active_algos = []

    # XGBoost (特殊处理)
    xgb = models.get("xgboost")
    if xgb is not None and xgb.is_trained:
        w = weights.get("xgboost", 0)
        rs = _xgb_red_scores(xgb, config, history)
        for n, s in rs.items():
            final_red[n] += w * s
        bs = _xgb_blue_scores(xgb, config, history)
        for n, s in bs.items():
            final_blue[n] += w * s
        if any(v > 0 for v in rs.values()):
            active_algos.append("xgboost")

    # 其他 ML 模型
    other_models = ["mlp", "random_forest", "markov", "naive_bayes",
                    "monte_carlo", "kmeans", "lstm"]
    for mname in other_models:
        m = models.get(mname)
        if m is not None and m.is_trained:
            w = weights.get(mname, 0)
            rs = _model_red_scores(m, config, history)
            for n, s in rs.items():
                final_red[n] += w * s
            bs = _model_blue_scores(m, config, history)
            for n, s in bs.items():
                final_blue[n] += w * s
            if any(v > 0 for v in rs.values()):
                active_algos.append(mname)

    # 传统策略
    if trad_pred is not None:
        w = weights.get("trad_综合推荐", 0)
        if w > 0:
            red_norm, blue_norm = _trad_scores(trad_pred, config, history)
            for n, s in red_norm.items():
                final_red[n] += w * s
            for n, s in blue_norm.items():
                final_blue[n] += w * s
            if red_norm:
                active_algos.append("trad_综合推荐")

    if not active_algos:
        return [], []

    total_w = sum(weights.get(a, 0) for a in active_algos)
    if total_w == 0:
        return [], []

    # 选号
    if is_repeatable:
        # 可重复: 优先用 XGBoost 按位置预测
        if xgb is not None and xgb.is_trained:
            reds, _, _ = xgb.predict(history)
            if len(reds) == red_count:
                return reds, []
        sorted_reds = sorted(final_red.items(), key=lambda x: -x[1])
        reds = [n for n, _ in sorted_reds[:red_count]]
    else:
        sorted_reds = sorted(final_red.items(), key=lambda x: -x[1])
        reds = sorted([n for n, _ in sorted_reds[:red_count]])

    blues = []
    if blue_count > 0 and final_blue:
        sorted_blues = sorted(final_blue.items(), key=lambda x: -x[1])
        blues = sorted([n for n, _ in sorted_blues[:blue_count]])

    return reds, blues


def backtest_lottery(lottery_key, config, n_backtests, use_slow=True):
    """回测单个彩种

    use_slow: 是否使用慢速模型 (MLP, LSTM)
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

    total = len(history)
    n_bt = min(n_backtests, total - 50)
    backtest_points = list(range(total - n_bt, total))

    algo_stats = defaultdict(lambda: {
        "red_hits": 0, "red_total": 0,
        "blue_hits": 0, "blue_total": 0,
        "full_red": 0, "full_blue": 0, "full_match": 0,
        "at_least_1_red": 0, "at_least_2_red": 0, "at_least_1_blue": 0,
        "count": 0,
        "prize_1": 0, "prize_2": 0, "prize_3": 0,
        "prize_4": 0, "prize_5": 0, "prize_6": 0, "no_prize": 0,
    })

    print(f"\n  [{name}] 回测 {n_bt} 期 (is_repeatable={is_repeatable})...")

    # 持久化模型 (每5期重训一次)
    cached_models = {}  # {algo_name: model}
    last_train_idx = -100

    # 默认权重 (均等)
    backtest_weights = {
        "xgboost": 0.15, "mlp": 0.10,
        "random_forest": 0.15, "markov": 0.10,
        "naive_bayes": 0.10, "monte_carlo": 0.10,
        "kmeans": 0.10, "lstm": 0.10,
        "trad_综合推荐": 0.10,
    }

    for i, idx in enumerate(backtest_points):
        train_data = history[:idx]
        actual = history[idx]
        actual_reds = actual["reds"]
        actual_blues = actual["blues"]

        should_retrain = (idx - last_train_idx) >= 5

        # --- 训练各模型 ---
        # 快速模型 (每5期重训)
        if should_retrain:
            # XGBoost
            try:
                m = XGBoostPredictor(config)
                r = m.train(train_data)
                cached_models["xgboost"] = m if r.get("success") else None
            except Exception:
                cached_models["xgboost"] = None

            # Random Forest
            try:
                m = RandomForestPredictor(config)
                r = m.train(train_data)
                cached_models["random_forest"] = m if r.get("success") else None
            except Exception:
                cached_models["random_forest"] = None

            # Naive Bayes
            try:
                m = NaiveBayesPredictor(config)
                r = m.train(train_data)
                cached_models["naive_bayes"] = m if r.get("success") else None
            except Exception:
                cached_models["naive_bayes"] = None

            # KMeans
            try:
                m = KMeansPredictor(config)
                r = m.train(train_data)
                cached_models["kmeans"] = m if r.get("success") else None
            except Exception:
                cached_models["kmeans"] = None

            last_train_idx = idx

        # Markov & Monte Carlo (每期都重训, 极快)
        try:
            m = MarkovPredictor(config)
            m.train(train_data)
            cached_models["markov"] = m
        except Exception:
            cached_models["markov"] = None

        try:
            m = MonteCarloPredictor(config)
            m.train(train_data)
            cached_models["monte_carlo"] = m
        except Exception:
            cached_models["monte_carlo"] = None

        # MLP & LSTM (慢, 仅前3期用)
        if use_slow and i < 3:
            try:
                m = MLPredictor(config)
                r = m.train(train_data)
                cached_models["mlp"] = m if r.get("success") else None
            except Exception:
                cached_models["mlp"] = None

            try:
                m = LSTMPredictor(config)
                m.epochs = 15
                lstm_data = train_data[-300:] if len(train_data) > 300 else train_data
                r = m.train(lstm_data)
                cached_models["lstm"] = m if r.get("success") else None
            except Exception:
                cached_models["lstm"] = None
        elif i >= 3:
            cached_models["mlp"] = None
            cached_models["lstm"] = None

        trad_pred = LotteryPredictor(config)

        # --- 评估每个算法 ---
        for algo_name in ["xgboost", "mlp", "random_forest", "markov",
                          "naive_bayes", "monte_carlo", "kmeans", "lstm"]:
            m = cached_models.get(algo_name)
            if m is None or not m.is_trained:
                continue
            try:
                reds, blues, _ = m.predict(train_data)
                if reds:
                    rh = count_hits(reds, actual_reds, is_repeatable)
                    bh = count_hits(blues, actual_blues, False) if blues and blue_count > 0 else 0
                    _update(algo_stats[algo_name], rh, len(actual_reds), bh, len(actual_blues))
            except Exception:
                pass

        # 传统策略
        try:
            strategies = trad_pred.predict_multi_strategy(train_data)
            for sname, (reds, blues) in strategies.items():
                rh = count_hits(reds, actual_reds, is_repeatable)
                bh = count_hits(blues, actual_blues, False) if blues and blue_count > 0 else 0
                _update(algo_stats[f"trad_{sname}"], rh, len(actual_reds), bh, len(actual_blues))
        except Exception:
            pass

        # 权重融合
        try:
            reds, blues = weighted_vote_predict(
                config, train_data, cached_models, trad_pred, backtest_weights
            )
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
            "full_match_rate": round(stats["full_match"] / max(stats["count"], 1), 4),
            "at_least_1_red": stats["at_least_1_red"],
            "at_least_2_red": stats["at_least_2_red"],
            "at_least_1_blue": stats["at_least_1_blue"],
            "at_least_1_red_rate": round(stats["at_least_1_red"] / max(stats["count"], 1), 4),
            "at_least_2_red_rate": round(stats["at_least_2_red"] / max(stats["count"], 1), 4),
            "at_least_1_blue_rate": round(stats["at_least_1_blue"] / max(stats["count"], 1), 4),
            # 各奖项等级获奖次数及概率
            "prize_1": stats.get("prize_1", 0),
            "prize_2": stats.get("prize_2", 0),
            "prize_3": stats.get("prize_3", 0),
            "prize_4": stats.get("prize_4", 0),
            "prize_5": stats.get("prize_5", 0),
            "prize_6": stats.get("prize_6", 0),
            "no_prize": stats.get("no_prize", 0),
            "prize_1_rate": round(stats.get("prize_1", 0) / max(stats["count"], 1), 4),
            "prize_2_rate": round(stats.get("prize_2", 0) / max(stats["count"], 1), 4),
            "prize_3_rate": round(stats.get("prize_3", 0) / max(stats["count"], 1), 4),
            "prize_4_rate": round(stats.get("prize_4", 0) / max(stats["count"], 1), 4),
            "prize_5_rate": round(stats.get("prize_5", 0) / max(stats["count"], 1), 4),
            "prize_6_rate": round(stats.get("prize_6", 0) / max(stats["count"], 1), 4),
            # 总获奖次数 (二等及以上算高奖)
            "total_prize": stats.get("prize_1", 0) + stats.get("prize_2", 0) + stats.get("prize_3", 0)
                           + stats.get("prize_4", 0) + stats.get("prize_5", 0) + stats.get("prize_6", 0),
            "total_prize_rate": round(
                (stats.get("prize_1", 0) + stats.get("prize_2", 0) + stats.get("prize_3", 0)
                 + stats.get("prize_4", 0) + stats.get("prize_5", 0) + stats.get("prize_6", 0))
                / max(stats["count"], 1), 4),
            "high_prize": stats.get("prize_1", 0) + stats.get("prize_2", 0) + stats.get("prize_3", 0),
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
    if rh >= 1:
        stats["at_least_1_red"] += 1
    if rh >= 2:
        stats["at_least_2_red"] += 1
    if bh >= 1:
        stats["at_least_1_blue"] += 1

    # --- 各奖项等级获奖次数统计 ---
    # 等级判定依据中国福彩/体彩通用规则 (红球命中数 + 蓝球命中数)
    # 一等奖: 全中 (红球全中 + 蓝球全中)
    # 二等奖: 红球全中 + 蓝球未全中
    # 三等奖: 红球命中 rt-1 + 蓝球全中
    # 四等奖: 红球命中 rt-1 + 蓝球未全中, 或 红球命中 rt-2 + 蓝球全中
    # 五等奖: 红球命中 rt-2 + 蓝球未全中, 或 红球命中 rt-3 + 蓝球全中
    # 六等奖: 红球命中 < rt-3 + 蓝球全中
    # 注: 简化模型, 适用于双色球/大乐透; 纯数字彩种(福彩3D/排列3/排列5/七星彩)按全中或部分中
    full_blue = (bt > 0 and bh == bt) or bt == 0
    full_red = (rh == rt)

    if full_red and full_blue:
        stats["prize_1"] = stats.get("prize_1", 0) + 1  # 一等奖 (全中)
    elif full_red and not full_blue:
        stats["prize_2"] = stats.get("prize_2", 0) + 1  # 二等奖 (红全中, 蓝未全中)
    elif rh == rt - 1 and full_blue:
        stats["prize_3"] = stats.get("prize_3", 0) + 1  # 三等奖
    elif (rh == rt - 1 and not full_blue) or (rh == rt - 2 and full_blue):
        stats["prize_4"] = stats.get("prize_4", 0) + 1  # 四等奖
    elif (rh == rt - 2 and not full_blue) or (rh == rt - 3 and full_blue):
        stats["prize_5"] = stats.get("prize_5", 0) + 1  # 五等奖
    elif full_blue and not full_red:
        stats["prize_6"] = stats.get("prize_6", 0) + 1  # 六等奖
    else:
        stats["no_prize"] = stats.get("no_prize", 0) + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="回测期数")
    parser.add_argument("--lottery", type=str, default="", help="只回测某个彩种")
    parser.add_argument("--no-slow", action="store_true", help="跳过慢速模型 (MLP/LSTM)")
    args = parser.parse_args()

    print(f"回测验证 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"回测期数: {args.n}")
    print(f"慢速模型 (MLP/LSTM): {'禁用' if args.no_slow else '启用'}")

    all_results = []
    for key, config in LOTTERY_CONFIGS.items():
        if args.lottery and key != args.lottery:
            continue
        t0 = time.time()
        r = backtest_lottery(key, config, args.n, use_slow=not args.no_slow)
        if r:
            elapsed = time.time() - t0
            print(f"  完成 ({elapsed:.1f}s)")
            all_results.append(r)

    # 打印汇总
    print(f"\n{'='*80}")
    print("回测结果汇总:")
    print(f"{'='*80}")
    print(f"\n各彩种各算法命中率:")
    print(f"{'彩种':<10} {'算法':<22} {'红球命中率':<14} {'至少1红':<14} {'至少2红':<14} {'全中率':<14} {'达标≥50%':<10}")
    print("-" * 110)

    reached_50 = []
    for r in all_results:
        name = r["name"]
        for algo, s in r["results"].items():
            red_pct = f"{s['red_rate']*100:.1f}% ({s['red_hits']}/{s['red_total']})"
            at_least_1 = f"{s['at_least_1_red_rate']*100:.1f}% ({s['at_least_1_red']}/{s['count']})"
            at_least_2 = f"{s['at_least_2_red_rate']*100:.1f}% ({s['at_least_2_red']}/{s['count']})"
            full_rate = f"{s['full_match_rate']*100:.1f}% ({s['full_match']}/{s['count']})"
            reached = "✓" if s["at_least_1_red_rate"] >= 0.5 else ""
            if reached:
                reached_50.append((name, algo, s["at_least_1_red_rate"], s["count"], "at_least_1_red"))
            print(f"{name:<10} {algo:<22} {red_pct:<14} {at_least_1:<14} {at_least_2:<14} {full_rate:<14} {reached:<10}")

    print(f"\n{'='*110}")
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

    # --- 按获奖次数排名 ---
    print(f"\n{'='*110}")
    print("按获奖次数排名 (总获奖次数 = 一等+二等+...+六等):")
    print(f"{'='*110}")
    print(f"\n{'彩种':<10} {'算法':<22} {'一等':<6} {'二等':<6} {'三等':<6} {'四等':<6} {'五等':<6} {'六等':<6} {'总获奖':<8} {'获奖率':<10}")
    print("-" * 110)

    all_prize_ranking = []
    for r in all_results:
        name = r["name"]
        for algo, s in r["results"].items():
            p1 = s.get("prize_1", 0)
            p2 = s.get("prize_2", 0)
            p3 = s.get("prize_3", 0)
            p4 = s.get("prize_4", 0)
            p5 = s.get("prize_5", 0)
            p6 = s.get("prize_6", 0)
            total_prize = s.get("total_prize", 0)
            total_prize_rate = s.get("total_prize_rate", 0)
            all_prize_ranking.append((name, algo, p1, p2, p3, p4, p5, p6,
                                       total_prize, total_prize_rate, s["count"]))
            print(f"{name:<10} {algo:<22} {p1:<6} {p2:<6} {p3:<6} {p4:<6} {p5:<6} {p6:<6} {total_prize:<8} {total_prize_rate*100:.1f}%")

    # 按总获奖次数排序 (跨彩种全局排名)
    all_prize_ranking.sort(key=lambda x: -x[8])  # 按总获奖次数降序

    print(f"\n{'='*110}")
    print("★ 全局 Top 10 获奖次数最多的算法 (跨彩种排名):")
    print(f"{'='*110}")
    for i, (name, algo, p1, p2, p3, p4, p5, p6, tp, tpr, cnt) in enumerate(all_prize_ranking[:10], 1):
        high_prize = p1 + p2 + p3
        print(f"  {i:>2}. {name:<10} / {algo:<22} | "
              f"一等={p1} 二等={p2} 三等={p3} 四等={p4} 五等={p5} 六等={p6} | "
              f"总获奖={tp}/{cnt} ({tpr*100:.1f}%) 高等奖(1-3等)={high_prize}")

    # 按高等奖排名
    all_prize_ranking.sort(key=lambda x: -(x[2] + x[3] + x[4]))
    print(f"\n{'='*110}")
    print("★ 全局 Top 10 高等奖 (一等+二等+三等) 最多的算法:")
    print(f"{'='*110}")
    for i, (name, algo, p1, p2, p3, p4, p5, p6, tp, tpr, cnt) in enumerate(all_prize_ranking[:10], 1):
        high_prize = p1 + p2 + p3
        if high_prize == 0:
            break
        print(f"  {i:>2}. {name:<10} / {algo:<22} | "
              f"一等={p1} 二等={p2} 三等={p3} | 高等奖合计={high_prize}/{cnt} ({high_prize/cnt*100:.1f}%)")

    # 全中概率排名
    all_full_match = []
    for r in all_results:
        for algo, s in r["results"].items():
            if s["full_match"] > 0:
                all_full_match.append((r["name"], algo, s["full_match"], s["count"], s["full_match_rate"]))
    all_full_match.sort(key=lambda x: -x[2])

    print(f"\n{'='*110}")
    if all_full_match:
        print(f"★ 全中 (一等奖) 算法排名 - 共 {len(all_full_match)} 个算法实现过全中:")
        for name, algo, fm, cnt, fmr in all_full_match:
            print(f"  - {name:<10} / {algo:<22}: 全中 {fm} 次 / {cnt} 期 ({fmr*100:.2f}%)")
    else:
        print("★ 全中 (一等奖): 暂无算法在回测中实现过全中。")

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
