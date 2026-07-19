#!/usr/bin/env python3
"""
权重融合预测 - 使用动态权重综合各算法的预测

策略:
- 每个算法对每个号码"投票" (基于算法对该号码的偏好程度)
- 最终号码得分 = Σ (算法权重 × 算法对号码的打分)
- 选取得分最高的 red_count 个红球 + blue_count 个蓝球

用法: python3 scripts/weighted_predict.py
输出: data/weighted_predictions.json
"""
import sys
import os
import json
import math
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lottery.config import LOTTERY_CONFIGS
from lottery.data import DataLoader
from lottery.models import XGBoostPredictor, MLPredictor
from lottery.models.predictor import LotteryPredictor

WEIGHTS_FILE = os.path.join("data", "algorithm_weights.json")
OUTPUT_FILE = os.path.join("data", "weighted_predictions.json")


def load_json(path, default=None):
    if not os.path.exists(path):
        return default if default is not None else {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_algo_red_scores(algo_name, config, history, xgb_pred=None, mlp_pred=None, trad_pred=None):
    """获取某算法对每个红球号码的偏好分数 (归一化到0-1)"""
    red_min, red_max = config["red_range"]
    red_count = config["red_count"]
    scores = {n: 0.0 for n in range(red_min, red_max + 1)}

    try:
        if algo_name == "xgboost" and xgb_pred is not None and xgb_pred.is_trained:
            reds, blues, info = xgb_pred.predict(history)
            # 用原始预测值的接近程度打分
            raw_preds = info.get("raw_red_preds", [])
            for i, pred in enumerate(raw_preds):
                if i < red_count:
                    # 对预测值附近的号码给高分
                    center = int(round(pred))
                    for n in range(red_min, red_max + 1):
                        dist = abs(n - center)
                        # 高斯衰减
                        s = math.exp(-dist * dist / 2)
                        scores[n] = max(scores[n], s)

        elif algo_name == "mlp" and mlp_pred is not None and mlp_pred.is_trained:
            reds, blues, info = mlp_pred.predict(history)
            top_probs = info.get("red_top_probs", [])
            # 用概率直接打分
            max_p = max((p for _, p in top_probs), default=1)
            for n, p in top_probs:
                if red_min <= n <= red_max:
                    scores[n] = p / max_p if max_p > 0 else 0

        elif algo_name.startswith("trad_"):
            strategy = algo_name[5:]
            if trad_pred is not None:
                red_scores, blue_scores = trad_pred._score_numbers(history, "combined")
                if red_scores:
                    max_s = max(red_scores.values()) or 1
                    for n, s in red_scores.items():
                        if n in scores:
                            scores[n] = s / max_s
    except Exception as e:
        pass

    return scores


def get_algo_blue_scores(algo_name, config, history, xgb_pred=None, mlp_pred=None, trad_pred=None):
    """获取某算法对每个蓝球号码的偏好分数"""
    blue_min, blue_max = config["blue_range"]
    blue_count = config["blue_count"]
    if blue_count == 0:
        return {}

    scores = {n: 0.0 for n in range(blue_min, blue_max + 1)}

    try:
        if algo_name == "xgboost" and xgb_pred is not None and xgb_pred.is_trained:
            reds, blues, info = xgb_pred.predict(history)
            raw_preds = info.get("raw_blue_preds", [])
            for i, pred in enumerate(raw_preds):
                if i < blue_count:
                    center = int(round(pred))
                    for n in range(blue_min, blue_max + 1):
                        dist = abs(n - center)
                        s = math.exp(-dist * dist / 2)
                        scores[n] = max(scores[n], s)

        elif algo_name == "mlp" and mlp_pred is not None and mlp_pred.is_trained:
            reds, blues, info = mlp_pred.predict(history)
            top_probs = info.get("blue_top_probs", [])
            max_p = max((p for _, p in top_probs), default=1)
            for n, p in top_probs:
                if blue_min <= n <= blue_max:
                    scores[n] = p / max_p if max_p > 0 else 0

        elif algo_name.startswith("trad_"):
            if trad_pred is not None:
                red_scores, blue_scores = trad_pred._score_numbers(history, "combined")
                if blue_scores:
                    max_s = max(blue_scores.values()) or 1
                    for n, s in blue_scores.items():
                        if n in scores:
                            scores[n] = s / max_s
    except Exception as e:
        pass

    return scores


def weighted_predict_lottery(lottery_key, config, weights_data):
    """用动态权重融合预测单个彩种"""
    name = config["name"]
    loader = DataLoader(config)
    history = loader.load_history()
    if not history or len(history) < 50:
        return {"lottery": lottery_key, "name": name, "error": "数据不足"}

    red_min, red_max = config["red_range"]
    red_count = config["red_count"]
    blue_count = config["blue_count"]
    is_repeatable = (red_max - red_min + 1) <= 10 and red_count >= 3

    # 加载训练好的模型 (没有就跳过)
    xgb_pred = None
    mlp_pred = None
    try:
        xgb_pred = XGBoostPredictor(config)
        model_dir = os.path.join(os.path.dirname(config["data_file"]), "models", "xgboost")
        xgb_pred.load(model_dir)
    except Exception:
        xgb_pred = None
    try:
        mlp_pred = MLPredictor(config)
        model_dir = os.path.join(os.path.dirname(config["data_file"]), "models", "mlp")
        mlp_pred.load(model_dir)
    except Exception:
        mlp_pred = None

    trad_pred = LotteryPredictor(config)

    # 获取权重
    lot_data = weights_data.get("lotteries", {}).get(lottery_key, {})
    weights = lot_data.get("weights", {})
    if not weights:
        # 默认均等权重
        algos = ["xgboost", "mlp", "trad_热号推荐", "trad_冷号回补",
                 "trad_综合推荐", "trad_随机机选"]
        weights = {a: 1.0 / len(algos) for a in algos}

    print(f"  [{name}] 权重: {weights}")

    # 收集每个算法的红球/蓝球分数
    algo_red_scores = {}
    algo_blue_scores = {}
    active_algos = []

    for algo in weights.keys():
        rs = get_algo_red_scores(algo, config, history, xgb_pred, mlp_pred, trad_pred)
        bs = get_algo_blue_scores(algo, config, history, xgb_pred, mlp_pred, trad_pred)
        if any(v > 0 for v in rs.values()):
            algo_red_scores[algo] = rs
            algo_blue_scores[algo] = bs
            active_algos.append(algo)

    if not active_algos:
        return {"lottery": lottery_key, "name": name, "error": "无可用算法"}

    # 重新归一化权重 (只用活跃算法)
    total_w = sum(weights[a] for a in active_algos)
    if total_w == 0:
        norm_weights = {a: 1.0 / len(active_algos) for a in active_algos}
    else:
        norm_weights = {a: weights[a] / total_w for a in active_algos}

    # 融合红球分数
    final_red_scores = {n: 0.0 for n in range(red_min, red_max + 1)}
    for algo in active_algos:
        w = norm_weights[algo]
        rs = algo_red_scores[algo]
        for n, s in rs.items():
            if n in final_red_scores:
                final_red_scores[n] += w * s

    # 融合蓝球分数
    final_blue_scores = {}
    if blue_count > 0:
        blue_min, blue_max = config["blue_range"]
        final_blue_scores = {n: 0.0 for n in range(blue_min, blue_max + 1)}
        for algo in active_algos:
            w = norm_weights[algo]
            bs = algo_blue_scores.get(algo, {})
            for n, s in bs.items():
                if n in final_blue_scores:
                    final_blue_scores[n] += w * s

    # 选取得分最高的号码
    if is_repeatable:
        # 可重复号码彩种 (七星彩/排列3/排列5): 按位置选择
        # 这里简化: 用综合分数top作为各位置的预测
        # 实际上每个位置独立 - 用MLP的位置概率或XGBoost的位置预测更合理
        # 简化处理: 取top red_count 个号码 (允许重复)
        sorted_reds = sorted(final_red_scores.items(), key=lambda x: -x[1])
        reds = [n for n, _ in sorted_reds[:red_count]]
        # 对于可重复彩种, 实际应该按位置 - 这里用每个位置top1
        if xgb_pred is not None and xgb_pred.is_trained:
            xgb_reds, _, _ = xgb_pred.predict(history)
            if len(xgb_reds) == red_count:
                reds = xgb_reds
    else:
        # 不可重复: 取top red_count (去重天然满足)
        sorted_reds = sorted(final_red_scores.items(), key=lambda x: -x[1])
        reds = sorted([n for n, _ in sorted_reds[:red_count]])

    blues = []
    if blue_count > 0 and final_blue_scores:
        sorted_blues = sorted(final_blue_scores.items(), key=lambda x: -x[1])
        blues = sorted([n for n, _ in sorted_blues[:blue_count]])

    last = history[-1]
    result = {
        "lottery": lottery_key,
        "name": name,
        "schedule": config["schedule"],
        "predict_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "latest_issue": last["issue"],
        "latest_reds": last["reds"],
        "latest_blues": last["blues"],
        "history_count": len(history),
        "weighted_prediction": {
            "reds": reds,
            "blues": blues,
        },
        "weights_used": norm_weights,
        "active_algos": active_algos,
        "top_red_scores": [(n, round(s, 4)) for n, s in sorted_reds[:10]],
        "top_blue_scores": [(n, round(s, 4)) for n, s in sorted(
            final_blue_scores.items(), key=lambda x: -x[1])[:5]] if final_blue_scores else [],
        "is_repeatable": is_repeatable,
    }
    return result


def main():
    print(f"权重融合预测 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    weights_data = load_json(WEIGHTS_FILE, default={})
    if not weights_data:
        print(f"⚠️ 权重文件不存在 ({WEIGHTS_FILE}), 使用默认均等权重")

    all_results = []
    for key, config in LOTTERY_CONFIGS.items():
        print(f"\n预测 {config['name']} ({key}) ...")
        result = weighted_predict_lottery(key, config, weights_data)
        if "error" in result:
            print(f"  ✗ {result['error']}")
        else:
            pred = result["weighted_prediction"]
            print(f"  最新期: {result['latest_issue']} 红={result['latest_reds']} 蓝={result['latest_blues']}")
            print(f"  ★ 权重融合预测: 红={pred['reds']} 蓝={pred['blues']}")
        all_results.append(result)

    save_json(OUTPUT_FILE, all_results)
    print(f"\n权重融合预测已保存: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
