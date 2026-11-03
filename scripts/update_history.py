#!/usr/bin/env python3
"""
预测历史归档脚本 - 维护 docs/prediction-history.md 的机器数据源
data/prediction_history.json

职责:
1. 预测阶段 (--mode record): 把本次「下一期预测」写入历史, 状态 pending
2. 开奖阶段 (--mode settle): 用最新历史数据回填已开奖期号的开奖号码,
   计算每个模型的命中情况, 状态 settled

这样 README 只展示「下一期预测」, 历史文件展示「历次预测 + 实际开奖 + 命中」。

用法:
  python3 scripts/update_history.py --mode record   # 记录本次预测
  python3 scripts/update_history.py --mode settle   # 结算已开奖的预测
  python3 scripts/update_history.py                 # 先 settle 再 record
"""
import sys
import os
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lottery.config import LOTTERY_CONFIGS
from lottery.data import DataLoader, next_issue

HISTORY_FILE = os.path.join("data", "prediction_history.json")
PRED_FILE = os.path.join("data", "predictions.json")
WEIGHTED_FILE = os.path.join("data", "weighted_predictions.json")

# 只归档这几组预测, 避免历史文件无限膨胀
ARCHIVE_MODELS = [
    ("xgboost", "XGBoost"),
    ("mlp", "MLP神经网络"),
    ("random_forest", "随机森林"),
    ("markov", "马尔可夫链"),
    ("naive_bayes", "朴素贝叶斯"),
    ("monte_carlo", "蒙特卡洛"),
    ("kmeans", "K-Means"),
    ("lstm", "LSTM"),
    ("traditional/热号推荐", "热号推荐"),
    ("traditional/冷号回补", "冷号回补"),
    ("traditional/综合推荐", "综合推荐"),
]


def load_json(path, default=None):
    if not os.path.exists(path):
        return default if default is not None else []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _get_path(pred, dotted_key):
    """支持 'traditional/热号推荐' 这类带斜杠的路径取值"""
    cur = pred
    for part in dotted_key.split("/"):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def is_repeatable(config):
    """可重复号码彩种(七星彩/排列3/排列5)需按位置匹配"""
    return config.get("ordered", False)


def count_hits(predicted, actual, repeatable=False):
    if repeatable:
        return sum(1 for p, a in zip(predicted, actual) if p == a)
    return len(set(predicted) & set(actual))


def collect_predictions(pred_entry, lottery_key):
    """从预测结果中提取待归档的模型预测"""
    predictions = pred_entry.get("predictions", {})
    out = []

    for key, label in ARCHIVE_MODELS:
        item = _get_path(predictions, key)
        if isinstance(item, dict) and item.get("reds"):
            out.append({
                "model": label,
                "reds": sorted(int(x) for x in item["reds"]),
                "blues": sorted(int(x) for x in item.get("blues", [])),
            })

    # 权重融合预测单独追加
    for wp in load_json(WEIGHTED_FILE, default=[]):
        if wp.get("lottery") != lottery_key:
            continue
        pred = wp.get("weighted_prediction", {})
        if pred.get("reds"):
            out.append({
                "model": "权重融合",
                "reds": sorted(int(x) for x in pred["reds"]),
                "blues": sorted(int(x) for x in pred.get("blues", [])),
            })
        break

    return out


def record_predictions(history):
    """记录本次预测 (状态 pending)"""
    predictions = load_json(PRED_FILE, default=[])
    if not predictions:
        print("无预测结果, 跳过归档")
        return history

    # 已存在的 (彩种, 目标期号) 索引, 避免重复写入
    existing = {
        (h["lottery"], h["target_issue"]) for h in history
    }

    added = 0
    for pred in predictions:
        lottery_key = pred.get("lottery")
        config = LOTTERY_CONFIGS.get(lottery_key)
        if not config:
            continue

        latest_issue = pred.get("latest_issue")
        target_issue = next_issue(latest_issue)
        if not target_issue:
            continue

        models = collect_predictions(pred, lottery_key)
        if not models:
            continue

        if (lottery_key, target_issue) in existing:
            # 已归档过: 用最新预测覆盖 (同一天多次运行)
            for h in history:
                if h["lottery"] == lottery_key and h["target_issue"] == target_issue:
                    h["models"] = models
                    h["predict_time"] = pred.get("predict_time", "")
            continue

        history.append({
            "lottery": lottery_key,
            "name": pred.get("name", lottery_key),
            "target_issue": target_issue,
            "based_on_issue": latest_issue,
            "predict_time": pred.get("predict_time", ""),
            "status": "pending",
            "actual_reds": [],
            "actual_blues": [],
            "draw_date": "",
            "models": models,
        })
        existing.add((lottery_key, target_issue))
        added += 1

    print(f"  归档新增 {added} 条预测 (累计 {len(history)} 条)")
    return history


def settle_predictions(history):
    """回填已开奖预测的开奖号码并计算命中"""
    if not history:
        return history

    # 每个彩种的期号 -> 开奖记录 映射
    draw_index = {}
    for key, config in LOTTERY_CONFIGS.items():
        loader = DataLoader(config)
        draw_index[key] = {r["issue"]: r for r in loader.load_history()}

    settled = 0
    for entry in history:
        if entry.get("status") == "settled":
            continue

        lottery_key = entry["lottery"]
        config = LOTTERY_CONFIGS.get(lottery_key)
        draw = draw_index.get(lottery_key, {}).get(entry["target_issue"])
        if not draw:
            continue

        actual_reds = sorted(draw["reds"])
        actual_blues = sorted(draw["blues"])
        repeatable = is_repeatable(config)

        entry["actual_reds"] = actual_reds
        entry["actual_blues"] = actual_blues
        entry["draw_date"] = draw.get("date", "")
        entry["settle_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry["status"] = "settled"

        best_hits = -1
        for m in entry.get("models", []):
            red_hits = count_hits(m["reds"], actual_reds, repeatable)
            blue_hits = count_hits(m["blues"], actual_blues) if m["blues"] else 0
            m["red_hits"] = red_hits
            m["blue_hits"] = blue_hits
            m["red_total"] = len(actual_reds)
            m["blue_total"] = len(actual_blues)
            m["full_red"] = red_hits == len(actual_reds)
            m["full_blue"] = blue_hits == len(actual_blues) if actual_blues else True
            m["full_match"] = m["full_red"] and m["full_blue"]
            best_hits = max(best_hits, red_hits)

        entry["best_red_hits"] = max(best_hits, 0)
        settled += 1

    if settled:
        print(f"  结算 {settled} 条已开奖预测")
    return history


def main():
    parser = argparse.ArgumentParser(description="预测历史归档")
    parser.add_argument("--mode", choices=["record", "settle", "both"],
                        default="both", help="record=归档预测, settle=结算开奖, both=先结算后归档")
    args = parser.parse_args()

    print(f"预测历史归档 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    history = load_json(HISTORY_FILE, default=[])

    if args.mode in ("settle", "both"):
        history = settle_predictions(history)
    if args.mode in ("record", "both"):
        history = record_predictions(history)

    # 按期号倒序, 最新的排前面
    history.sort(key=lambda x: (x["lottery"], x["target_issue"]), reverse=True)

    save_json(HISTORY_FILE, history)
    pending = sum(1 for h in history if h["status"] == "pending")
    settled = sum(1 for h in history if h["status"] == "settled")
    print(f"已保存: {HISTORY_FILE} (共 {len(history)} 条, 待开奖 {pending}, 已开奖 {settled})")


if __name__ == "__main__":
    main()
