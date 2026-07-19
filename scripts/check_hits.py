#!/usr/bin/env python3
"""
预测命中检查脚本 - 对比上次预测与最新开奖结果
- 读取 data/predictions.json (上次预测)
- 读取最新历史数据
- 对比命中情况
- 命中的预测永久记录到 data/prediction_hits.json
用法: python3 scripts/check_hits.py
"""
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lottery.config import LOTTERY_CONFIGS
from lottery.data import DataLoader


HITS_FILE = os.path.join("data", "prediction_hits.json")
PRED_FILE = os.path.join("data", "predictions.json")


def load_json(path, default=None):
    if not os.path.exists(path):
        return default if default is not None else []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def count_hits(predicted, actual, is_repeatable=False):
    """计算命中数

    - 不可重复彩种(双色球等): 集合交集
    - 可重复彩种(七星彩/排列3/排列5): 严格按位置匹配
    """
    if is_repeatable:
        # 按位置匹配 (长度可能不同，取最小长度)
        hits = sum(1 for p, a in zip(predicted, actual) if p == a)
    else:
        # 集合交集
        hits = len(set(predicted) & set(actual))
    return hits


def check_lottery(pred_entry, history_hits):
    """检查单个彩种的预测命中情况"""
    lottery_key = pred_entry.get("lottery")
    name = pred_entry.get("name", lottery_key)
    config = LOTTERY_CONFIGS.get(lottery_key)
    if not config:
        return None

    # 上次预测时记录的最新期号
    last_pred_issue = pred_entry.get("latest_issue")
    if not last_pred_issue:
        return None

    # 加载最新历史数据，找到 last_pred_issue 之后的下一期
    loader = DataLoader(config)
    history = loader.load_history()
    if not history:
        return None

    # 找到上次预测时最新期的索引
    last_idx = -1
    for i, h in enumerate(history):
        if h["issue"] == last_pred_issue:
            last_idx = i
            break

    if last_idx < 0 or last_idx + 1 >= len(history):
        # 没有新的开奖期，无法对比
        return None

    # 新一期的实际开奖号码
    new_draw = history[last_idx + 1]
    actual_reds = new_draw["reds"]
    actual_blues = new_draw["blues"]

    # 判断彩种类型
    red_min, red_max = config["red_range"]
    red_count = config["red_count"]
    is_repeatable = (red_max - red_min + 1) <= 10 and red_count >= 3

    # 检查每个模型的预测
    predictions = pred_entry.get("predictions", {})
    results = []

    def check_one(model_name, pred_reds, pred_blues):
        if not pred_reds:
            return None
        red_hits = count_hits(pred_reds, actual_reds, is_repeatable)
        blue_hits = count_hits(pred_blues, actual_blues, False) if pred_blues else 0

        total_red = len(actual_reds)
        total_blue = len(actual_blues)

        # 命中等级:
        # - 红球全中: red_hits == total_red
        # - 蓝球全中: blue_hits == total_blue
        # - 完全命中: 红球+蓝球全中
        full_red = red_hits == total_red
        full_blue = blue_hits == total_blue if total_blue > 0 else True
        full_match = full_red and full_blue

        return {
            "model": model_name,
            "predicted_reds": pred_reds,
            "predicted_blues": pred_blues,
            "actual_reds": actual_reds,
            "actual_blues": actual_blues,
            "red_hits": red_hits,
            "red_total": total_red,
            "blue_hits": blue_hits,
            "blue_total": total_blue,
            "full_red_match": full_red,
            "full_blue_match": full_blue,
            "full_match": full_match,
        }

    # XGBoost
    if "xgboost" in predictions and "reds" in predictions["xgboost"]:
        r = check_one("xgboost", predictions["xgboost"]["reds"], predictions["xgboost"].get("blues", []))
        if r:
            results.append(r)

    # MLP
    if "mlp" in predictions and "reds" in predictions["mlp"]:
        r = check_one("mlp", predictions["mlp"]["reds"], predictions["mlp"].get("blues", []))
        if r:
            results.append(r)

    # 传统策略
    trad = predictions.get("traditional", {})
    if isinstance(trad, dict):
        for sname, sval in trad.items():
            if isinstance(sval, dict) and "reds" in sval:
                r = check_one(f"trad_{sname}", sval["reds"], sval.get("blues", []))
                if r:
                    results.append(r)

    return {
        "lottery": lottery_key,
        "name": name,
        "predict_for_issue": new_draw["issue"],
        "predict_for_date": new_draw.get("date", ""),
        "predicted_after_issue": last_pred_issue,
        "actual_reds": actual_reds,
        "actual_blues": actual_blues,
        "is_repeatable": is_repeatable,
        "check_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }


def main():
    print(f"预测命中检查 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    predictions = load_json(PRED_FILE, default=[])
    if not predictions:
        print("无预测记录，跳过")
        return

    # 加载历史命中记录
    history_hits = load_json(HITS_FILE, default=[])

    # 检查每个彩种
    new_checks = []
    for pred in predictions:
        result = check_lottery(pred, history_hits)
        if result:
            new_checks.append(result)
            print(f"\n[{result['name']}] 预测目标期: {result['predict_for_issue']}")
            print(f"  实际开奖: 红{result['actual_reds']} 蓝{result['actual_blues']}")
            for r in result["results"]:
                tag = ""
                if r["full_match"]:
                    tag = " ★完全命中"
                elif r["full_red_match"]:
                    tag = " ★红球全中"
                elif r["full_blue_match"] and r["blue_total"] > 0:
                    tag = " ★蓝球全中"
                print(f"  [{r['model']}] 预测 红{r['predicted_reds']} 蓝{r['predicted_blues']} "
                      f"-> 红{r['red_hits']}/{r['red_total']} 蓝{r['blue_hits']}/{r['blue_total']}{tag}")

    if not new_checks:
        print("\n无新的开奖期可对比 (上次预测后尚无新开奖)")
        return

    # 永久记录命中 (只记录有命中的)
    added_count = 0
    for check in new_checks:
        # 只记录至少有一个红球命中或蓝球命中的
        has_hit = any(
            r["red_hits"] > 0 or r["blue_hits"] > 0
            for r in check["results"]
        )
        if has_hit:
            history_hits.append(check)
            added_count += 1

    # 保存
    save_json(HITS_FILE, history_hits)
    print(f"\n新增 {added_count} 条命中记录")
    print(f"历史命中记录总数: {len(history_hits)}")
    print(f"已保存到: {HITS_FILE}")


if __name__ == "__main__":
    main()
