#!/usr/bin/env python3
"""
模型训练脚本 - 训练所有彩种的所有模型
- XGBoost, MLP, Random Forest, Markov, Naive Bayes, Monte Carlo, K-Means, LSTM
- Stacking 元学习器 (基于基础模型)

用法: python3 scripts/train_models.py [lottery_key] [--skip-lstm]
不指定lottery_key则训练所有彩种
"""
import sys
import os
import time
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lottery.config import LOTTERY_CONFIGS
from lottery.data import DataLoader
from lottery.models import (
    XGBoostPredictor, MLPredictor,
    RandomForestPredictor, MarkovPredictor,
    NaiveBayesPredictor, MonteCarloPredictor,
    KMeansPredictor, LSTMPredictor,
)


def get_model_dir(config, model_type):
    """获取模型保存目录"""
    return os.path.join(os.path.dirname(config["data_file"]), "models", model_type)


def train_one_model(name, predictor_cls, config, history, save_dir, **kwargs):
    """训练单个模型"""
    print(f"  训练 {name} ...")
    t0 = time.time()
    try:
        predictor = predictor_cls(config)
        # LSTM 用最近500期避免太慢
        if predictor_cls == LSTMPredictor:
            train_data = history[-500:] if len(history) > 500 else history
            predictor.epochs = 20  # 减少epoch加速
        else:
            train_data = history

        train_result = predictor.train(train_data, **kwargs) if "samples_for_meta" in kwargs else predictor.train(train_data)
        elapsed = time.time() - t0

        if train_result.get("success"):
            predictor.save(save_dir)
            print(f"  ✓ {name} 完成 ({elapsed:.1f}s), 指标: {train_result.get('metrics', {})}")
            return {
                "success": True,
                "elapsed": round(elapsed, 2),
                "metrics": train_result.get("metrics", {}),
                "samples": train_result.get("samples", 0),
            }
        else:
            print(f"  ✗ {name} 失败: {train_result.get('error')}")
            return {"success": False, "error": train_result.get("error")}
    except Exception as e:
        print(f"  ✗ {name} 异常: {e}")
        return {"success": False, "error": str(e)}


def train_lottery(lottery_key: str, config: dict, skip_lstm: bool = False) -> dict:
    """训练指定彩种的所有模型"""
    name = config["name"]
    print(f"\n{'='*60}")
    print(f"训练 {name} ({lottery_key})")
    print(f"{'='*60}")

    loader = DataLoader(config)
    history = loader.load_history()
    if not history or len(history) < 50:
        print(f"  数据不足 ({len(history)} 期)，跳过")
        return {"lottery": lottery_key, "skipped": True, "reason": "数据不足"}

    print(f"  历史数据: {len(history)} 期")
    print(f"  最新期: {history[-1]['issue']}")

    results = {
        "lottery": lottery_key,
        "name": name,
        "history_count": len(history),
        "latest_issue": history[-1]["issue"],
        "models": {},
    }

    # 基础模型列表
    base_models = [
        ("xgboost", XGBoostPredictor),
        ("mlp", MLPredictor),
        ("random_forest", RandomForestPredictor),
        ("markov", MarkovPredictor),
        ("naive_bayes", NaiveBayesPredictor),
        ("monte_carlo", MonteCarloPredictor),
        ("kmeans", KMeansPredictor),
    ]
    if not skip_lstm:
        base_models.append(("lstm", LSTMPredictor))

    for model_type, cls in base_models:
        save_dir = get_model_dir(config, model_type)
        r = train_one_model(model_type, cls, config, history, save_dir)
        results["models"][model_type] = r

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("lottery_key", nargs="?", default="",
                        help="只训练指定彩种 (默认全部)")
    parser.add_argument("--skip-lstm", action="store_true",
                        help="跳过LSTM (训练太慢时用)")
    args = parser.parse_args()

    if args.lottery_key and args.lottery_key in LOTTERY_CONFIGS:
        keys_to_train = [args.lottery_key]
    else:
        keys_to_train = list(LOTTERY_CONFIGS.keys())

    print(f"准备训练 {len(keys_to_train)} 个彩种: {', '.join(keys_to_train)}")
    if args.skip_lstm:
        print("  (跳过 LSTM)")

    all_results = []
    for key in keys_to_train:
        config = LOTTERY_CONFIGS[key]
        result = train_lottery(key, config, skip_lstm=args.skip_lstm)
        all_results.append(result)

    # 保存训练报告
    report_path = os.path.join("data", "training_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n训练报告已保存: {report_path}")

    # 汇总
    print(f"\n{'='*80}")
    print("训练完成汇总:")
    print(f"{'='*80}")
    model_types = ["xgboost", "mlp", "random_forest", "markov",
                   "naive_bayes", "monte_carlo", "kmeans", "lstm"]
    header = f"{'彩种':<10} " + " ".join(f"{m[:8]:<10}" for m in model_types)
    print(header)
    print("-" * len(header))
    for r in all_results:
        if r.get("skipped"):
            print(f"  {r['lottery']}: 跳过 ({r['reason']})")
            continue
        line = f"{r['name']:<10} "
        for m in model_types:
            ok = r["models"].get(m, {}).get("success", False)
            line += f"{'✓' if ok else '✗':<10} "
        print(line)


if __name__ == "__main__":
    main()
