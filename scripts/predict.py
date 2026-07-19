#!/usr/bin/env python3
"""
预测脚本 - 使用训练好的所有模型预测下一期号码，输出JSON
- XGBoost, MLP, Random Forest, Markov, Naive Bayes, Monte Carlo, K-Means, LSTM
- 传统策略 (热号/冷号/综合/转移概率/随机)

用法: python3 scripts/predict.py
输出: data/predictions.json
"""
import sys
import os
import json
from datetime import datetime

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


def get_model_dir(config, model_type):
    return os.path.join(os.path.dirname(config["data_file"]), "models", model_type)


def predict_with_model(model_name, predictor_cls, config, history):
    """用单个模型预测 (加载已训练模型)"""
    try:
        predictor = predictor_cls(config)
        model_dir = get_model_dir(config, model_name)
        if predictor.load(model_dir):
            reds, blues, info = predictor.predict(history)
            return {
                "reds": reds,
                "blues": blues,
                "info": info,
            }
        else:
            return {"error": f"模型未训练: {model_name}"}
    except Exception as e:
        return {"error": str(e)}


def predict_lottery(lottery_key: str, config: dict) -> dict:
    """对指定彩种进行预测"""
    name = config["name"]
    result = {
        "lottery": lottery_key,
        "name": name,
        "schedule": config["schedule"],
        "provider": config["provider"],
        "predict_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    loader = DataLoader(config)
    history = loader.load_history()
    if not history or len(history) < 50:
        result["error"] = f"数据不足 ({len(history)} 期)"
        return result

    last = history[-1]
    result["latest_issue"] = last["issue"]
    result["latest_reds"] = last["reds"]
    result["latest_blues"] = last["blues"]
    result["history_count"] = len(history)

    predictions = {}

    # 所有机器学习/统计模型
    ml_models = [
        ("xgboost", XGBoostPredictor),
        ("mlp", MLPredictor),
        ("random_forest", RandomForestPredictor),
        ("markov", MarkovPredictor),
        ("naive_bayes", NaiveBayesPredictor),
        ("monte_carlo", MonteCarloPredictor),
        ("kmeans", KMeansPredictor),
        ("lstm", LSTMPredictor),
    ]

    for model_name, cls in ml_models:
        predictions[model_name] = predict_with_model(model_name, cls, config, history)

    # 传统策略预测
    try:
        traditional = LotteryPredictor(config)
        trad_results = traditional.predict_multi_strategy(history)
        predictions["traditional"] = {
            strategy: {"reds": list(reds), "blues": list(blues)}
            for strategy, (reds, blues) in trad_results.items()
        }
    except Exception as e:
        predictions["traditional"] = {"error": str(e)}

    result["predictions"] = predictions
    return result


def main():
    print(f"预测脚本启动 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_predictions = []
    for key, config in LOTTERY_CONFIGS.items():
        print(f"\n预测 {config['name']} ({key}) ...")
        pred = predict_lottery(key, config)

        if "latest_issue" in pred:
            print(f"  最新期: {pred['latest_issue']} 红球={pred['latest_reds']} 蓝球={pred['latest_blues']}")

        for model_name, model_pred in pred.get("predictions", {}).items():
            if "reds" in model_pred:
                reds = model_pred["reds"]
                blues = model_pred.get("blues", [])
                print(f"  [{model_name}] 红球={reds} 蓝球={blues}")
            elif "error" in model_pred:
                print(f"  [{model_name}] 错误: {model_pred['error']}")
            elif isinstance(model_pred, dict):
                for sname, sval in model_pred.items():
                    if isinstance(sval, dict) and "reds" in sval:
                        print(f"  [{model_name}/{sname}] 红球={sval['reds']} 蓝球={sval.get('blues', [])}")

        all_predictions.append(pred)

    # 保存预测结果
    output_path = os.path.join("data", "predictions.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, ensure_ascii=False, indent=2)
    print(f"\n预测结果已保存: {output_path}")


if __name__ == "__main__":
    main()
