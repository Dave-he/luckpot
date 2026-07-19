#!/usr/bin/env python3
"""
模型训练脚本 - 训练所有彩种的XGBoost和MLP模型
用法: python3 scripts/train_models.py [lottery_key]
不指定lottery_key则训练所有彩种
"""
import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lottery.config import LOTTERY_CONFIGS
from lottery.data import DataLoader
from lottery.models import XGBoostPredictor, MLPredictor


def get_model_dir(config, model_type):
    """获取模型保存目录"""
    return os.path.join(os.path.dirname(config["data_file"]), "models", model_type)


def train_lottery(lottery_key: str, config: dict) -> dict:
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

    # 训练XGBoost
    print(f"\n  [1/2] 训练 XGBoost ...")
    t0 = time.time()
    try:
        xgb_pred = XGBoostPredictor(config)
        train_result = xgb_pred.train(history)
        elapsed = time.time() - t0

        if train_result.get("success"):
            model_dir = get_model_dir(config, "xgboost")
            xgb_pred.save(model_dir)
            print(f"  ✓ XGBoost训练完成 ({elapsed:.1f}s), 指标: {train_result['metrics']}")
            results["models"]["xgboost"] = {
                "success": True,
                "elapsed": round(elapsed, 2),
                "metrics": train_result["metrics"],
                "samples": train_result["samples"],
            }
        else:
            print(f"  ✗ XGBoost训练失败: {train_result.get('error')}")
            results["models"]["xgboost"] = {"success": False, "error": train_result.get("error")}
    except Exception as e:
        print(f"  ✗ XGBoost异常: {e}")
        results["models"]["xgboost"] = {"success": False, "error": str(e)}

    # 训练MLP
    print(f"\n  [2/2] 训练 MLP (神经网络) ...")
    t0 = time.time()
    try:
        mlp_pred = MLPredictor(config)
        train_result = mlp_pred.train(history)
        elapsed = time.time() - t0

        if train_result.get("success"):
            model_dir = get_model_dir(config, "mlp")
            mlp_pred.save(model_dir)
            print(f"  ✓ MLP训练完成 ({elapsed:.1f}s), 指标: {train_result['metrics']}")
            results["models"]["mlp"] = {
                "success": True,
                "elapsed": round(elapsed, 2),
                "metrics": train_result["metrics"],
                "samples": train_result["samples"],
            }
        else:
            print(f"  ✗ MLP训练失败: {train_result.get('error')}")
            results["models"]["mlp"] = {"success": False, "error": train_result.get("error")}
    except Exception as e:
        print(f"  ✗ MLP异常: {e}")
        results["models"]["mlp"] = {"success": False, "error": str(e)}

    return results


def main():
    args = sys.argv[1:]
    if args and args[0] in LOTTERY_CONFIGS:
        keys_to_train = [args[0]]
    else:
        keys_to_train = list(LOTTERY_CONFIGS.keys())

    print(f"准备训练 {len(keys_to_train)} 个彩种: {', '.join(keys_to_train)}")

    all_results = []
    for key in keys_to_train:
        config = LOTTERY_CONFIGS[key]
        result = train_lottery(key, config)
        all_results.append(result)

    # 保存训练报告
    report_path = os.path.join("data", "training_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n训练报告已保存: {report_path}")

    # 汇总
    print(f"\n{'='*60}")
    print("训练完成汇总:")
    print(f"{'='*60}")
    for r in all_results:
        if r.get("skipped"):
            print(f"  {r['lottery']}: 跳过 ({r['reason']})")
            continue
        xgb_ok = r["models"].get("xgboost", {}).get("success", False)
        mlp_ok = r["models"].get("mlp", {}).get("success", False)
        print(f"  {r['name']:<8}: XGBoost={'✓' if xgb_ok else '✗'} MLP={'✓' if mlp_ok else '✗'}")


if __name__ == "__main__":
    main()
