#!/usr/bin/env python3
"""
数据更新脚本 - 抓取所有彩种最新数据
用法: python3 scripts/update_data.py
"""
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lottery.config import LOTTERY_CONFIGS
from lottery.spiders import get_provider
from lottery.data import DataLoader


# 主数据源失败时的降级顺序。
# GitHub Actions 的出口 IP 常被 cwl.gov.cn / webapi.sporttery.cn 的 WAF 拦截
# (403 / 567), 而 500.com 始终可用, 因此把它作为通用兜底源。
FALLBACK_PROVIDERS = ["data500"]


def _fetch_with_fallback(lottery_key: str, config: dict) -> tuple:
    """依次尝试主数据源与备用数据源, 返回 (records, used_provider)"""
    chain = [config["provider"]] + [
        p for p in FALLBACK_PROVIDERS if p != config["provider"]
    ]

    for provider_name in chain:
        try:
            provider = get_provider(provider_name)
            # 备用源用自己的彩种 key, 主源用 provider_code
            code = lottery_key if provider_name == "data500" else config["provider_code"]
            records = provider.fetch_all(code)
        except Exception as e:
            print(f"  [{provider_name}] 抓取异常: {e}")
            continue

        if records:
            return records, provider_name
        print(f"  [{provider_name}] 无数据, 尝试下一个数据源")

    return [], None


def update_lottery(lottery_key: str, config: dict) -> dict:
    """更新指定彩种的数据"""
    name = config["name"]
    result = {
        "lottery": lottery_key,
        "name": name,
        "provider": config["provider"],
        "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    loader = DataLoader(config)
    old_history = loader.load_history()
    old_latest = old_history[-1]["issue"] if old_history else None
    result["old_count"] = len(old_history)
    result["old_latest"] = old_latest

    try:
        records, used_provider = _fetch_with_fallback(lottery_key, config)
        result["used_provider"] = used_provider or config["provider"]
        if used_provider and used_provider != config["provider"]:
            print(f"  ⚠ 主源 {config['provider']} 不可用, 已降级到 {used_provider}")

        if not records:
            result["status"] = "no_data"
            return result

        loader.save_data(records, append=True)
        new_history = loader.load_history()
        new_latest = new_history[-1]["issue"] if new_history else None

        result["new_count"] = len(new_history)
        result["new_latest"] = new_latest
        result["added"] = len(new_history) - len(old_history)
        result["status"] = "success"

        if records:
            last = records[-1]
            result["latest_reds"] = last["reds"]
            result["latest_blues"] = last["blues"]

        print(f"  ✓ {name}: {len(old_history)} -> {len(new_history)} 期 (新增 {result['added']} 期, 最新 {new_latest})")
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        print(f"  ✗ {name}: {e}")

    return result


def main():
    print(f"数据更新脚本启动 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"准备更新 {len(LOTTERY_CONFIGS)} 个彩种\n")

    all_results = []
    for key, config in LOTTERY_CONFIGS.items():
        print(f"更新 {config['name']} ({key}) ...")
        result = update_lottery(key, config)
        all_results.append(result)

    # 保存更新报告
    report_path = os.path.join("data", "update_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n更新报告已保存: {report_path}")

    # 汇总
    print(f"\n{'='*60}")
    print("数据更新汇总:")
    for r in all_results:
        status_icon = "✓" if r.get("status") == "success" else "✗"
        added = r.get("added", 0)
        latest = r.get("new_latest", "-")
        print(f"  {status_icon} {r['name']:<8}: 新增{added}期, 最新={latest}")


if __name__ == "__main__":
    main()
