#!/usr/bin/env python3
"""
生成 docs/prediction-history.md - 历次预测的完整归档

数据来源: data/prediction_history.json (由 scripts/update_history.py 维护)
用法: python3 scripts/generate_history.py
"""
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HISTORY_FILE = os.path.join("data", "prediction_history.json")
OUTPUT_FILE = os.path.join("docs", "prediction-history.md")


def fmt_reds(reds):
    return " ".join(f"{int(r):02d}" for r in reds) if reds else "-"


def fmt_blues(blues):
    if not blues:
        return "-"
    return " ".join(f"{int(b):02d}" for b in blues)


def load_json(path, default=None):
    if not os.path.exists(path):
        return default if default is not None else []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_overall_stats(settled):
    """全局模型命中统计"""
    stats = {}
    for h in settled:
        for m in h.get("models", []):
            name = m["model"]
            s = stats.setdefault(name, {
                "total": 0, "red_hits": 0, "blue_hits": 0,
                "red_total": 0, "blue_total": 0,
                "full_red": 0, "full_blue": 0, "full_match": 0,
                "at_least_1": 0,
            })
            s["total"] += 1
            s["red_hits"] += m.get("red_hits", 0)
            s["blue_hits"] += m.get("blue_hits", 0)
            s["red_total"] += m.get("red_total", 0)
            s["blue_total"] += m.get("blue_total", 0)
            if m.get("full_red"):
                s["full_red"] += 1
            if m.get("full_blue"):
                s["full_blue"] += 1
            if m.get("full_match"):
                s["full_match"] += 1
            if m.get("red_hits", 0) >= 1:
                s["at_least_1"] += 1
    return stats


def main():
    print(f"生成预测历史 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    history = load_json(HISTORY_FILE, default=[])
    settled = [h for h in history if h.get("status") == "settled"]
    pending = [h for h in history if h.get("status") == "pending"]

    s = []
    s.append("# 预测历史归档")
    s.append("")
    s.append("> 本文件由 GitHub Actions 自动生成，记录每一期预测的完整明细与实际开奖结果。")
    s.append(">")
    s.append("> 每期开奖后自动回填开奖号码并计算命中，新一期预测生成时自动追加归档。")
    s.append("")
    s.append(f"**最后更新**: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    s.append("")
    s.append(f"**统计概览**: 累计归档 `{len(history)}` 期 | 已开奖 `{len(settled)}` 期 | 待开奖 `{len(pending)}` 期")
    s.append("")
    s.append("[← 返回 README 查看最新预测](../README.md)")
    s.append("")

    if not history:
        s.append("暂无归档数据。")
        s.append("")
        content = "\n".join(s)
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"已生成: {OUTPUT_FILE}")
        return

    # ---- 全局模型统计 ----
    if settled:
        stats = build_overall_stats(settled)
        s.append("## 模型命中统计 (全部已开奖期)")
        s.append("")
        s.append("| 模型 | 参与期数 | 红球命中 | 命中率 | 至少中1红 | 红球全中 | 完全命中 |")
        s.append("| --- | --- | --- | --- | --- | --- | --- |")
        for name, v in sorted(stats.items(), key=lambda x: -x[1]["red_hits"]):
            rate = (v["red_hits"] / v["red_total"] * 100) if v["red_total"] else 0
            at1 = (v["at_least_1"] / v["total"] * 100) if v["total"] else 0
            s.append(f"| {name} | {v['total']} | {v['red_hits']}/{v['red_total']} | "
                     f"{rate:.1f}% | {at1:.1f}% ({v['at_least_1']}/{v['total']}) | "
                     f"{v['full_red']} | {v['full_match']} |")
        s.append("")

    # ---- 待开奖 ----
    if pending:
        s.append("## 待开奖预测")
        s.append("")
        s.append("| 彩种 | 目标期号 | 预测时间 | 权重融合预测 | 状态 |")
        s.append("| --- | --- | --- | --- | --- |")
        for h in pending:
            ensemble = next((m for m in h.get("models", []) if m["model"] == "权重融合"), None)
            pred_str = "-"
            if ensemble:
                pred_str = f"`{fmt_reds(ensemble['reds'])}`"
                if ensemble.get("blues"):
                    pred_str += f" + `{fmt_blues(ensemble['blues'])}`"
            s.append(f"| {h.get('name', h['lottery'])} | **{h['target_issue']}** | "
                     f"{h.get('predict_time', '-')} | {pred_str} | ⏳ 待开奖 |")
        s.append("")

    # ---- 已开奖明细 ----
    if settled:
        s.append("## 已开奖明细")
        s.append("")
        # 按彩种分组
        by_lottery = {}
        for h in settled:
            by_lottery.setdefault(h.get("name", h["lottery"]), []).append(h)

        for name, entries in by_lottery.items():
            s.append(f"### {name}")
            s.append("")
            s.append("| 期号 | 开奖日期 | 实际开奖 | 模型 | 预测号码 | 命中 |")
            s.append("| --- | --- | --- | --- | --- | --- |")
            for h in entries:
                actual = fmt_reds(h.get("actual_reds", []))
                if h.get("actual_blues"):
                    actual += f" + {fmt_blues(h['actual_blues'])}"
                for i, m in enumerate(h.get("models", [])):
                    pred = fmt_reds(m["reds"])
                    if m.get("blues"):
                        pred += f" + {fmt_blues(m['blues'])}"
                    hits = f"红{m.get('red_hits', 0)}/{m.get('red_total', 0)}"
                    if m.get("blue_total"):
                        hits += f" 蓝{m.get('blue_hits', 0)}/{m['blue_total']}"
                    tag = ""
                    if m.get("full_match"):
                        tag = " ★完全命中"
                    elif m.get("full_red"):
                        tag = " ★红球全中"
                    elif m.get("full_blue") and m.get("blue_total"):
                        tag = " ★蓝球全中"
                    # 期号/日期只在第一行显示
                    issue_cell = f"**{h['target_issue']}**" if i == 0 else ""
                    date_cell = h.get("draw_date", "-") if i == 0 else ""
                    actual_cell = f"`{actual}`" if i == 0 else ""
                    s.append(f"| {issue_cell} | {date_cell} | {actual_cell} | {m['model']} | "
                             f"`{pred}` | {hits}{tag} |")
            s.append("")

    s.append("---")
    s.append("")
    s.append("> ⚠️ **免责声明**: 彩票开奖完全随机，本文档仅供学习研究，不构成任何投资建议。")
    s.append("")

    content = "\n".join(s)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"已生成: {OUTPUT_FILE} ({len(content)} 字符, {len(history)} 期归档)")


if __name__ == "__main__":
    main()
