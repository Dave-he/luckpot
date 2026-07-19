#!/usr/bin/env python3
"""
根据预测结果生成 README.md
用法: python3 scripts/generate_readme.py
读取: data/predictions.json, data/training_report.json, data/update_report.json
输出: README.md
"""
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fmt_reds(reds):
    """格式化红球: 不足2位补0"""
    return " ".join(f"{int(r):02d}" for r in reds)


def fmt_blues(blues):
    """格式化蓝球"""
    if not blues:
        return "-"
    return " ".join(f"{int(b):02d}" for b in blues)


def load_json(path, default=None):
    if not os.path.exists(path):
        return default if default is not None else []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prediction_table(pred):
    """构建单个彩种的预测结果表格"""
    rows = []
    rows.append(f"### {pred['name']} ({pred['lottery'].upper()})")
    rows.append("")
    rows.append(f"- **最新期号**: {pred.get('latest_issue', '-')}")
    rows.append(f"- **开奖号码**: 红球 `{fmt_reds(pred.get('latest_reds', []))}`"
                + (f" | 蓝球 `{fmt_blues(pred.get('latest_blues', []))}`"
                   if pred.get('latest_blues') else ""))
    rows.append(f"- **历史数据**: {pred.get('history_count', '-')} 期")
    rows.append(f"- **开奖周期**: {pred.get('schedule', '-')}")
    rows.append("")

    predictions = pred.get("predictions", {})

    rows.append("| 模型 / 策略 | 红球预测 | 蓝球预测 | 备注 |")
    rows.append("| --- | --- | --- | --- |")

    # XGBoost
    xgb = predictions.get("xgboost", {})
    if "reds" in xgb:
        info = xgb.get("info", {})
        raw_reds = info.get("raw_red_preds", [])
        note = f"原始预测: {raw_reds}" if raw_reds else ""
        rows.append(f"| XGBoost | `{fmt_reds(xgb['reds'])}` | `{fmt_blues(xgb.get('blues', []))}` | {note} |")
    elif "error" in xgb:
        rows.append(f"| XGBoost | - | - | 错误: {xgb['error']} |")

    # MLP
    mlp = predictions.get("mlp", {})
    if "reds" in mlp:
        info = mlp.get("info", {})
        top_probs = info.get("red_top_probs", [])[:5]
        note = "Top5: " + ", ".join(f"{n}({p})" for n, p in top_probs) if top_probs else ""
        rows.append(f"| MLP神经网络 | `{fmt_reds(mlp['reds'])}` | `{fmt_blues(mlp.get('blues', []))}` | {note} |")
    elif "error" in mlp:
        rows.append(f"| MLP神经网络 | - | - | 错误: {mlp['error']} |")

    # 传统策略
    trad = predictions.get("traditional", {})
    if isinstance(trad, dict):
        strategy_names = {
            "热号推荐": "热号推荐",
            "冷号回补": "冷号回补",
            "综合推荐": "综合推荐",
            "随机机选": "随机机选",
        }
        for key, label in strategy_names.items():
            if key in trad and "reds" in trad[key]:
                rows.append(f"| {label} | `{fmt_reds(trad[key]['reds'])}` | `{fmt_blues(trad[key].get('blues', []))}` | - |")
            elif key in trad and "error" in trad[key]:
                rows.append(f"| {label} | - | - | 错误: {trad[key]['error']} |")

    rows.append("")
    return "\n".join(rows)


def build_training_section(report):
    """构建模型训练状态部分"""
    if not report:
        return ""

    rows = []
    rows.append("## 模型训练状态")
    rows.append("")
    rows.append("| 彩种 | 历史期数 | XGBoost | MLP | XGBoost指标 | MLP准确率 |")
    rows.append("| --- | --- | --- | --- | --- | --- |")

    for item in report:
        name = item.get("name", item.get("lottery", "-"))
        history_count = item.get("history_count", "-")
        xgb = item.get("models", {}).get("xgboost", {})
        mlp = item.get("models", {}).get("mlp", {})
        xgb_status = "✓" if xgb.get("success") else "✗"
        mlp_status = "✓" if mlp.get("success") else "✗"

        xgb_metrics = xgb.get("metrics", {})
        xgb_note = ""
        if xgb_metrics:
            xgb_note = f"MAE={xgb_metrics.get('red_mae', '-')}"
            if "blue_mae" in xgb_metrics:
                xgb_note += f", 蓝球MAE={xgb_metrics['blue_mae']}"

        mlp_metrics = mlp.get("metrics", {})
        mlp_note = ""
        if mlp_metrics:
            mlp_note = f"acc={mlp_metrics.get('red_acc_mean', '-')}"
            if "blue_acc_mean" in mlp_metrics:
                mlp_note += f", 蓝球acc={mlp_metrics['blue_acc_mean']}"

        rows.append(f"| {name} | {history_count} | {xgb_status} | {mlp_status} | {xgb_note} | {mlp_note} |")

    rows.append("")
    return "\n".join(rows)


def build_data_update_section(update_report):
    """构建数据更新部分"""
    if not update_report:
        return ""

    rows = []
    rows.append("## 数据更新状态")
    rows.append("")
    rows.append("| 彩种 | 数据源 | 旧期数 | 新期数 | 新增 | 最新期号 | 状态 |")
    rows.append("| --- | --- | --- | --- | --- | --- | --- |")

    for item in update_report:
        name = item.get("name", item.get("lottery", "-"))
        provider = item.get("provider", "-")
        old_count = item.get("old_count", "-")
        new_count = item.get("new_count", "-")
        added = item.get("added", "-")
        new_latest = item.get("new_latest", "-")
        status = item.get("status", "-")
        status_icon = {"success": "✓", "no_data": "○", "error": "✗"}.get(status, "?")
        rows.append(f"| {name} | {provider} | {old_count} | {new_count} | {added} | {new_latest} | {status_icon} |")

    rows.append("")
    return "\n".join(rows)


def build_backtest_section(backtest_data):
    """构建回测验证部分"""
    if not backtest_data or "results" not in backtest_data:
        return ""

    rows = []
    rows.append("## 回测验证 (历史命中率)")
    rows.append("")
    rows.append(f"> 回测期数: `{backtest_data.get('backtest_count', '-')}` | "
                f"更新时间: `{backtest_data.get('update_time', '-')}`")
    rows.append("")
    rows.append("> 用历史数据回测各算法，**指标: 至少命中1个红球的比例** (50%为达标线)")
    rows.append("")

    # 达标统计
    rows.append("### 各彩种 Top 算法 (按'至少命中1红'准确率)")
    rows.append("")
    rows.append("| 彩种 | Top算法 | 至少1红准确率 | 至少2红准确率 | 红球命中率 | 完全命中 | 达标 |")
    rows.append("| --- | --- | --- | --- | --- | --- | --- |")

    reached_count = 0
    total_count = 0
    for r in backtest_data["results"]:
        name = r["name"]
        results = r.get("results", {})
        if not results:
            continue
        # 找至少1红最高的算法
        top_algo_name = max(results.items(), key=lambda x: x[1].get("at_least_1_red_rate", 0))[0]
        top = results[top_algo_name]
        at_least_1 = top.get("at_least_1_red_rate", 0)
        at_least_2 = top.get("at_least_2_red_rate", 0)
        red_rate = top.get("red_rate", 0)
        full = top.get("full_match", 0)
        count = top.get("count", 0)

        reached = at_least_1 >= 0.5
        if reached:
            reached_count += 1
        total_count += 1
        icon = "✓ 达标" if reached else "✗ 未达标"

        rows.append(f"| {name} | {top_algo_name} | "
                    f"{at_least_1*100:.1f}% ({top.get('at_least_1_red',0)}/{count}) | "
                    f"{at_least_2*100:.1f}% ({top.get('at_least_2_red',0)}/{count}) | "
                    f"{red_rate*100:.1f}% | {full}/{count} | {icon} |")

    rows.append("")
    rows.append(f"**达标统计**: {reached_count}/{total_count} 个彩种的 Top 算法达到 50% 准确率")
    rows.append("")

    # weighted_ensemble 表现
    rows.append("### 权重融合算法 (weighted_ensemble) 各彩种表现")
    rows.append("")
    rows.append("| 彩种 | 至少1红准确率 | 至少2红准确率 | 红球命中率 | 完全命中 |")
    rows.append("| --- | --- | --- | --- | --- |")

    for r in backtest_data["results"]:
        name = r["name"]
        results = r.get("results", {})
        we = results.get("weighted_ensemble")
        if not we:
            continue
        at_least_1 = we.get("at_least_1_red_rate", 0)
        at_least_2 = we.get("at_least_2_red_rate", 0)
        red_rate = we.get("red_rate", 0)
        full = we.get("full_match", 0)
        count = we.get("count", 0)
        rows.append(f"| {name} | {at_least_1*100:.1f}% ({we.get('at_least_1_red',0)}/{count}) | "
                    f"{at_least_2*100:.1f}% ({we.get('at_least_2_red',0)}/{count}) | "
                    f"{red_rate*100:.1f}% | {full}/{count} |")

    rows.append("")
    return "\n".join(rows)


def build_weights_section(weights_data):
    """构建算法权重部分"""
    if not weights_data or "lotteries" not in weights_data:
        return ""

    rows = []
    rows.append("## 动态算法权重")
    rows.append("")
    meta = weights_data.get("_meta", {})
    rows.append(f"> 更新时间: `{meta.get('update_time', '-')}` | "
                f"回测期数: `{meta.get('backtest_count', '-')}` | "
                f"EMA平滑系数: `{meta.get('merge_alpha', 0)}`")
    rows.append("")
    rows.append("> 基于历史回测命中率自动调整各算法权重，权重越高表示该算法在该彩种上表现越好。")
    rows.append("")

    rows.append("### 各彩种 Top 算法 & 命中率")
    rows.append("")
    rows.append("| 彩种 | Top算法 | 权重 | 红球命中率 | 蓝球命中率 | 完全命中 | 是否达标(≥50%) |")
    rows.append("| --- | --- | --- | --- | --- | --- | --- |")

    for key, data in weights_data["lotteries"].items():
        name = data.get("name", key)
        weights = data.get("weights", {})
        scores = data.get("algorithm_scores", {})
        if not weights or not scores:
            continue

        # 找得分最高的算法
        top_algo_name = max(scores.items(), key=lambda x: x[1]["score"])[0]
        top_score = scores[top_algo_name]
        top_weight = weights.get(top_algo_name, 0)
        red_rate = top_score.get("red_rate", 0)
        blue_rate = top_score.get("blue_rate", 0) if data.get("blue_count", 0) > 0 else 0
        full_match = top_score.get("full_match", 0)

        # 综合准确率 (红球命中率为主)
        accuracy = red_rate
        reached = "✓ 达标" if accuracy >= 0.5 else f"○ {accuracy*100:.1f}%"

        rows.append(f"| {name} | {top_algo_name} | {top_weight:.3f} | "
                    f"{red_rate*100:.1f}% | "
                    f"{(blue_rate*100):.1f}%" if data.get("blue_count", 0) > 0
                    else f"| {name} | {top_algo_name} | {top_weight:.3f} | "
                    f"{red_rate*100:.1f}% | - | {full_match} | {reached} |")
        if data.get("blue_count", 0) > 0:
            rows[-1] = (f"| {name} | {top_algo_name} | {top_weight:.3f} | "
                        f"{red_rate*100:.1f}% | {blue_rate*100:.1f}% | "
                        f"{full_match} | {reached} |")

    rows.append("")

    # 权重详情
    rows.append("### 权重详情")
    rows.append("")
    rows.append("| 彩种 | XGBoost | MLP | 热号 | 冷号 | 综合 | 随机 |")
    rows.append("| --- | --- | --- | --- | --- | --- | --- |")

    for key, data in weights_data["lotteries"].items():
        name = data.get("name", key)
        w = data.get("weights", {})

        def fmt(algo):
            v = w.get(algo, 0)
            return f"{v:.3f}" if v > 0 else "-"

        rows.append(f"| {name} | {fmt('xgboost')} | {fmt('mlp')} | "
                    f"{fmt('trad_热号推荐')} | {fmt('trad_冷号回补')} | "
                    f"{fmt('trad_综合推荐')} | {fmt('trad_随机机选')} |")

    rows.append("")
    return "\n".join(rows)


def build_weighted_predictions_section(weighted_preds):
    """构建权重融合预测结果部分"""
    if not weighted_preds:
        return ""

    rows = []
    rows.append("## 权重融合预测 (综合所有算法)")
    rows.append("")
    rows.append("> 基于动态权重融合 XGBoost + MLP + 传统策略的预测结果，理论上比单一算法更稳定。")
    rows.append("")

    rows.append("| 彩种 | 最新期 | 实际开奖 | ★ 权重融合预测 | Top算法 |")
    rows.append("| --- | --- | --- | --- | --- |")

    for p in weighted_preds:
        if "error" in p:
            continue
        name = p.get("name", p.get("lottery"))
        latest = p.get("latest_issue", "-")
        actual_reds = fmt_reds(p.get("latest_reds", []))
        actual_blues = fmt_blues(p.get("latest_blues", []))
        pred = p.get("weighted_prediction", {})
        pred_reds = fmt_reds(pred.get("reds", []))
        pred_blues = fmt_blues(pred.get("blues", []))

        # 找权重最高的算法
        weights = p.get("weights_used", {})
        top_algo = max(weights.items(), key=lambda x: x[1])[0] if weights else "-"

        actual_str = f"红`{actual_reds}`"
        if p.get("latest_blues"):
            actual_str += f" 蓝`{actual_blues}`"
        pred_str = f"红`{pred_reds}`"
        if pred.get("blues"):
            pred_str += f" 蓝`{pred_blues}`"

        rows.append(f"| {name} | {latest} | {actual_str} | {pred_str} | {top_algo} |")

    rows.append("")
    return "\n".join(rows)


def build_hits_section(hits):
    """构建预测命中记录部分"""
    if not hits:
        return ""

    rows = []
    rows.append("## 预测命中记录 (永久保存)")
    rows.append("")
    rows.append(f"> 累计记录 {len(hits)} 次对比结果 (命中即永久存档)")
    rows.append("")

    # 统计各模型命中情况
    model_stats = {}
    for h in hits:
        for r in h.get("results", []):
            m = r["model"]
            if m not in model_stats:
                model_stats[m] = {"total": 0, "red_hits": 0, "blue_hits": 0,
                                  "full_red": 0, "full_blue": 0, "full_match": 0}
            model_stats[m]["total"] += 1
            model_stats[m]["red_hits"] += r["red_hits"]
            model_stats[m]["blue_hits"] += r["blue_hits"]
            if r["full_red_match"]:
                model_stats[m]["full_red"] += 1
            if r["full_blue_match"]:
                model_stats[m]["full_blue"] += 1
            if r["full_match"]:
                model_stats[m]["full_match"] += 1

    rows.append("### 模型命中统计")
    rows.append("")
    rows.append("| 模型 | 对比次数 | 红球命中总数 | 蓝球命中总数 | 红球全中 | 蓝球全中 | 完全命中 |")
    rows.append("| --- | --- | --- | --- | --- | --- | --- |")
    for m, s in sorted(model_stats.items(), key=lambda x: -x[1]["red_hits"]):
        rows.append(f"| {m} | {s['total']} | {s['red_hits']} | {s['blue_hits']} | "
                    f"{s['full_red']} | {s['full_blue']} | {s['full_match']} |")
    rows.append("")

    # 完全命中 / 红球全中 / 蓝球全中的高亮记录
    highlights = []
    for h in hits:
        for r in h.get("results", []):
            if r["full_match"] or r["full_red_match"] or (r["full_blue_match"] and r["blue_total"] > 0):
                tag = []
                if r["full_match"]:
                    tag.append("★完全命中")
                elif r["full_red_match"]:
                    tag.append("★红球全中")
                elif r["full_blue_match"]:
                    tag.append("★蓝球全中")
                highlights.append({
                    "lottery": h.get("name", h.get("lottery")),
                    "issue": h.get("predict_for_issue"),
                    "date": h.get("predict_for_date", ""),
                    "model": r["model"],
                    "predicted_reds": r["predicted_reds"],
                    "predicted_blues": r["predicted_blues"],
                    "actual_reds": r["actual_reds"],
                    "actual_blues": r["actual_blues"],
                    "tag": " ".join(tag),
                    "red_hits": r["red_hits"],
                    "red_total": r["red_total"],
                    "blue_hits": r["blue_hits"],
                    "blue_total": r["blue_total"],
                })

    if highlights:
        rows.append("### 高亮命中 (红球全中 / 蓝球全中 / 完全命中)")
        rows.append("")
        rows.append("| 彩种 | 期号 | 模型 | 预测红球 | 实际红球 | 预测蓝球 | 实际蓝球 | 命中 | 标记 |")
        rows.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for h in highlights:
            rows.append(
                f"| {h['lottery']} | {h['issue']} | {h['model']} | "
                f"`{fmt_reds(h['predicted_reds'])}` | `{fmt_reds(h['actual_reds'])}` | "
                f"`{fmt_blues(h['predicted_blues'])}` | `{fmt_blues(h['actual_blues'])}` | "
                f"红{h['red_hits']}/{h['red_total']} 蓝{h['blue_hits']}/{h['blue_total']} | {h['tag']} |"
            )
        rows.append("")

    # 最近10次对比记录
    rows.append("### 最近对比记录 (最多20条)")
    rows.append("")
    rows.append("| 彩种 | 期号 | 模型 | 预测红球 | 实际红球 | 预测蓝球 | 实际蓝球 | 命中 |")
    rows.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    recent = hits[-20:]
    for h in recent:
        name = h.get("name", h.get("lottery"))
        issue = h.get("predict_for_issue")
        for r in h.get("results", []):
            tag = ""
            if r["full_match"]:
                tag = " ★完全命中"
            elif r["full_red_match"]:
                tag = " ★红球全中"
            elif r["full_blue_match"] and r["blue_total"] > 0:
                tag = " ★蓝球全中"
            rows.append(
                f"| {name} | {issue} | {r['model']} | "
                f"`{fmt_reds(r['predicted_reds'])}` | `{fmt_reds(r['actual_reds'])}` | "
                f"`{fmt_blues(r['predicted_blues'])}` | `{fmt_blues(r['actual_blues'])}` | "
                f"红{r['red_hits']}/{r['red_total']} 蓝{r['blue_hits']}/{r['blue_total']}{tag} |"
            )
    rows.append("")

    return "\n".join(rows)


def main():
    print(f"生成 README - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    predictions = load_json("data/predictions.json", default=[])
    training_report = load_json("data/training_report.json", default=[])
    update_report = load_json("data/update_report.json", default=[])
    hits = load_json("data/prediction_hits.json", default=[])
    weights_data = load_json("data/algorithm_weights.json", default={})
    weighted_preds = load_json("data/weighted_predictions.json", default=[])
    backtest_data = load_json("data/backtest_report.json", default={})

    # 模型文件统计
    model_stats = []
    for key in ["ssq", "dlt", "fc3d", "qlc", "qxc", "pls", "plw"]:
        for mtype in ["xgboost", "mlp"]:
            path = f"data/{key}/models/{mtype}/"
            if os.path.exists(path):
                files = os.listdir(path)
                model_stats.append(f"`{key}/{mtype}` ({len(files)}文件)")

    sections = []

    # 标题
    sections.append("# 彩票号码预测系统")
    sections.append("")
    sections.append("> 基于历史数据 + XGBoost + 神经网络(MLP) + 传统统计策略的彩票号码预测系统")
    sections.append("")
    sections.append("[![每日预测](https://github.com/Dave-he/luckpot/actions/workflows/daily.yml/badge.svg)](https://github.com/Dave-he/luckpot/actions/workflows/daily.yml)")
    sections.append("")
    sections.append("系统每日通过 GitHub Actions 自动运行：抓取最新数据 → 训练模型 → 生成预测 → 推送更新。")
    sections.append("")

    # 预测时间
    if predictions:
        predict_time = predictions[0].get("predict_time", "-")
    else:
        predict_time = "-"
    sections.append(f"**最近预测时间**: `{predict_time}`")
    sections.append("")

    # 下一期预测结果 - 主要内容
    sections.append("## 下一期预测结果")
    sections.append("")
    sections.append("> ⚠️ **免责声明**: 彩票开奖完全随机，本系统预测仅供学习研究，不构成任何投资建议。请理性购彩。")
    sections.append("")

    for pred in predictions:
        sections.append(build_prediction_table(pred))

    # 权重融合预测 (放在普通预测之后)
    weighted_section = build_weighted_predictions_section(weighted_preds)
    if weighted_section:
        sections.append(weighted_section)

    # 动态算法权重
    weights_section = build_weights_section(weights_data)
    if weights_section:
        sections.append(weights_section)

    # 回测验证
    backtest_section = build_backtest_section(backtest_data)
    if backtest_section:
        sections.append(backtest_section)

    # 预测命中记录
    hits_section = build_hits_section(hits)
    if hits_section:
        sections.append(hits_section)

    # 模型训练状态
    train_section = build_training_section(training_report)
    if train_section:
        sections.append(train_section)

    # 数据更新状态
    update_section = build_data_update_section(update_report)
    if update_section:
        sections.append(update_section)

    # 支持的彩种
    sections.append("## 支持的彩种")
    sections.append("")
    sections.append("| 彩种 | 代码 | 红球 | 蓝球 | 开奖周期 |")
    sections.append("| --- | --- | --- | --- | --- |")
    lottery_info = [
        ("双色球", "ssq", "6个 (1-33)", "1个 (1-16)", "每周二、四、日"),
        ("大乐透", "dlt", "5个 (1-35)", "2个 (1-12)", "每周一、三、六"),
        ("福彩3D", "fc3d", "3个 (0-9)", "无", "每日开奖"),
        ("七乐彩", "qlc", "7个 (1-30)", "1个 (1-30)", "每周一、三、五"),
        ("七星彩", "qxc", "7个 (0-9)", "无", "每周二、五、日"),
        ("排列3", "pls", "3个 (0-9)", "无", "每日开奖"),
        ("排列5", "plw", "5个 (0-9)", "无", "每日开奖"),
    ]
    for name, code, red, blue, schedule in lottery_info:
        sections.append(f"| {name} | {code} | {red} | {blue} | {schedule} |")
    sections.append("")

    # 数据源
    sections.append("## 数据源")
    sections.append("")
    sections.append("- **cwl.gov.cn** - 中国福利彩票(双色球/福彩3D/七乐彩)")
    sections.append("- **webapi.sporttery.cn** - 中国体育彩票(大乐透/七星彩/排列3/排列5)")
    sections.append("- **500.com** - 备用数据源")
    sections.append("")

    # 预测模型
    sections.append("## 预测模型")
    sections.append("")
    sections.append("### 1. XGBoost 回归模型")
    sections.append("- 对每个号码位置训练一个 XGBoost 回归器")
    sections.append("- 特征: 历史 20 期号码、和值、奇偶比、跨度、频率统计")
    sections.append("- 输出: 每个位置的号码预测值(回归到合法范围)")
    sections.append("")
    sections.append("### 2. MLP 神经网络 (纯 numpy 实现)")
    sections.append("- 三层全连接网络: 输入 → 128 → 128 → 输出")
    sections.append("- 激活函数: ReLU + Softmax")
    sections.append("- 损失函数: 多分类交叉熵")
    sections.append("- 输出: 每个号码位置的概率分布")
    sections.append("- 不依赖 tensorflow/pytorch，部署轻量")
    sections.append("")
    sections.append("### 3. 传统统计策略")
    sections.append("- **热号推荐**: 基于最近 30 期号码频率")
    sections.append("- **冷号回补**: 基于号码遗漏值")
    sections.append("- **综合推荐**: 频率 + 遗漏 + 转移概率综合打分")
    sections.append("- **随机机选**: 加权随机采样(参考综合评分)")
    sections.append("")

    # 项目结构
    sections.append("## 项目结构")
    sections.append("")
    sections.append("```")
    sections.append("lottery/")
    sections.append("├── config.py             # 7个彩种配置")
    sections.append("├── spiders/              # 数据爬虫 (cwl/sporttery/data500)")
    sections.append("├── data/                 # 数据加载与处理")
    sections.append("├── analysis/             # 频率/统计分析")
    sections.append("└── models/               # 预测模型")
    sections.append("    ├── predictor.py      # 传统策略预测器")
    sections.append("    ├── xgboost_model.py   # XGBoost 模型")
    sections.append("    └── mlp_model.py       # MLP 神经网络模型")
    sections.append("scripts/")
    sections.append("├── update_data.py        # 数据抓取脚本")
    sections.append("├── train_models.py       # 模型训练脚本")
    sections.append("├── predict.py            # 预测脚本")
    sections.append("├── check_hits.py         # 预测命中检查脚本")
    sections.append("├── update_weights.py     # 动态算法权重更新脚本")
    sections.append("├── weighted_predict.py   # 权重融合预测脚本")
    sections.append("├── backtest.py           # 历史回测验证脚本")
    sections.append("└── generate_readme.py    # README 生成脚本")
    sections.append("data/")
    sections.append("├── {lottery}/history.csv # 各彩种历史数据")
    sections.append("├── {lottery}/models/     # 训练好的模型")
    sections.append("├── predictions.json      # 当前各算法预测结果")
    sections.append("├── weighted_predictions.json # 权重融合预测结果")
    sections.append("├── algorithm_weights.json # 动态算法权重 (永久保存)")
    sections.append("├── prediction_hits.json  # 预测命中记录 (永久保存)")
    sections.append("├── backtest_report.json  # 历史回测报告")
    sections.append("├── training_report.json  # 训练报告")
    sections.append("└── update_report.json    # 数据更新报告")
    sections.append(".github/workflows/daily.yml # GitHub Actions 每日定时任务")
    sections.append("```")
    sections.append("")

    # 自动化流程
    sections.append("## 自动化流程")
    sections.append("")
    sections.append("GitHub Actions 每天北京时间 08:00 自动执行:")
    sections.append("")
    sections.append("1. **抓取数据** - `python3 scripts/update_data.py`")
    sections.append("2. **训练模型** - `python3 scripts/train_models.py`")
    sections.append("3. **检查上次预测命中** - `python3 scripts/check_hits.py` (永久记录到 prediction_hits.json)")
    sections.append("4. **更新动态权重** - `python3 scripts/update_weights.py` (基于历史回测)")
    sections.append("5. **生成新预测** - `python3 scripts/predict.py`")
    sections.append("6. **权重融合预测** - `python3 scripts/weighted_predict.py` (综合所有算法)")
    sections.append("7. **历史回测验证** - `python3 scripts/backtest.py` (每周一/四)")
    sections.append("8. **更新 README** - `python3 scripts/generate_readme.py`")
    sections.append("9. **提交推送** - 自动 commit 并 push 到 GitHub")
    sections.append("")
    sections.append("也可手动在 GitHub Actions 页面触发运行。")
    sections.append("")

    # 本地运行
    sections.append("## 本地运行")
    sections.append("")
    sections.append("```bash")
    sections.append("# 安装依赖")
    sections.append("pip install -r requirements.txt")
    sections.append("")
    sections.append("# 抓取最新数据")
    sections.append("python3 scripts/update_data.py")
    sections.append("")
    sections.append("# 训练所有模型")
    sections.append("python3 scripts/train_models.py")
    sections.append("")
    sections.append("# 训练单个彩种(如双色球)")
    sections.append("python3 scripts/train_models.py ssq")
    sections.append("")
    sections.append("# 生成预测")
    sections.append("python3 scripts/predict.py")
    sections.append("")
    sections.append("# 检查上次预测命中情况 (永久记录命中)")
    sections.append("python3 scripts/check_hits.py")
    sections.append("")
    sections.append("# 更新动态算法权重 (基于历史回测)")
    sections.append("python3 scripts/update_weights.py --backtest 20")
    sections.append("")
    sections.append("# 权重融合预测 (综合所有算法)")
    sections.append("python3 scripts/weighted_predict.py")
    sections.append("")
    sections.append("# 历史回测验证")
    sections.append("python3 scripts/backtest.py --n 30")
    sections.append("")
    sections.append("# 生成 README")
    sections.append("python3 scripts/generate_readme.py")
    sections.append("")
    sections.append("# 交互式菜单")
    sections.append("python3 main.py")
    sections.append("```")
    sections.append("")

    # 免责声明
    sections.append("## ⚠️ 免责声明")
    sections.append("")
    sections.append("本项目仅用于**学习研究目的**，旨在探索数据抓取、机器学习、自动化运维等技术。")
    sections.append("")
    sections.append("- 彩票开奖是完全随机事件，任何预测模型都无法真正提高中奖概率")
    sections.append("- 历史数据不能预测未来的随机事件")
    sections.append("- 请理性购彩，量力而行，切勿沉迷")
    sections.append("- 本项目作者不对任何购彩损失负责")
    sections.append("")

    # 写入文件
    content = "\n".join(sections)
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(content)
    print(f"README.md 已生成 ({len(content)} 字符)")


if __name__ == "__main__":
    main()
