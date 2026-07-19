#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lottery.config import LOTTERY_CONFIGS
from lottery.spiders import get_provider
from lottery.data import DataLoader, DataProcessor
from lottery.analysis import FrequencyAnalyzer, StatisticsAnalyzer
from lottery.models import LotteryPredictor


def color_red(text: str) -> str:
    return f"\033[91m{text}\033[0m"


def color_blue(text: str) -> str:
    return f"\033[94m{text}\033[0m"


def color_yellow(text: str) -> str:
    return f"\033[93m{text}\033[0m"


def color_green(text: str) -> str:
    return f"\033[92m{text}\033[0m"


def color_cyan(text: str) -> str:
    return f"\033[96m{text}\033[0m"


def format_numbers(reds, blues, config):
    red_str = " ".join(color_red(f"{n:02d}") for n in sorted(reds))
    if config["blue_count"] > 0 and blues:
        blue_str = " ".join(color_blue(f"{n:02d}") for n in sorted(blues))
        return f"{red_str} + {blue_str}"
    return red_str


def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


class LotteryApp:
    def __init__(self):
        self.lottery_key = "ssq"
        self.configs = LOTTERY_CONFIGS
        self.loaders = {k: DataLoader(c) for k, c in self.configs.items()}
        self.processors = {k: DataProcessor(c) for k, c in self.configs.items()}
        self.freq_analyzers = {k: FrequencyAnalyzer(c) for k, c in self.configs.items()}
        self.stat_analyzers = {k: StatisticsAnalyzer(c) for k, c in self.configs.items()}
        self.predictors = {k: LotteryPredictor(c) for k, c in self.configs.items()}

    @property
    def config(self):
        return self.configs[self.lottery_key]

    @property
    def loader(self):
        return self.loaders[self.lottery_key]

    @property
    def processor(self):
        return self.processors[self.lottery_key]

    @property
    def freq_analyzer(self):
        return self.freq_analyzers[self.lottery_key]

    @property
    def stat_analyzer(self):
        return self.stat_analyzers[self.lottery_key]

    @property
    def predictor(self):
        return self.predictors[self.lottery_key]

    def load_history(self):
        return self.loader.load_history()

    def select_lottery(self):
        print_header("选择彩票类型")
        items = list(self.configs.items())
        for i, (key, cfg) in enumerate(items, 1):
            current = "✓" if key == self.lottery_key else " "
            provider_tag = color_cyan(f"[{cfg['provider']}]")
            print(f"  {i}. {cfg['name']:<8} {provider_tag} {cfg['schedule']} (当前: {current})")

        choice = input(f"\n请选择 (1-{len(items)}, 默认1): ").strip()
        try:
            idx = int(choice) - 1 if choice else 0
            if 0 <= idx < len(items):
                self.lottery_key = items[idx][0]
                print(f"\n已切换到: {color_yellow(self.config['name'])} (数据源: {color_cyan(self.config['provider'])})")
        except ValueError:
            print(color_yellow("无效选择"))

    def crawl_data(self):
        print_header(f"抓取{self.config['name']}最新数据")
        latest = self.loader.get_latest_issue()
        if latest:
            print(f"本地最新期号: {color_green(latest)}")

        provider = get_provider(self.config["provider"])
        print(f"数据源: {color_cyan(self.config['provider'])}")
        print("正在从网络获取数据...")
        records = provider.fetch_all(self.config["provider_code"])
        if records:
            self.loader.save_data(records, append=True)
            total = len(self.load_history())
            print(f"\n共保存 {color_green(str(len(records)))} 条新数据，本地总计 {color_green(str(total))} 条")
            if records:
                last = records[-1]
                print(f"最新一期: {color_green(last['issue'])}")
                print(f"开奖号码: {format_numbers(last['reds'], last['blues'], self.config)}")
        else:
            print(color_yellow("未获取到数据，可能是网络问题"))

    def crawl_all_lotteries(self):
        print_header("批量抓取所有彩种数据")
        for key, cfg in self.configs.items():
            print(f"\n{color_yellow('▶ ' + cfg['name'])} (数据源: {cfg['provider']})")
            provider = get_provider(cfg["provider"])
            loader = self.loaders[key]
            try:
                records = provider.fetch_all(cfg["provider_code"])
                if records:
                    loader.save_data(records, append=True)
                    total = len(loader.load_history())
                    print(f"  ✓ 新增{len(records)}条，总计{total}条")
                    if records:
                        last = records[-1]
                        print(f"  最新期: {last['issue']} -> {format_numbers(last['reds'], last['blues'], cfg)}")
                else:
                    print(f"  ✗ 未获取到数据")
            except Exception as e:
                print(f"  ✗ 抓取失败: {e}")

    def show_history(self):
        print_header(f"{self.config['name']} 最近开奖记录")
        history = self.load_history()
        if not history:
            print(color_yellow("暂无数据，请先抓取或导入数据"))
            return

        print(f"共 {color_green(str(len(history)))} 期数据，最近20期：\n")
        recent = history[-20:]
        print(f"{'期号':<12} {'号码':<50}")
        print("-" * 60)
        for record in reversed(recent):
            print(f"{record['issue']:<12} {format_numbers(record['reds'], record['blues'], self.config)}")

    def show_frequency(self):
        print_header(f"{self.config['name']} 号码频率分析")
        history = self.load_history()
        if not history:
            print(color_yellow("暂无数据，请先抓取或导入数据"))
            return

        counts = self.freq_analyzer.count_all(history)
        total_draws = len(history)

        print(f"\n基于 {color_green(str(total_draws))} 期历史数据统计\n")

        if self.config["blue_count"] > 0:
            label = "前区" if self.lottery_key == "dlt" else "红球"
        else:
            label = "号码"
        print(color_red(f"【{label}出现频率统计】"))
        print("-" * 60)
        for n in range(self.config["red_range"][0], self.config["red_range"][1] + 1):
            cnt = counts["red"].get(n, 0)
            ratio = cnt / total_draws * 100
            bar = "█" * int(ratio)
            print(f"  {n:02d}: {cnt:4d}次 ({ratio:5.2f}%) {bar}")

        if self.config["blue_count"] > 0:
            label = "后区" if self.lottery_key == "dlt" else "蓝球"
            print(f"\n{color_blue('【' + label + '出现频率统计】')}")
            print("-" * 60)
            for n in range(self.config["blue_range"][0], self.config["blue_range"][1] + 1):
                cnt = counts["blue"].get(n, 0)
                ratio = cnt / total_draws * 100
                bar = "█" * int(ratio)
                print(f"  {n:02d}: {cnt:4d}次 ({ratio:5.2f}%) {bar}")

    def show_hot_cold(self):
        print_header(f"{self.config['name']} 冷热号分析 (近50期)")
        history = self.load_history()
        if not history:
            print(color_yellow("暂无数据，请先抓取或导入数据"))
            return

        hot = self.freq_analyzer.get_hot_numbers(history, recent_n=50, top_n=10)
        cold = self.freq_analyzer.get_cold_numbers(history, recent_n=50, bottom_n=10)
        overdue = self.freq_analyzer.get_overdue_numbers(history)

        red_label = "前区" if self.lottery_key == "dlt" else "红球"
        blue_label = "后区" if self.lottery_key == "dlt" else "蓝球"

        print(f"\n{color_red('🔥 ' + red_label + '热号 TOP10 (近50期)')}:")
        print(f"  {', '.join(f'{n:02d}({c}次)' for n, c in hot['red'])}")

        if self.config["blue_count"] > 0:
            print(f"\n{color_blue('🔥 ' + blue_label + '热号 TOP10')}:")
            print(f"  {', '.join(f'{n:02d}({c}次)' for n, c in hot['blue'])}")

        print(f"\n{color_red('❄️  ' + red_label + '冷号 TOP10')}:")
        print(f"  {', '.join(f'{n:02d}({c}次)' for n, c in cold['red'])}")

        if self.config["blue_count"] > 0:
            print(f"\n{color_blue('❄️  ' + blue_label + '冷号 TOP10')}:")
            print(f"  {', '.join(f'{n:02d}({c}次)' for n, c in cold['blue'])}")

        print(f"\n{color_yellow('⏰ ' + red_label + '遗漏值排行')}:")
        print(f"  {', '.join(f'{n:02d}({m}期)' for n, m in overdue['red'][:10])}")

        if self.config["blue_count"] > 0:
            print(f"\n{color_yellow('⏰ ' + blue_label + '遗漏值排行')}:")
            print(f"  {', '.join(f'{n:02d}({m}期)' for n, m in overdue['blue'][:10])}")

    def show_statistics(self):
        print_header(f"{self.config['name']} 统计分析")
        history = self.load_history()
        if not history:
            print(color_yellow("暂无数据，请先抓取或导入数据"))
            return

        sum_stats = self.stat_analyzer.sum_analysis(history)
        odd_even = self.stat_analyzer.odd_even_analysis(history)
        region = self.stat_analyzer.region_analysis(history)
        consec = self.stat_analyzer.consecutive_numbers(history)
        repeat = self.stat_analyzer.repeat_last_draw(history)

        red_label = "前区" if self.lottery_key == "dlt" else "红球"
        blue_label = "后区" if self.lottery_key == "dlt" else "蓝球"

        print(f"\n{color_yellow('【和值分析】')}")
        if sum_stats.get("red_sum"):
            rs = sum_stats["red_sum"]
            print(f"  {red_label}和值: 平均={rs['mean']}, 标准差={rs['std']}, 范围={rs['min']}-{rs['max']}, 中位数={rs['median']}")
        if self.config["blue_count"] > 0 and sum_stats.get("blue_sum"):
            bs = sum_stats["blue_sum"]
            print(f"  {blue_label}和值: 平均={bs['mean']}, 标准差={bs['std']}, 范围={bs['min']}-{bs['max']}")

        print(f"\n{color_yellow('【奇偶比分析 (前5热门)】')}")
        print(f"  {red_label}奇偶比:")
        for k, v in list(odd_even["red"].items())[:5]:
            print(f"    {k:<8} {v['count']:4d}次 占比{v['ratio']}%")
        if self.config["blue_count"] > 0:
            print(f"  {blue_label}奇偶比:")
            for k, v in odd_even["blue"].items():
                print(f"    {k:<8} {v['count']:4d}次 占比{v['ratio']}%")

        print(f"\n{color_yellow('【区间比分析 (前5热门)】')}")
        print(f"  {red_label}三区间比:")
        for k, v in list(region["red"].items())[:5]:
            print(f"    {k:<10} {v['count']:4d}次 占比{v['ratio']}%")

        print(f"\n{color_yellow('【连号分析】')}")
        for k, v in consec["red"].items():
            print(f"    {k}组连号: {v['count']:4d}次 占比{v['ratio']}%")

        print(f"\n{color_yellow('【重号分析】')}")
        print(f"  与上期重复{red_label}平均个数: {repeat['red_avg_repeat']}")
        if self.config["blue_count"] > 0:
            print(f"  与上期重复{blue_label}平均个数: {repeat['blue_avg_repeat']}")

    def predict_numbers(self):
        print_header(f"{self.config['name']} 号码预测")
        history = self.load_history()
        if not history:
            print(color_yellow("暂无数据，请先抓取或导入数据"))
            return

        print(f"基于 {color_green(str(len(history)))} 期历史数据分析\n")

        if history:
            last = history[-1]
            print(f"上期开奖: {last['issue']}")
            print(f"开奖号码: {format_numbers(last['reds'], last['blues'], self.config)}\n")

        results = self.predictor.predict_multi_strategy(history)
        for strategy_name, (reds, blues) in results.items():
            print(f"  {color_yellow(strategy_name)}:")
            print(f"    {format_numbers(reds, blues, self.config)}")

        print(f"\n{color_green('【加权随机5注】')}:")
        weighted = self.predictor.predict_weighted_random(history, n_sets=5)
        for i, (reds, blues) in enumerate(weighted, 1):
            print(f"  第{i}注: {format_numbers(reds, blues, self.config)}")

        print(f"\n{color_yellow('⚠️  免责声明：彩票开奖为随机事件，预测仅供参考娱乐，请理性购彩！')}")

    def show_data_status(self):
        print_header("所有彩种数据状态")
        print(f"{'彩种':<10} {'数据源':<12} {'期数':<8} {'最新期号':<12} {'最新号码'}")
        print("-" * 70)
        for key, cfg in self.configs.items():
            history = self.loaders[key].load_history()
            count = len(history)
            if history:
                last = history[-1]
                nums = format_numbers(last['reds'], last['blues'], cfg)
                print(f"{cfg['name']:<10} {color_cyan(cfg['provider']):<20} {color_green(str(count)):<8} {last['issue']:<12} {nums}")
            else:
                print(f"{cfg['name']:<10} {color_cyan(cfg['provider']):<20} {color_yellow('无数据'):<8} -")

    def show_main_menu(self):
        while True:
            print_header(f"彩票分析系统 - {self.config['name']}")
            print(f"  当前彩票: {color_yellow(self.config['name'])} | 数据源: {color_cyan(self.config['provider'])} | {self.config['schedule']}")
            print(f"""
  1. 切换彩票类型 (支持7种彩种)
  2. 抓取当前彩种最新数据
  3. 批量抓取所有彩种数据
  4. 查看历史开奖记录
  5. 号码频率统计
  6. 冷热号/遗漏分析
  7. 统计分析 (和值/奇偶/区间/连号)
  8. 号码预测推荐
  9. 查看所有彩种数据状态

  0. 退出
            """)
            choice = input("请选择功能 (0-9): ").strip()

            if choice == "1":
                self.select_lottery()
            elif choice == "2":
                self.crawl_data()
            elif choice == "3":
                self.crawl_all_lotteries()
            elif choice == "4":
                self.show_history()
            elif choice == "5":
                self.show_frequency()
            elif choice == "6":
                self.show_hot_cold()
            elif choice == "7":
                self.show_statistics()
            elif choice == "8":
                self.predict_numbers()
            elif choice == "9":
                self.show_data_status()
            elif choice == "0":
                print(color_green("\n感谢使用，祝您好运！"))
                break
            else:
                print(color_yellow("无效选择，请重新输入"))

            input("\n按回车键继续...")


def main():
    print(color_green("""
    ╔══════════════════════════════════════════════════╗
    ║            彩票数据爬取与分析系统 v3.0           ║
    ║     Lottery Data Crawler & Analysis System      ║
    ║     支持7种彩种 / 3个数据源 / 多策略预测        ║
    ╚══════════════════════════════════════════════════╝
    """))
    app = LotteryApp()
    app.show_main_menu()


if __name__ == "__main__":
    main()
