#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lottery.config import SSQ_CONFIG, DLT_CONFIG
from lottery.spiders import SSQSpider, DLTSpider
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


def format_numbers(reds, blues, lottery_type="ssq"):
    red_str = " ".join(color_red(f"{n:02d}") for n in sorted(reds))
    blue_str = " ".join(color_blue(f"{n:02d}") for n in sorted(blues))
    return f"{red_str} + {blue_str}"


def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


class LotteryApp:
    def __init__(self):
        self.lottery_type = "ssq"
        self.configs = {"ssq": SSQ_CONFIG, "dlt": DLT_CONFIG}
        self.spiders = {"ssq": SSQSpider, "dlt": DLTSpider}
        self.loaders = {}
        self.processors = {}
        self.freq_analyzers = {}
        self.stat_analyzers = {}
        self.predictors = {}
        self._init_components()

    def _init_components(self):
        for key, config in self.configs.items():
            self.loaders[key] = DataLoader(config)
            self.processors[key] = DataProcessor(config)
            self.freq_analyzers[key] = FrequencyAnalyzer(config)
            self.stat_analyzers[key] = StatisticsAnalyzer(config)
            self.predictors[key] = LotteryPredictor(config)

    @property
    def config(self):
        return self.configs[self.lottery_type]

    @property
    def loader(self):
        return self.loaders[self.lottery_type]

    @property
    def processor(self):
        return self.processors[self.lottery_type]

    @property
    def freq_analyzer(self):
        return self.freq_analyzers[self.lottery_type]

    @property
    def stat_analyzer(self):
        return self.stat_analyzers[self.lottery_type]

    @property
    def predictor(self):
        return self.predictors[self.lottery_type]

    def load_history(self):
        return self.loader.load_history()

    def select_lottery(self):
        print_header("选择彩票类型")
        print(f"  1. 双色球 (当前: {'✓' if self.lottery_type == 'ssq' else ' '})")
        print(f"  2. 大乐透 (当前: {'✓' if self.lottery_type == 'dlt' else ' '})")
        choice = input("\n请选择 (1/2, 默认1): ").strip()
        if choice == "2":
            self.lottery_type = "dlt"
        else:
            self.lottery_type = "ssq"
        print(f"\n已切换到: {color_yellow(self.config['name'])}")

    def crawl_data(self):
        print_header(f"抓取{self.config['name']}最新数据")
        latest = self.loader.get_latest_issue()
        if latest:
            print(f"本地最新期号: {color_green(latest)}")

        spider_cls = self.spiders[self.lottery_type]
        spider = spider_cls()
        print("正在从网络获取数据...")
        records = spider.fetch_all()
        if records:
            self.loader.save_data(records, append=False)
            print(f"\n共保存 {color_green(str(len(records)))} 条数据")
            if records:
                last = records[-1]
                print(f"最新一期: {color_green(last['issue'])}")
                print(f"开奖号码: {format_numbers(last['reds'], last['blues'], self.lottery_type)}")
        else:
            print(color_yellow("未获取到数据，可能是网络问题"))

    def import_old_data(self):
        print_header("导入旧版数据")
        if self.lottery_type == "ssq":
            old_files = ["/workspace/ball.csv", "/workspace/result.csv"]
        else:
            old_files = []
        count = self.loader.merge_existing_csv(old_files)
        print(f"已导入 {color_green(str(count))} 条历史数据")

    def show_history(self):
        print_header(f"{self.config['name']} 最近开奖记录")
        history = self.load_history()
        if not history:
            print(color_yellow("暂无数据，请先抓取或导入数据"))
            return

        print(f"共 {color_green(str(len(history)))} 期数据，最近20期：\n")
        recent = history[-20:]
        print(f"{'期号':<12} {'红球':<30} {'蓝球':<15}")
        print("-" * 60)
        for record in reversed(recent):
            red_str = " ".join(f"{n:02d}" for n in record["reds"])
            blue_str = " ".join(f"{n:02d}" for n in record["blues"])
            print(f"{record['issue']:<12} {color_red(red_str):<30} {color_blue(blue_str):<15}")

    def show_frequency(self):
        print_header(f"{self.config['name']} 号码频率分析")
        history = self.load_history()
        if not history:
            print(color_yellow("暂无数据，请先抓取或导入数据"))
            return

        counts = self.freq_analyzer.count_all(history)
        total_draws = len(history)

        print(f"\n基于 {color_green(str(total_draws))} 期历史数据统计\n")

        print(color_red("【红球出现频率统计】"))
        print("-" * 60)
        for n in range(self.config["red_range"][0], self.config["red_range"][1] + 1):
            cnt = counts["red"].get(n, 0)
            ratio = cnt / total_draws * 100
            bar = "█" * int(ratio)
            print(f"  {n:02d}: {cnt:4d}次 ({ratio:5.2f}%) {bar}")

        print(f"\n{color_blue('【蓝球出现频率统计】')}")
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

        print(f"\n{color_red('🔥 红球热号 TOP10 (近50期出现次数)')}:")
        hot_red_str = ", ".join(f"{n:02d}({c}次)" for n, c in hot["red"])
        print(f"  {hot_red_str}")

        print(f"\n{color_blue('🔥 蓝球热号 TOP10 (近50期出现次数)')}:")
        hot_blue_str = ", ".join(f"{n:02d}({c}次)" for n, c in hot["blue"])
        print(f"  {hot_blue_str}")

        print(f"\n{color_red('❄️  红球冷号 TOP10 (近50期出现次数)')}:")
        cold_red_str = ", ".join(f"{n:02d}({c}次)" for n, c in cold["red"])
        print(f"  {cold_red_str}")

        print(f"\n{color_blue('❄️  蓝球冷号 TOP10 (近50期出现次数)')}:")
        cold_blue_str = ", ".join(f"{n:02d}({c}次)" for n, c in cold["blue"])
        print(f"  {cold_blue_str}")

        print(f"\n{color_yellow('⏰ 红球遗漏值排行 (当前连续未出期数)')}:")
        od_red = ", ".join(f"{n:02d}({m}期)" for n, m in overdue["red"][:10])
        print(f"  {od_red}")

        print(f"\n{color_yellow('⏰ 蓝球遗漏值排行 (当前连续未出期数)')}:")
        od_blue = ", ".join(f"{n:02d}({m}期)" for n, m in overdue["blue"][:10])
        print(f"  {od_blue}")

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

        print(f"\n{color_yellow('【和值分析】')}")
        if sum_stats.get("red_sum"):
            rs = sum_stats["red_sum"]
            print(f"  红球和值: 平均={rs['mean']}, 标准差={rs['std']}, 范围={rs['min']}-{rs['max']}, 中位数={rs['median']}")
        if sum_stats.get("blue_sum"):
            bs = sum_stats["blue_sum"]
            print(f"  蓝球和值: 平均={bs['mean']}, 标准差={bs['std']}, 范围={bs['min']}-{bs['max']}")

        print(f"\n{color_yellow('【奇偶比分析 (前5热门)】')}")
        print("  红球奇偶比:")
        for k, v in list(odd_even["red"].items())[:5]:
            print(f"    {k:<8} {v['count']:4d}次 占比{v['ratio']}%")
        print("  蓝球奇偶比:")
        for k, v in odd_even["blue"].items():
            print(f"    {k:<8} {v['count']:4d}次 占比{v['ratio']}%")

        print(f"\n{color_yellow('【区间比分析 (前5热门)】')}")
        print("  红球三区间比:")
        for k, v in list(region["red"].items())[:5]:
            print(f"    {k:<10} {v['count']:4d}次 占比{v['ratio']}%")

        print(f"\n{color_yellow('【连号分析】')}")
        for k, v in consec["red"].items():
            print(f"    {k}组连号: {v['count']:4d}次 占比{v['ratio']}%")

        print(f"\n{color_yellow('【重号分析】')}")
        print(f"  与上期重复红球平均个数: {repeat['red_avg_repeat']}")
        print(f"  与上期重复蓝球平均个数: {repeat['blue_avg_repeat']}")

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
            print(f"开奖号码: {format_numbers(last['reds'], last['blues'], self.lottery_type)}\n")

        results = self.predictor.predict_multi_strategy(history)
        for strategy_name, (reds, blues) in results.items():
            print(f"  {color_yellow(strategy_name)}:")
            print(f"    {format_numbers(reds, blues, self.lottery_type)}")

        print(f"\n{color_green('【加权随机5注】')}:")
        weighted = self.predictor.predict_weighted_random(history, n_sets=5)
        for i, (reds, blues) in enumerate(weighted, 1):
            print(f"  第{i}注: {format_numbers(reds, blues, self.lottery_type)}")

        print(f"\n{color_yellow('⚠️  免责声明：彩票开奖为随机事件，预测仅供参考娱乐，请理性购彩！')}")

    def show_main_menu(self):
        while True:
            print_header(f"彩票分析系统 - {self.config['name']}")
            print(f"  当前彩票类型: {color_yellow(self.config['name'])}")
            print(f"""
  1. 切换彩票类型 (双色球/大乐透)
  2. 抓取最新开奖数据
  3. 导入旧版数据
  4. 查看历史开奖记录
  5. 号码频率统计
  6. 冷热号/遗漏分析
  7. 统计分析 (和值/奇偶/区间/连号)
  8. 号码预测推荐

  0. 退出
            """)
            choice = input("请选择功能 (0-8): ").strip()

            if choice == "1":
                self.select_lottery()
            elif choice == "2":
                self.crawl_data()
            elif choice == "3":
                self.import_old_data()
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
            elif choice == "0":
                print(color_green("\n感谢使用，祝您好运！"))
                break
            else:
                print(color_yellow("无效选择，请重新输入"))

            input("\n按回车键继续...")


def main():
    print(color_green("""
    ╔══════════════════════════════════════════════════╗
    ║            彩票数据爬取与分析系统 v2.0           ║
    ║     Lottery Data Crawler & Analysis System      ║
    ╚══════════════════════════════════════════════════╝
    """))
    app = LotteryApp()
    app.show_main_menu()


if __name__ == "__main__":
    main()
