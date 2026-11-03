from typing import List, Dict, Optional
import re

from .base import BaseProvider
from ..config import REQUEST_TIMEOUT


class Data500Provider(BaseProvider):
    """500.com 数据源 (备用源)

    支持全部 7 个彩种: ssq, dlt, fc3d(页面路径 sd), qlc, qxc, pls, plw

    说明:
    - 500.com 各彩种的页面路径和表格结构不统一, 这里按彩种分别映射。
    - 解析采用「期号 + 号码单元格」的定位方式, 不依赖固定的列下标,
      避免 500.com 调整表格列顺序/增删列时解析错位。
    - 编码为 gb2312/gb18030, 必须在读取后显式转码。
    """

    name = "data500"

    BASE_URL = "https://datachart.500.com/{lottery}/history/{path}/history.php"

    # 彩种 -> (页面路径段, 表格模板目录, 红球个数, 蓝球个数)
    GAME_URL_MAP = {
        "ssq": ("ssq", "newinc", 6, 1),
        "dlt": ("dlt", "newinc", 5, 2),
        "fc3d": ("sd", "inc", 3, 0),
        "qlc": ("qlc", "newinc", 7, 1),
        "qxc": ("qxc", "inc", 7, 0),
        "pls": ("pls", "inc", 3, 0),
        "plw": ("plw", "inc", 5, 0),
    }

    # 号码单元格类名在各彩种页面上有 cfont2 / t_cfont2 两种写法, 统一用后缀匹配
    _RED_CELL_RE = re.compile(r'<td class="[^"]*cfont2[^"]*"[^>]*>(.*?)</td>', re.S)
    _BLUE_CELL_RE = re.compile(r'<td class="[^"]*cfont4[^"]*"[^>]*>(.*?)</td>', re.S)
    _NUM_RE = re.compile(r"\d+")

    def _game_cfg(self, lottery_key: str):
        return self.GAME_URL_MAP.get(lottery_key)

    def fetch_page(self, lottery_key: str, limit: int = 500) -> Optional[str]:
        cfg = self._game_cfg(lottery_key)
        if not cfg:
            print(f"  500.com 不支持彩种: {lottery_key}")
            return None
        page_key, path, _, _ = cfg
        url = self.BASE_URL.format(lottery=page_key, path=path)
        params = {"limit": limit, "sort": 0}
        response = self._safe_get(url, self.headers, params,
                                   timeout=REQUEST_TIMEOUT, session=self.session)
        if not response:
            return None
        # 500.com 页面为 gb2312 编码, 用 gb18030 兼容解码
        response.encoding = "gb18030"
        return response.text

    def parse_html(self, html: str, lottery_key: str) -> List[Dict]:
        results = []
        if not html:
            return results

        cfg = self._game_cfg(lottery_key)
        if not cfg:
            return results
        _, _, red_count, blue_count = cfg

        try:
            for row in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
                def cell_nums(pattern) -> List[int]:
                    nums: List[int] = []
                    for cell in pattern.findall(row):
                        text = re.sub(r"<[^>]+>", "", cell)
                        nums.extend(int(x) for x in self._NUM_RE.findall(text))
                    return nums

                # 期号: 行内第一个 5~7 位纯数字的 td 文本
                issue = None
                for td in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S):
                    text = re.sub(r"<[^>]+>", "", td).strip()
                    if re.match(r"^\d{5,7}$", text):
                        issue = text
                        break
                if not issue:
                    continue

                # 主号码: 所有 *cfont2* 单元格的数字之和。
                # 不同彩种写法不一: 有的一个 td 一个号(双色球/大乐透),
                # 有的整组号码挤在一个 td 里用空格分隔(七乐彩/七星彩/福彩3D/排列3/排列5)。
                all_nums = cell_nums(self._RED_CELL_RE)
                if not all_nums:
                    continue

                blues = cell_nums(self._BLUE_CELL_RE) if blue_count > 0 else []

                # 蓝球号码与红球写在同一个 td 时(七乐彩), 尾部数字即为蓝球
                if blue_count > 0 and not blues and len(all_nums) > red_count:
                    all_nums, blues = all_nums[:red_count], all_nums[red_count:]

                reds = all_nums[:red_count]
                if lottery_key not in {"fc3d", "pls", "plw", "qxc"}:
                    reds.sort()

                if len(reds) != red_count:
                    continue
                if blue_count > 0 and len(blues) != blue_count:
                    continue

                date_match = re.search(r"(\d{4}-\d{2}-\d{2})", row)
                results.append({
                    "issue": issue,
                    "date": date_match.group(1) if date_match else "",
                    "reds": reds,
                    "blues": sorted(blues),
                })
        except Exception as e:
            print(f"  解析HTML失败: {e}")

        return results

    def fetch_all(self, lottery_key: str, callback=None) -> List[Dict]:
        html = self.fetch_page(lottery_key)
        if not html:
            return []

        records = self.parse_html(html, lottery_key)
        if callback:
            callback(records)

        print(f"  从500.com获取到 {len(records)} 条数据")

        from ..data.loader import issue_sort_key
        return sorted(records, key=lambda x: issue_sort_key(x["issue"]))

    def fetch_latest(self, lottery_key: str) -> Optional[Dict]:
        records = self.fetch_all(lottery_key)
        return records[-1] if records else None
