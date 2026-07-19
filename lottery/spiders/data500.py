from typing import List, Dict, Optional
from lxml import etree

from .base import BaseProvider
from ..config import REQUEST_TIMEOUT


class Data500Provider(BaseProvider):
    """500.com数据源 (备选)
    支持彩种: dlt(大乐透), ssq(双色球)等
    主要作为体彩源失效时的备选
    """

    name = "data500"

    BASE_URL = "https://datachart.500.com/{lottery}/history/newinc/history.php"

    GAME_URL_MAP = {
        "dlt": "dlt",
        "ssq": "ssq",
    }

    def fetch_page(self, lottery_key: str, limit: int = 5000) -> Optional[str]:
        url_key = self.GAME_URL_MAP.get(lottery_key, lottery_key)
        url = self.BASE_URL.format(lottery=url_key)
        params = {"limit": limit, "sort": 0}
        response = self._safe_get(url, self.headers, params,
                                   timeout=REQUEST_TIMEOUT, session=self.session)
        if not response:
            return None
        response.encoding = "utf-8"
        return response.text

    def parse_html(self, html: str, lottery_key: str) -> List[Dict]:
        results = []
        if not html:
            return results

        try:
            tree = etree.HTML(html)
            trs = tree.xpath('//tbody[@id="tdata"]/tr')
            for tr in trs:
                tds = tr.xpath("./td/text()")
                if len(tds) < 2:
                    continue

                try:
                    issue = tds[0].strip()
                    if not issue or not issue.isdigit():
                        continue

                    if lottery_key == "dlt":
                        if len(tds) < 8:
                            continue
                        reds = [int(tds[i].strip()) for i in range(1, 6) if tds[i].strip().isdigit()]
                        blues = [int(tds[i].strip()) for i in range(6, 8) if tds[i].strip().isdigit()]
                    elif lottery_key == "ssq":
                        if len(tds) < 8:
                            continue
                        reds = [int(tds[i].strip()) for i in range(1, 7) if tds[i].strip().isdigit()]
                        blues = [int(tds[i].strip()) for i in range(7, 8) if tds[i].strip().isdigit()]
                    else:
                        continue

                    if reds and blues:
                        results.append({
                            "issue": issue,
                            "date": "",
                            "reds": sorted(reds),
                            "blues": sorted(blues),
                        })
                except (ValueError, IndexError):
                    continue
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
