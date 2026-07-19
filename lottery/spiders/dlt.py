import re
import time
from typing import List, Dict, Optional
import requests
from lxml import etree

from ..config import HEADERS, REQUEST_TIMEOUT, REQUEST_DELAY


class DLTSpider:
    def __init__(self):
        self.name = "大乐透"
        self.base_url = "https://datachart.500.com/dlt/history/newinc/history.php"
        self.headers = HEADERS.copy()
        self.session = requests.Session()

    def fetch_page(self, start: str = "", end: str = "", limit: int = 100) -> Optional[str]:
        params = {"limit": limit, "sort": 0}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        try:
            response = self.session.get(
                self.base_url,
                params=params,
                headers=self.headers,
                timeout=REQUEST_TIMEOUT,
            )
            response.encoding = "utf-8"
            return response.text
        except requests.RequestException as e:
            print(f"请求出错: {e}")
            return None

    def parse_html(self, html: str) -> List[Dict]:
        results = []
        if not html:
            return results

        try:
            tree = etree.HTML(html)
            trs = tree.xpath('//tbody[@id="tdata"]/tr')
            for tr in trs:
                tds = tr.xpath("./td/text()")
                if len(tds) < 9:
                    continue

                try:
                    issue = tds[0].strip()
                    if not issue or not issue.isdigit():
                        continue

                    reds = []
                    for i in range(1, 6):
                        val = tds[i].strip()
                        if val.isdigit():
                            reds.append(int(val))

                    blues = []
                    for i in range(6, 8):
                        val = tds[i].strip()
                        if val.isdigit():
                            blues.append(int(val))

                    if len(reds) == 5 and len(blues) == 2:
                        results.append({
                            "issue": issue,
                            "date": "",
                            "reds": sorted(reds),
                            "blues": sorted(blues),
                        })
                except (ValueError, IndexError):
                    continue
        except Exception as e:
            print(f"解析HTML出错: {e}")

        return results

    def fetch_all(self, callback=None) -> List[Dict]:
        all_records = {}
        end_num = None

        html = self.fetch_page(limit=5000)
        if html:
            records = self.parse_html(html)
            for record in records:
                all_records[record["issue"]] = record
            if callback:
                callback(records)
            print(f"从500.com获取到 {len(records)} 条大乐透历史数据")

        return sorted(all_records.values(), key=lambda x: x["issue"])

    def fetch_latest(self) -> Optional[Dict]:
        records = self.fetch_all()
        return records[-1] if records else None
