import time
from typing import List, Dict, Optional
import requests

from ..config import HEADERS, REQUEST_TIMEOUT, REQUEST_DELAY


class SSQSpider:
    def __init__(self):
        self.name = "双色球"
        self.url = "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
        self.headers = HEADERS.copy()
        self.headers.update({
            "Referer": "https://www.cwl.gov.cn/ygkj/wqkjgg/ssq/",
            "X-Requested-With": "XMLHttpRequest",
        })
        self.session = requests.Session()
        self._init_cookies()

    def _init_cookies(self):
        try:
            self.session.get(
                "https://www.cwl.gov.cn/ygkj/wqkjgg/ssq/",
                headers=self.headers,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException:
            pass

    def fetch_page(self, page_no: int = 1, page_size: int = 100) -> Optional[Dict]:
        params = {
            "name": "ssq",
            "issueCount": "",
            "issueStart": "",
            "issueEnd": "",
            "dayStart": "",
            "dayEnd": "",
            "pageNo": page_no,
            "pageSize": page_size,
            "systemType": "PC",
        }
        try:
            response = self.session.get(
                self.url,
                params=params,
                headers=self.headers,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"请求第{page_no}页出错: {e}")
            return None
        except ValueError as e:
            print(f"解析JSON出错: {e}")
            return None

    def parse_response(self, data: Dict) -> List[Dict]:
        results = []
        if not data or data.get("state") != 0:
            return results

        for item in data.get("result", []):
            try:
                code = item.get("code", "").strip()
                date = item.get("date", "").strip()
                red_str = item.get("red", "").strip()
                blue_str = item.get("blue", "").strip()

                if not code or not red_str:
                    continue

                reds = sorted([int(x.strip()) for x in red_str.split(",") if x.strip()])
                blues = sorted([int(x.strip()) for x in blue_str.split(",") if x.strip()])

                if len(reds) == 6 and len(blues) == 1:
                    results.append({
                        "issue": code,
                        "date": date,
                        "reds": reds,
                        "blues": blues,
                    })
            except (ValueError, AttributeError):
                continue

        return results

    def fetch_all(self, max_pages: int = 0, callback=None) -> List[Dict]:
        all_records = {}
        page_no = 1
        total_pages = None

        while True:
            if max_pages > 0 and page_no > max_pages:
                break

            data = self.fetch_page(page_no)
            if not data:
                break

            if total_pages is None:
                total_pages = data.get("pageNum", 1)
                print(f"共 {total_pages} 页数据")

            records = self.parse_response(data)
            if not records:
                break

            for record in records:
                all_records[record["issue"]] = record

            if callback:
                callback(records)

            print(f"已获取第 {page_no}/{total_pages} 页，本页{len(records)}条记录")

            if page_no >= total_pages:
                break

            page_no += 1
            time.sleep(REQUEST_DELAY)

        return sorted(all_records.values(), key=lambda x: x["issue"])

    def fetch_latest(self) -> Optional[Dict]:
        data = self.fetch_page(1, 1)
        records = self.parse_response(data)
        return records[0] if records else None
