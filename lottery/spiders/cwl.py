import time
from typing import List, Dict, Optional

from .base import BaseProvider
from ..config import REQUEST_TIMEOUT, REQUEST_DELAY


class CWLProvider(BaseProvider):
    """中国福利彩票官网数据源 (cwl.gov.cn)
    支持彩种: ssq(双色球), 3d(福彩3D), qlc(七乐彩), kl8(快乐8)
    """

    name = "cwl"

    API_URL = "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
    HOME_URL = "https://www.cwl.gov.cn/ygkj/wqkjgg/"

    # 彩种对应的Referer
    REFERER_MAP = {
        "ssq": "https://www.cwl.gov.cn/ygkj/wqkjgg/ssq/",
        "3d": "https://www.cwl.gov.cn/ygkj/wqkjgg/fc3d/",
        "qlc": "https://www.cwl.gov.cn/ygkj/wqkjgg/qlc/",
        "kl8": "https://www.cwl.gov.cn/ygkj/wqkjgg/kl8/",
    }

    def __init__(self):
        super().__init__()
        self.headers.update({
            "X-Requested-With": "XMLHttpRequest",
        })

    def _init_cookies(self, lottery_key: str):
        referer = self.REFERER_MAP.get(lottery_key, self.HOME_URL)
        self.headers["Referer"] = referer
        try:
            self.session.get(referer, headers=self.headers, timeout=REQUEST_TIMEOUT)
        except Exception:
            pass

    def fetch_page(self, lottery_key: str, page_no: int = 1, page_size: int = 100) -> Optional[Dict]:
        self._init_cookies(lottery_key)
        params = {
            "name": lottery_key,
            "issueCount": "",
            "issueStart": "",
            "issueEnd": "",
            "dayStart": "",
            "dayEnd": "",
            "pageNo": page_no,
            "pageSize": page_size,
            "systemType": "PC",
        }
        response = self._safe_get(self.API_URL, self.headers, params,
                                   timeout=REQUEST_TIMEOUT, session=self.session)
        if not response:
            return None
        try:
            return response.json()
        except ValueError as e:
            print(f"  解析JSON失败: {e}")
            return None

    def parse_response(self, data: Dict, lottery_key: str) -> List[Dict]:
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
                blues = []
                if blue_str:
                    blues = sorted([int(x.strip()) for x in blue_str.split(",") if x.strip()])

                results.append({
                    "issue": code,
                    "date": date,
                    "reds": reds,
                    "blues": blues,
                })
            except (ValueError, AttributeError):
                continue

        return results

    def fetch_all(self, lottery_key: str, callback=None) -> List[Dict]:
        all_records = {}
        page_no = 1
        total_pages = None

        while True:
            data = self.fetch_page(lottery_key, page_no)
            if not data:
                break

            if total_pages is None:
                total_pages = data.get("pageNum", 1)
                print(f"  共 {total_pages} 页数据")

            records = self.parse_response(data, lottery_key)
            if not records:
                break

            for record in records:
                all_records[record["issue"]] = record

            if callback:
                callback(records)

            print(f"  已获取第 {page_no}/{total_pages} 页，本页{len(records)}条")

            if page_no >= total_pages:
                break

            page_no += 1
            self._sleep()

        from ..data.loader import issue_sort_key
        return sorted(all_records.values(), key=lambda x: issue_sort_key(x["issue"]))

    def fetch_latest(self, lottery_key: str) -> Optional[Dict]:
        data = self.fetch_page(lottery_key, 1, 1)
        records = self.parse_response(data, lottery_key)
        return records[0] if records else None
