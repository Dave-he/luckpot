import time
from typing import List, Dict, Optional

from .base import BaseProvider
from ..config import REQUEST_TIMEOUT


class SportteryProvider(BaseProvider):
    """中国体育彩票官网数据源 (webapi.sporttery.cn)
    支持彩种: dlt(大乐透, gameNo=85), qxc(七星彩, gameNo=04),
             pls(排列3, gameNo=35), plw(排列5, gameNo=350133)
    """

    name = "sporttery"

    API_URL = "https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry"
    REFERER = "https://www.lottery.gov.cn/"

    # 彩种配置: short_name -> (gameNo, red_count, blue_count)
    GAME_CONFIG = {
        "dlt": ("85", 5, 2),
        "qxc": ("04", 7, 0),
        "pls": ("35", 3, 0),
        "plw": ("350133", 5, 0),
    }

    def __init__(self):
        super().__init__()
        self.headers.update({
            "Referer": self.REFERER,
            "Host": "webapi.sporttery.cn",
        })

    def _resolve_game_no(self, lottery_key: str) -> Optional[str]:
        if lottery_key in self.GAME_CONFIG:
            return self.GAME_CONFIG[lottery_key][0]
        return lottery_key

    def fetch_page(self, lottery_key: str, page_no: int = 1, page_size: int = 100) -> Optional[Dict]:
        game_no = self._resolve_game_no(lottery_key)
        params = {
            "gameNo": game_no,
            "provinceId": 0,
            "pageSize": page_size,
            "isVerify": 1,
            "pageNo": page_no,
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
        if not data:
            return results

        error_code = data.get("errorCode")
        if error_code is not None and str(error_code) != "0":
            err_msg = data.get("errorMessage", "未知错误")
            print(f"  接口返回错误: {err_msg}")
            return results

        value = data.get("value", {})
        if not value:
            return results

        items = value.get("list", [])

        # 支持通过lottery_key或game_no查找配置
        game_cfg = self.GAME_CONFIG.get(lottery_key)
        if game_cfg is None:
            for k, cfg in self.GAME_CONFIG.items():
                if cfg[0] == str(lottery_key):
                    game_cfg = cfg
                    lottery_key = k
                    break
        if game_cfg is None:
            game_cfg = (lottery_key, 5, 2)

        _, red_count, blue_count = game_cfg

        for item in items:
            try:
                code = item.get("lotteryDrawNum", "").strip()
                date_str = item.get("lotteryDrawTime", "").strip()
                date = date_str[:10] if len(date_str) >= 10 else date_str

                result_str = item.get("lotteryDrawResult", "").strip()
                if not code or not result_str:
                    continue

                # 体彩号码用空格分隔，统一用空格处理
                nums = [int(x.strip()) for x in result_str.replace(",", " ").split() if x.strip().isdigit()]

                if lottery_key == "dlt":
                    # 大乐透: 前5个红球 + 后2个蓝球
                    if len(nums) < red_count + blue_count:
                        continue
                    reds = sorted(nums[:red_count])
                    blues = sorted(nums[red_count:red_count + blue_count])
                else:
                    # 七星彩/排列3/排列5: 全部为红球
                    if len(nums) < red_count:
                        continue
                    reds = sorted(nums[:red_count])
                    blues = []

                if len(reds) == red_count:
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
                value = data.get("value", {})
                total = value.get("total", 0) or value.get("totalNumber", 0)
                pages = value.get("pages", 0)
                page_size = value.get("pageSize", 100)
                if pages:
                    total_pages = pages
                elif total and page_size:
                    total_pages = (total + page_size - 1) // page_size
                else:
                    total_pages = 1
                print(f"  共 {total_pages} 页数据 (总计{total}条)")

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
