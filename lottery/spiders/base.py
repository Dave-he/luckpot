import time
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import requests

from ..config import HEADERS, REQUEST_TIMEOUT, REQUEST_DELAY


class BaseProvider(ABC):
    """彩票数据源抽象基类"""

    name: str = "base"

    def __init__(self):
        self.headers = HEADERS.copy()
        self.session = requests.Session()

    @abstractmethod
    def fetch_all(self, lottery_key: str, callback=None) -> List[Dict]:
        """抓取指定彩种的所有历史数据"""
        pass

    @abstractmethod
    def fetch_latest(self, lottery_key: str) -> Optional[Dict]:
        """抓取指定彩种的最新一期"""
        pass

    @staticmethod
    def _safe_get(url: str, headers: Dict, params: Dict = None,
                  timeout: int = REQUEST_TIMEOUT, session: requests.Session = None) -> Optional[requests.Response]:
        try:
            sess = session or requests
            response = sess.get(url, headers=headers, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            print(f"  请求失败: {e}")
            return None

    @staticmethod
    def _sleep():
        time.sleep(REQUEST_DELAY)
