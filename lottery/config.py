import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

SSQ_DATA_DIR = os.path.join(DATA_DIR, "ssq")
DLT_DATA_DIR = os.path.join(DATA_DIR, "dlt")

for d in [DATA_DIR, SSQ_DATA_DIR, DLT_DATA_DIR]:
    os.makedirs(d, exist_ok=True)

SSQ_CONFIG = {
    "name": "双色球",
    "short_name": "ssq",
    "red_count": 6,
    "red_range": (1, 33),
    "blue_count": 1,
    "blue_range": (1, 16),
    "data_file": os.path.join(SSQ_DATA_DIR, "history.csv"),
}

DLT_CONFIG = {
    "name": "大乐透",
    "short_name": "dlt",
    "red_count": 5,
    "red_range": (1, 35),
    "blue_count": 2,
    "blue_range": (1, 12),
    "data_file": os.path.join(DLT_DATA_DIR, "history.csv"),
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/133.0.0.0 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

REQUEST_TIMEOUT = 15
REQUEST_DELAY = 1
