import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

for d in [DATA_DIR]:
    os.makedirs(d, exist_ok=True)


def _config(name, short_name, red_count, red_range, blue_count, blue_range,
            provider, provider_code, schedule=""):
    data_dir = os.path.join(DATA_DIR, short_name)
    os.makedirs(data_dir, exist_ok=True)
    return {
        "name": name,
        "short_name": short_name,
        "red_count": red_count,
        "red_range": red_range,
        "blue_count": blue_count,
        "blue_range": blue_range,
        "data_file": os.path.join(data_dir, "history.csv"),
        "provider": provider,
        "provider_code": provider_code,
        "schedule": schedule,
    }


SSQ_CONFIG = _config("双色球", "ssq", 6, (1, 33), 1, (1, 16),
                     "cwl", "ssq", "每周二、四、日")

DLT_CONFIG = _config("大乐透", "dlt", 5, (1, 35), 2, (1, 12),
                     "sporttery", "85", "每周一、三、六")

FC3D_CONFIG = _config("福彩3D", "fc3d", 3, (0, 9), 0, (0, 0),
                      "cwl", "3d", "每日开奖")

QLC_CONFIG = _config("七乐彩", "qlc", 7, (1, 30), 1, (1, 30),
                     "cwl", "qlc", "每周一、三、五")

QXC_CONFIG = _config("七星彩", "qxc", 7, (0, 9), 0, (0, 0),
                     "sporttery", "04", "每周二、五、日")

PLS_CONFIG = _config("排列3", "pls", 3, (0, 9), 0, (0, 0),
                     "sporttery", "35", "每日开奖")

PLW_CONFIG = _config("排列5", "plw", 5, (0, 9), 0, (0, 0),
                     "sporttery", "350133", "每日开奖")

LOTTERY_CONFIGS = {
    "ssq": SSQ_CONFIG,
    "dlt": DLT_CONFIG,
    "fc3d": FC3D_CONFIG,
    "qlc": QLC_CONFIG,
    "qxc": QXC_CONFIG,
    "pls": PLS_CONFIG,
    "plw": PLW_CONFIG,
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
