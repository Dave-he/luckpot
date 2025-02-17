import requests
import os
import csv

PAGE_NUM = 365
url = 'https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice'

headers = {
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Accept-Language': 'zh-CN,zh;q=0.9,id-ID;q=0.8,id;q=0.7,en-US;q=0.6,en;q=0.5',
    'Connection': 'keep-alive',
    'Referer': 'https://www.cwl.gov.cn/ygkj/wqkjgg/',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest',
    'sec-ch-ua': '"Not(A:Brand";v="99", "Google Chrome";v="133", "Chromium";v="133"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"'
}

cookies = {
    'UniqueID': 'sHTAEKiBAl1fs3by1739761892285',
    'Sites': '_21',
    'HMF_CI': '2e73d2e38e1d07396dcfa1bf586ca0d14d43db3a9e43c83188dc161715686f30040d4bb53515b2a8f3f5540e93f4392fbac8d0a93f48b5dc9988407a2c31829c76',
    '21_vq': '2'
}


def get_data(pageNo):
    params = {
        'name': 'ssq',
        'pageNo': pageNo,
        'pageSize': 30,
        'systemType': 'PC'
    }
    try:
        response = requests.get(url, params=params, headers=headers, cookies=cookies, timeout=10)
        # response.raise_for_status()  # 检查请求是否成功
        return response.json()
    except requests.RequestException as e:
        print(f"请求出错: {e}")
    except ValueError as e:
        print(f"解析 JSON 数据出错: {e}")


def write(pathfile, data):
    if not os.path.exists(pathfile):
        with open(pathfile, 'w') as f:
            csv.writer(f, dialect='excel').writerow(['Date Time','red1','red2','red3','red4','red5','red6','blue'])
    f = open(pathfile, 'a', newline='')
    wr = csv.writer(f, dialect='excel')
    wr.writerow(data)

def parse_data(res):
    # 解析JSON数据
    # res = json.loads(data)
    global PAGE_NUM
    PAGE_NUM = res.get('pageNum')

    if res.get('state') == 0:
        # 如果返回码为0，表示请求成功
        # 提取所需信息
        result = res.get('result')
        for item in result:
            print(item.get('code'), item.get('date'),
                  item.get('red'), item.get('blue'))
            data = []
            data.append(item.get('code'))
            data.extend(item.get('red').split(','))
            data.append(item.get('blue'))
            write('result.csv', data)


if __name__ == '__main__':
    curr = 1
    while curr <= PAGE_NUM:
        data = get_data(curr)
        curr += 1
        if data is None:
            continue
        parse_data(data)
