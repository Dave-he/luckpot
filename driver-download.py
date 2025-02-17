import requests
from functools import partial
import json
json_format = partial(json.dumps, ensure_ascii=False, indent=4, sort_keys=True)

url = 'https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json'
response = requests.get(url).json()
print(json_format(response))