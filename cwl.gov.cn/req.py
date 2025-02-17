import requests

url = 'http://www.cwl.gov.cn/cwl_admin/kjxx/findDrawNotice'
res = requests.get(url)   
print(res.text)