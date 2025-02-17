import os
import csv
import time
from selenium import webdriver
from lxml import etree

ENDNUM = '20009'
STARTNUM = '19122'

class Balls(object):
    numlis = []

    def __init__(self):
        self.start_url = 'https://datachart.500.com/dlt/history/newinc/history.php?limit=100&sort=0'
      
        # 创建 ChromeOptions 对象
        options = webdriver.ChromeOptions()
        options.add_argument('--ignore-certificate-errors')  # 忽略证书错误
        options.add_argument('--disable-extensions')  # 禁用扩展
        options.add_argument('--headless')  # 无头模式示例
        options.add_argument('--disable-gpu')  # 禁用 GPU 加速
        options.add_argument('--no-sandbox')  # 禁用沙盒
    
        # 初始化 WebDriver
        self.web = webdriver.Chrome(options=options)
        

    def run_spider(self):
        self.web.get(self.start_url)
        # # self.web.find_element('link187')
        # start = self.web.find_element('start')
        # end = self.web.find_element('end')
        # start.clear()
        # end.clear()  # 先清除
        # time.sleep(1.5)
        # start.send_keys('00001')
        # end.send_keys(ENDNUM)
        # sub = self.web.find_element(by='xpath', value=
        #     '//*[@id="container"]/div/table/tbody/tr[1]/td/div/div[1]/div/table/tbody/tr/td[2]/img')
        # time.sleep(0.8)
        # sub.click()
        # print(self.web.page_source)
        return self.web.page_source

    def get_spider_source(self):
        sourse = self.run_spider()
        print(sourse)
        html = etree.HTML(sourse)
        i = 0
        result = '0'
        while(result !=  STARTNUM):
            i+=1
            stage_num = html.xpath(
                '//tbody[@id="tdata"]/tr[{}]/td[1]/text()'.format(i))
            red_num1 = html.xpath(
                '//tbody[@id="tdata"]/tr[{}]/td[2]/text()'.format(i))
            red_num2 = html.xpath(
                '//tbody[@id="tdata"]/tr[{}]/td[3]/text()'.format(i))
            red_num3 = html.xpath(
                '//tbody[@id="tdata"]/tr[{}]/td[4]/text()'.format(i))
            red_num4 = html.xpath(
                '//tbody[@id="tdata"]/tr[{}]/td[5]/text()'.format(i))
            red_num5 = html.xpath(
                '//tbody[@id="tdata"]/tr[{}]/td[6]/text()'.format(i))
            red_num6 = html.xpath(
                '//tbody[@id="tdata"]/tr[{}]/td[7]/text()'.format(i))
            blue_num = html.xpath(
                '//tbody[@id="tdata"]/tr[{}]/td[8]/text()'.format(i))

            print(stage_num)
            result = stage_num[0]
            self.numlis.append([stage_num[0], red_num1[0], red_num2[0],
                                red_num3[0], red_num4[0], red_num5[0], red_num6[0], blue_num[0]])

        for data in list(reversed(self.numlis)):
            self.write_to_csv('ball.csv', data)

    def write_to_csv(self, pathfile, data):
        if not os.path.exists(pathfile):
            with open(pathfile, 'w') as f:
                csv.writer(f, dialect='excel').writerow(['Date Time','red1','red2','red3','red4','red5','red6','blue'])
        f = open(pathfile, 'a', newline='')
        wr = csv.writer(f, dialect='excel')
        wr.writerow(data)


b = Balls()
b.get_spider_source()
