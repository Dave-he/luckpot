import csv

'''
红球有6个，1-32
蓝球有1个，1-16
转换为一个向量，1-32 位为 红球， 33-48 蓝球的
'''
def convert(rowList):
    data = []
    index = 0
    for i in range(48):
        data.append('0')

    for row in rowList:
        if int(row)  < 33:
            if  index < 6:
                data[int(row) -1] = '1'
            else:
                print(int(row) + 32 -1)
                data[int(row) + 32 -1] = '1'
            index+=1

    return ','.join(data)

ball = []
with open('ball.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        d = convert(row)
        print(d)
        ball.append(d)

with open('new.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    for row in ball:
        writer.writerow(row)