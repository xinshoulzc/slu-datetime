#coding:utf-8

from __future__ import print_function

import json
import os
import random


def swap(a, b):
    return b, a


def get_year():
    data = list()
    for i in xrange(1980, 2020):
        data.append(list([str(i)]))
        data.append(list([str(i), '年']))
        data.append(list([str(i) + '年']))
        data.append(list([str(i) + '年份']))
    # print (data)
    return data, len(data)


def get_month():
    data = list()
    month = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '十一', '十二']
    for i in xrange(1, 13):
        data.append(list([str(i), '月']))
        data.append(list([str(i) + '月']))
        data.append(list([str(i) + '月份']))
        data.append(list([month[i - 1], '月']))
        data.append(list([month[i - 1] + '月']))
        data.append(list([month[i - 1] + '月份']))
    # print (data)
    return data, len(data)


def get_week():
    data = list()
    day = ['一', '二', '三', '四', '五', '六', '日', '天']
    for i in xrange(len(day)):
        data.append(list(['周' + day[i]]))
        data.append(list(['星期' + day[i]]))
    # print (data)
    return data, len(data)


def get_day():
    data = list()
    for i in xrange(1, 31):
        data.append(list([str(i), '日']))
        data.append(list([str(i), '号']))
        data.append(list([str(i) + '日']))
        data.append(list([str(i) + '号']))
    return data, len(data)


def generate_data_one(num=500):
    y, l_y = get_year()
    m, l_m = get_month()
    d, l_d = get_day()
    w, l_w = get_week()
    output = list()
    for _ in xrange(num):
        data = list()
        answer = list()
        answer.append([0] * 8)
        answer.append([0] * 8)
        if random.randint(0, 1):
            tmp = random.randint(0, l_y - 1)
            data.extend(y[tmp])
            answer[0][4] = tmp / 4 + 1980
        if random.randint(0, 1):
            tmp = random.randint(0, l_m - 1)
            data.extend(m[tmp])
            answer[0][5] = tmp / 6 + 1
        if random.randint(0, 1):
            tmp = random.randint(0, l_d - 1)
            data.extend(d[tmp])
            answer[0][6] = tmp / 4 + 1
        else:
            tmp = random.randint(0, l_w - 1)
            data.extend(w[tmp])
            answer[0][7] = min(tmp / 2 + 1, 7)
        output.append([data, answer])
    return output


def generate_data_two(num=200):
    y, l_y = get_year()
    output = list()
    for _ in xrange(num):
        data = list()
        answer = list()
        answer.append([0] * 8)
        answer.append([0] * 8)
        fro = random.randint(0, l_y - 1)
        to = random.randint(0, l_y - 1)
        if fro / 4 == to / 4: continue
        if fro != min(fro, to): y_fro, y_to = swap(y[fro], y[to])
        else: y_fro, y_to = y[fro], y[to]
        data.extend(y_fro)
        data.extend(y_to)
        tmp = random.randint(0, 4)
        if tmp: data.insert(1, u'到')
        elif tmp == 1: data.insert(1, u'至')
        elif tmp == 2: data.extend(list(u'之间'))
        else: data.extend(list([u'间']))
        fro, to = min(fro, to) / 4 + 1980, max(fro, to) / 4 + 1980
        answer[0][4] = fro
        answer[1][4] = to
        output.append([data, answer])
    return output


def genrate_data_three():
    pass


def load_original_data(input_seq_path, input_target_path, factor=20):
    output = list()
    with open(input_seq_path, 'r') as input_seq_file:
        with open(input_target_path, 'r') as input_target_file:
            seq_line = input_seq_file.readline()
            target_line = input_target_file.readline()
            while seq_line and target_line:
                seq = json.loads(seq_line)
                target = json.loads(target_line)
                for _ in xrange(factor):
                    output.append([seq, target])
                seq_line = input_seq_file.readline()
                target_line = input_target_file.readline()
    return output


def write_data(output_seq='.', output_target='.', output=list()):
    random.shuffle(output)
    output_seq_path = os.path.join(output_seq, 'train', 'train.seq')
    output_target_path = os.path.join(output_target, 'train', 'train.target')
    output_seq_file = open(output_seq_path, 'w')
    output_target_file = open(output_target_path, 'w')
    for i in xrange(len(output)):
        line = ' '.join(map(lambda x: x.encode('utf-8'), output[i][0]))
        output_seq_file.write(line + '\n')
        line = ' '.join(map(lambda x: str(x), reduce(lambda x, y: x+y, output[i][1])))
        output_target_file.write(line + '\n')
    output_seq_file.close()
    output_target_file.close()

if __name__ == "__main__":
    output = list()
    output.extend(generate_data_one())
    output.extend(generate_data_two())
    output.extend(load_original_data('./data/results.json', './data/train.datetime'))
    write_data(output=output)
    print (len(output))
