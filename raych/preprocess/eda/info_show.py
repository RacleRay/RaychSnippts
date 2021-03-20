#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   deal_stop_w.py
@Author  :   Racle
@Version :   1.0
@Desc    :   None
'''

import json


def analysis():
    stop_1 = set()
    stop_2 = []
    with open('stopwords.txt', 'r', encoding='utf-8') as f:
        for i in f.readlines():
            stop_1.add(i.strip())
    with open('stopwords_zh.txt', 'r', encoding='utf-8') as f:
        for i in f.readlines():
            if i.strip() not in stop_1:
                stop_2.append(i.strip())
                print('只存在stopwords_zh中：', i.strip())


def show_json(path):
    with open(path, 'r') as f:
        line = f.readlines()
        print(len(line))
        for i in line:
            print(json.loads(i))
            print(len(json.loads(i)))
            print(type(json.loads(i)))


if __name__ == '__main__':
    # analysis()
    show_json('sentiment_analysis_testb.json')