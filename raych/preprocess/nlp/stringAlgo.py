#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   stringAlgo.py
@Author  :   Racle
@Version :   1.0
@Desc    :   一些字符串处理工具函数
'''


###########################################################################
# 最长公共子序列

def longest_common_string_dp(str1, str2):
    "动态规划: return: string, str1中end idx，str2中end idx"
    len1 = len(str1)
    len2 = len(str2)
    result = ''
    common_end_in_str1 = 0
    common_end_in_str2 = 0
    common_len = 0

    # M[i][j]: 以str1[i], str2[j]结尾的最长公共子串的的长度
    M = [[None] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        M[i][0] = 0
    for j in range(len2 + 1):
        M[0][j] = 0
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if str1[i -1] == str2[j - 1]:
                M[i][j] = M[i-1][j-1] + 1
                if M[i][j] > common_len:
                    common_len = M[i][j]
                    common_end_in_str1 = i
                    common_end_in_str2 = j
            else:
                M[i][j] = 0

    result = str1[common_end_in_str1 - common_len: common_end_in_str1]

    return result, common_end_in_str1, common_end_in_str2


def longest_common_string(str1, str2):
    "滑动窗口  return: string, str1中end idx，str2中end idx "
    len1 = len(str1)
    len2 = len(str2)
    result = ''
    common_end_in_str1 = 0
    common_end_in_str2 = 0
    common_len = 0

    # 固定str1，滑动str2与str1进行比较，str2从右向左移动
    i = 1
    while i < len1 + len2:
        str1_begin = str2_begin = 0
        tmp_len = 0
        if i < len1:
            str1_begin =  len1 - i  # len1 - 1, len1 - 2, ..., 0, ..., 0
        else:
            str2_begin =  i - len1  # 0，       0，     , ..., 1，..., len2 - 1

        j = 0
        while (str1_begin + j < len1) and (str2_begin + j < len2):
            if str1[str1_begin + j] == str2[str2_begin + j]:
                tmp_len += 1
            else:
                if tmp_len > common_len:
                    common_len = tmp_len
                    common_end_in_str1 = str1_begin + j
                    common_end_in_str2 = str2_begin + j
                else:
                    tmp_len = 0
            j += 1
        i += 1

    result = str1[common_end_in_str1 - common_len: common_end_in_str1]

    return result, common_end_in_str1, common_end_in_str2


###########################################################################
# 最长重复子串

def maxPrefixLen(str1, str2):
    "排序后的 suffix，查找其公共前缀"
    i = 0
    while i < len(str1) and i < len(str2):
        if str1[i] == str2[i]:
            i += 1
        else:
            break
        i += 1
    return i


def getMaxCommonStr(string):
    "返回最长重复子串。 使用suffix后缀，可以从不同的idx开始获得子串。而prefix只会从相同的开头开始获得子串"
    length = len(string)
    suffixList = []
    i = 0
    while i < length:
        suffixList.append(string[i: ])
        i += 1
    suffixList.sort()

    commonStr = ''
    maxLen = 0
    i = 0
    while i < length - 1:
        tmpLen = maxPrefixLen(suffixList[i], suffixList[i + 1])
        if tmpLen > maxLen:
            maxLen = tmpLen
            commonStr = suffixList[i][: tmpLen]
        i += 1

    return commonStr


###########################################################################
# 字符串处理

def text_segmentate(text, maxlen, seps='\n', strips=None):
    """将文本按照标点符号划分为若干个短句
    """
    text = text.strip().strip(strips)
    if seps and len(text) > maxlen:
        pieces = text.split(seps[0])
        text, texts = '', []
        for i, p in enumerate(pieces):
            if text and p and len(text) + len(p) > maxlen - 1:
                texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
                text = ''
            if i + 1 == len(pieces):
                text = text + p
            else:
                text = text + p + seps[0]
        if text:
            texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
        return texts
    else:
        return [text]


###########################################################################
# 字符串全角半角

def strQ2B(ustring):
    """全角符号转对应的半角符号
    """
    rstring = ''
    for uchar in ustring:
        inside_code = ord(uchar)
        # 全角空格直接转换
        if inside_code == 12288:
            inside_code = 32
        # 全角字符（除空格）根据关系转化
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


###########################################################################
# 分词匹配demo ，没使用，有开源包

def backward_maximal_matching(string, vocab):
    "逆向匹配比正向更准一些"
    result = []
    end_pos = len(string)

    while end_pos > 0:
        found = False
        for start_pos in range(end_pos):
            if string[start_pos: end_pos] in vocab:
                found = True
                result.append(string[start_pos: end_pos])
                break
        if found:
            end_pos = start_pos
        else: # 单字输出
            result.append(string[end_pos - 1])
            end_pos -= 1

    return result




if __name__ == '__main__':
    string = "你好啊朋友你好啊"
    print(getMaxCommonStr(string))


    txt = """wiki软件由软件设计模式社区开发，用来书写与讨论模式语言。沃德·坎宁安于1995年3月25日成立第一个wiki网站：WikiWikiWeb，用来补充他自己经营的软件设计模式网站。他发明wiki这个名字以及相关概念，并且实现第一个wiki引擎。坎宁安说自己是根据檀香山的Wiki Wiki公车取名的，“wiki”在夏威夷语为“快速”之意，这是他到檀香山学会的第一个夏威夷语[来源请求]，故他将“wiki-wiki”作为“快速”的意思以避免将“这东西”取名为“快速网”（quick-web）[4][3][5]。

坎宁安说，wiki的构想来自他自己在1980年代晚期利用苹果电脑HyperCard程序作出的一个小功能[6]。HyperCard类似名片整理程序，可用来纪录人物与相关事物。HyperCard管理许多称为“卡片”的资料，每张卡片上都可划分字段、加上图片、有样式的文字或按钮等等，而且这些内容都可在查阅卡片的同时修改编辑。HyperCard类似于后来的网页，但是缺乏一些重要特征。

坎宁安认为原来的HyperCard程序十分有用，但创造卡片与卡片之间的链接却很困难。于是他不用HyperCard程序原本的创造链接功能，而改用“随选搜索”的方式自己增添了一个新的链接功能。用户只要将链接输入卡片上的一个特殊字段，而这个字段每一行都有一个按钮。按下按钮时如果卡片已经存在，按钮就会带用户去那张卡片，否则就发出哔声，而继续压着按钮不放，程序就会为用户产生一张卡片。

坎宁安向他的朋友展示了这个程序和他自己写的人事卡片，往往会有人指出卡片之中的内容不太对，他们就可当场利用HyperCard初始的功能修正内容，并利用坎宁安加入的新功能补充链接。

坎宁安后来在别处又写了这样的功能，而且这次他还增加了多用户写作功能。新功能之一是程序会在每一次任何一张卡片被更改时，自动在“最近更改”卡片上增加一个连往被更改卡片的链接。坎宁安自己常常看“最近更改”卡片，而且还会注意到空白的说明字段会让他想要描述一下更改的摘要[7]。"""
    print(text_segmentate(txt, 30, seps=['\n', '。', '？', '！']))