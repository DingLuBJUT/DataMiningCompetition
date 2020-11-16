# -*- coding:utf-8 -*-
"""
Tools for Risk Prediction Of Illegal Fund Raising

Description:
this file for tools with basic functions.

"""
# 2020/11/07,Junlu,Ding,create

"""

id:企业唯一标识 
START_DATE:起始时间 
END_DATE:终止时间
TAX_CATEGORIES:税种 
TAX_ITEMS:税目
TAXATION_BASIS:计税依据
TAX_RATE:税率
DEDUCTION:扣除数 
TAX_AMOUNT:税额

1、企业不同税种个数
2、企业不同税目个数
3、企业总共税额
4、企业总共缴税次数
5、企业总税额最大的税种
5、企业总税额最小的税种
6、企业缴税次数最多的税种
7、企业缴税次数最少的税种
8、企业缴税不同年份个数
9、企业缴税不同月份个数

"""

import pandas as pd


class TaxInfo:
    def __init__(self, data):

        self.data = data

        return

    def tax_categories_num(self):

        return

    def tax_items_num(self):
        return

    def total_tax_amount(self):
        return

    def pay_tax_num(self):
        return

    def feature_process_v1(self):

        return