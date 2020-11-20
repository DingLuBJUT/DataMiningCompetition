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

import datetime
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder


class TaxInfo:
    def __init__(self, data):

        self.data = data
        self.data['year'] = self.data['START_DATE'].apply(lambda x: x if x is None else datetime.datetime.strptime(x, '%Y/%m/%d')).apply(lambda x: x.year)
        self.data['month'] = self.data['START_DATE'].apply(lambda x: x if x is None else datetime.datetime.strptime(x, '%Y/%m/%d')).apply(lambda x: x.month)

        self.data_type = {
            'tax_categories_num': 'int64',
            'tax_items_num': 'int64',
            'total_tax_amount': 'int64',
            'pay_tax_num': 'int64',
            'max_categories': 'category',
            'min_categories': 'category',
            'max_pay_categories': 'category',
            'min_pay_categories': 'category',
            'unique_year_num': 'int64',
            'unique_month_num': 'int64'
        }

        self.fill_values = {
            'tax_categories_num': -1,
            'tax_items_num': -1,
            'total_tax_amount': -1,
            'pay_tax_num': -1,
            'max_categories': '-1',
            'min_categories': '-1',
            'max_pay_categories': '-1',
            'min_pay_categories': '-1',
            'unique_year_num': -1,
            'unique_month_num': -1
        }
        self.drop_columns = [
            'START_DATE',
            'END_DATE',
            'TAX_CATEGORIES',
            'TAX_ITEMS',
            'TAXATION_BASIS',
            'TAX_RATE',
            'DEDUCTION',
            'TAX_AMOUNT',
            'year',
            'month'
        ]

        self.distinct_features = [
            'id',
            'tax_categories_num',
            'tax_items_num',
            'total_tax_amount',
            'pay_tax_num',
            'max_categories',
            'min_categories',
            'max_pay_categories',
            'min_pay_categories',
            'unique_year_num',
            'unique_month_num'
        ]

        return

    @staticmethod
    def drop_id(data_frame):
        data_frame.drop(['id'], inplace=True, axis=1)
        return

    def convert_data_type(self, data_frame):
        for name in self.data_type.keys():
            data_frame[name] = data_frame[name].astype(self.data_type[name])
        return data_frame

    def fill_nan(self, data_frame):
        for name in self.fill_values.keys():
            data_frame[name] = data_frame[name].fillna(self.fill_values[name])
        return data_frame

    def label_encoder(self):
        label_encode = LabelEncoder()
        for name in self.data_type.keys():
            if self.data_type[name] == 'category':
                element = self.data[name].unique().tolist()
                label_encode = label_encode.fit(element)
                self.data[name] = label_encode.transform(self.data[name])
        return

    def tax_categories_num(self):
        trans_data = self.data[['id', 'TAX_CATEGORIES']].groupby(['id']).agg({'TAX_CATEGORIES': pd.Series.nunique}).reset_index()
        trans_data.columns = ['id', 'tax_categories_num']
        self.data = self.data.merge(trans_data, on='id', how='inner')
        return

    def tax_items_num(self):
        trans_data = self.data[['id', 'TAX_ITEMS']].groupby(['id']).agg({'TAX_ITEMS': pd.Series.nunique}).reset_index()
        trans_data.columns = ['id', 'tax_items_num']
        self.data = self.data.merge(trans_data, on='id', how='inner')
        return

    def total_tax_amount(self):
        trans_data = self.data[['id', 'TAX_AMOUNT']].groupby(['id']).sum().reset_index()
        trans_data.columns = ['id', 'total_tax_amount']
        self.data = self.data.merge(trans_data, on='id', how='inner')
        return

    def pay_tax_num(self):
        trans_data = self.data[['id']]
        trans_data['index'] = [1] * len(trans_data)
        trans_data = trans_data.groupby(['id']).count().reset_index()
        trans_data.columns = ['id', 'pay_tax_num']
        self.data = self.data.merge(trans_data, on='id', how='inner')
        return

    def max_amount_categories(self):
        trans_data = self.data[['id', 'TAX_CATEGORIES', 'TAX_AMOUNT']]
        trans_data = trans_data.groupby(['id']).max().reset_index()
        trans_data.columns = ['id', 'max_categories', 'max_tax_amount']
        trans_data.drop(['max_tax_amount'], inplace=True, axis=1)
        self.data = self.data.merge(trans_data, on='id', how='inner')
        return

    def min_amount_categories(self):
        trans_data = self.data[['id', 'TAX_CATEGORIES', 'TAX_AMOUNT']]
        trans_data = trans_data.groupby(['id']).min().reset_index()
        trans_data.columns = ['id', 'min_categories', 'min_tax_amount']
        trans_data.drop(['min_tax_amount'], inplace=True, axis=1)
        self.data = self.data.merge(trans_data, on='id', how='inner')
        return

    def max_num_categories(self):
        trans_data = self.data[['id', 'TAX_CATEGORIES']]
        trans_data['index'] = [1] * len(trans_data)
        trans_data = trans_data.groupby(['id', 'TAX_CATEGORIES']).count().reset_index()
        trans_data.columns = ['id', 'TAX_CATEGORIES', 'count_index']
        trans_data = trans_data.merge(trans_data[['id', 'count_index']].groupby(['id']).max().reset_index(),
                                      on=['id', 'count_index'], how='inner')
        trans_data.columns = ['id', 'max_pay_categories', 'max_pay_categories_num']
        trans_data.drop(['max_pay_categories_num'], inplace=True, axis=1)
        self.data = self.data.merge(trans_data, on=['id'], how='inner')
        return

    def min_num_categories(self):
        trans_data = self.data[['id', 'TAX_CATEGORIES']]
        trans_data['index'] = [1] * len(trans_data)
        trans_data = trans_data.groupby(['id', 'TAX_CATEGORIES']).count().reset_index()
        trans_data.columns = ['id', 'TAX_CATEGORIES', 'count_index']
        trans_data = trans_data.merge(trans_data[['id', 'count_index']].groupby(['id']).min().reset_index(),
                                      on=['id', 'count_index'], how='inner')
        trans_data.columns = ['id', 'min_pay_categories', 'min_pay_categories_num']
        trans_data.drop(['min_pay_categories_num'], inplace=True, axis=1)
        self.data = self.data.merge(trans_data, on=['id'], how='inner')
        return

    def unique_year_num(self):
        trans_data = self.data[['id', 'year']].groupby(['id']).agg({'year': pd.Series.nunique}).reset_index()
        trans_data.columns = ['id', 'unique_year_num']
        self.data = self.data.merge(trans_data, on='id', how='inner')
        return

    def unique_month_num(self):
        trans_data = self.data[['id', 'month']].groupby(['id']).agg({'month': pd.Series.nunique}).reset_index()
        trans_data.columns = ['id', 'unique_month_num']
        self.data = self.data.merge(trans_data, on='id', how='inner')
        return

    def feature_process_v1(self):
        # 行去重
        self.data.drop_duplicates(subset=['id',
                                          'START_DATE',
                                          'END_DATE',
                                          'TAX_CATEGORIES',
                                          'TAX_ITEMS',
                                          'TAXATION_BASIS',
                                          'TAX_RATE',
                                          'DEDUCTION',
                                          'TAX_AMOUNT'], inplace=True)

        self.tax_categories_num()
        self.tax_items_num()
        self.total_tax_amount()
        self.pay_tax_num()
        self.max_amount_categories()
        self.min_amount_categories()
        self.max_num_categories()
        self.min_num_categories()
        self.unique_year_num()
        self.unique_month_num()
        self.label_encoder()
        self.data.drop(self.drop_columns, inplace=True, axis=1)
        self.data.drop_duplicates(subset=self.distinct_features, inplace=True)
        return self.data