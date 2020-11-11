# -*- coding:utf-8 -*-
"""
Tools for Risk Prediction Of Illegal Fund Raising

Description:
this file for tools with basic functions.

"""
# 2020/11/19,Junlu,Ding,create


"""

 positive_negtive:新闻正负面性
    积极、中立、消极
 public_date:发布日期
 
 
 1、发布新闻总个数(没有用-1填充)
 2、消极新闻个数(没有用0填充)
 3、中立新闻个数(没有用0填充)
 4、积极新闻个数(没有用0填充)
 5、最早发布新闻年份
 6、最新发布新闻年份
 8、不同年份个数
 9、发布新闻总个数/不同年份个数

"""

import datetime
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class NewsInfo:
    def __init__(self,data):
        self.data_type = {
            'positive_negtive':'category'
        }
        self.data = data
        return

    def label_encoder(self, name, column_type):
        label_encode = LabelEncoder()
        value_data = self.data[self.data[name].isnull() == 0]
        null_data = self.data[self.data[name].isnull() != 0]
        value_data[name] = label_encode.fit_transform(value_data[name])
        self.data = pd.concat([null_data, value_data])
        self.data[name] = self.data[name].astype(column_type)
        return

    def news_total_num(self):
        total_num_news = self.data.groupby(['id']).count().reset_index()
        total_num_news.drop(['positive_negtive'], axis=1, inplace=True)
        total_num_news.columns = ['id','total_num']
        return total_num_news[['id','total_num']]

    def news_category_num(self):
        category_num_news = self.data.groupby(['id', 'positive_negtive']).count().reset_index()
        category_num_news.columns = ['id','category','category_num']
        category_num_news = category_num_news.reset_index(['id','category'])['category_num'].unstack()
        category_num_news = category_num_news.fillna(0).reset_index()
        category_num_news.columns = ['id', 'middle_num', 'neg_num', 'pos_num']
        return category_num_news

    def news_datetime_feature(self):
        self.data['public_date'] = self.data['public_date'].apply(
            lambda date_time: date_time if len(date_time) == 10 else '2020-10-15')
        self.data['public_date'] = self.data['public_date'].apply(
            lambda x: x if x is None else datetime.datetime.strptime(x, '%Y-%m-%d'))
        self.data['public_date'] = self.data['public_date'].apply(lambda x: x.year)
        min_date_time = self.data.groupby(['id'])[['public_date']].min().reset_index()
        min_date_time.columns = ['id','min_date_time']
        max_date_time = self.data.groupby(['id'])[['public_date']].max().reset_index()
        max_date_time.columns = ['id', 'max_date_time']
        distinct_count_date_time = self.data.groupby(['id']).agg({'public_date': pd.Series.nunique}).reset_index()
        distinct_count_date_time.columns = ['id','distinct_count_date_time']
        max_min_date_time = max_date_time.merge(min_date_time, on='id', how='inner')
        date_time_feature_news = max_min_date_time.merge(distinct_count_date_time, on='id', how='inner')
        return date_time_feature_news


    def feature_process_v1(self):
        total_num_news = self.news_total_num()
        category_num_news = self.news_category_num()
        date_time_feature_news = self.news_datetime_feature()
        self.data = total_num_news.merge(category_num_news, on='id', how='inner')
        self.data = self.data.merge(date_time_feature_news, on='id', how='inner')
        return self.data




