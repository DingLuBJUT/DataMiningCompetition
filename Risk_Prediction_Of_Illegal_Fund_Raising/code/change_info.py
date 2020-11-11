# -*- coding:utf-8 -*-
"""
Tools for Risk Prediction Of Illegal Fund Raising

Description:
this file for tools with basic functions.

"""
# 2020/11/19,Junlu,Ding,create


import datetime
import pandas as pd
from sklearn.preprocessing import LabelEncoder

"""
id:企业唯一标识, 
bgxmdm:变更信息代码
bgq:变更前(不是企业ID)
bgh:变更后(不是企业ID)
bgrq:变更日期
* bgq等于bgh的情况不存在

1、变更信息代码label encoder
2、企业变更总次数
3、企业不同变更代码个数

4、企业不同bgq个数
5、企业不同bgh个数
6、企业对应变更代码对应不同bgq平均数
7、企业对应变更代码对应不同bgh平均数

8、企业bgq作为bgh被使用平均次数
9、企业bgh作为bgq被使用平均次数

# bgq是否作为bgh被使用
# bgh是否作为bgq被使用


** 时间相关 **

1、同一天(同一时间)变更次数最大值
2、同一天(同一时间)变更次数最小值
3、同一天(同一时间)不同变更代码最大值
4、同一天(同一时间)不同变更代码最小值

5、不同日期(xxxx-xx-xx)个数
6、最大日期
7、最小日期


d8071a739aa75a3bc6898f27ae7c6b3bf4ce1dab0d6df19d,137.0,696844db768cef6c,89b9c45b399f58d8,2015080315  1642.0
f000950527a6feb6ddaa8f26bf9f04a19cf81a32733e7347,137.0,2cae8fd090601edc,696844db768cef6c,20170316101927.0
"""


class ChangeInfo:
    def __init__(self, data):
        self.data = data
        self.data_type = {
            'total_change_num': 'int64',
            'diff_bgxmdm_num': 'int64',
            'bgq_num': 'int64',
            'bgh_num': 'int64',
            'bgxmdm_bgq_num': 'int64',
            'bgxmdm_bgh_num': 'int64',
            'mean_bgq_as_bgh_num': 'int64',
            'mean_bgh_as_bgq_num': 'int64',
            'change_max_num_day': 'int64',
            'change_min_num_day': 'int64',
            'identification_max_num_day': 'int64',
            'identification_min_num_day': 'int64'
        }
        self.drop_columns = [
                                'bgxmdm',
                                'bgh',
                                'bgq',
                                'bgrq'
                            ]

        self.distinct_features = [
                                      'id',
                                      'total_change_num',
                                      'diff_bgxmdm_num',
                                      'bgq_num',
                                      'bgh_num',
                                      'bgxmdm_bgq_num',
                                      'bgxmdm_bgh_num',
                                      'mean_bgq_as_bgh_num',
                                      'mean_bgh_as_bgq_num',
                                      'change_max_num_day',
                                      'change_min_num_day',
                                      'identification_max_num_day',
                                      'identification_min_num_day'
                                 ]

        self.fill_values = {
            'total_change_num': -1,
            'diff_bgxmdm_num': -1,
            'bgq_num': -1,
            'bgh_num': -1,
            'bgxmdm_bgq_num': -1,
            'bgxmdm_bgh_num': -1,
            'mean_bgq_as_bgh_num': -1,
            'mean_bgh_as_bgq_num': -1,
            'change_max_num_day': -1,
            'change_min_num_day': -1,
            'identification_max_num_day': -1,
            'identification_min_num_day': -1
        }
        return

    def convert_data_type(self, data_frame):
        for name in self.data_type.keys():
            data_frame[name] = data_frame[name].astype(self.data_type[name])
        return data_frame

    def fill_nan(self, data_frame):
        for name in self.fill_values.keys():
            data_frame[name] = data_frame[name].fillna(self.fill_values[name])
        return data_frame

    def spare_time(self):
        self.data['bgrq'] = self.data['bgrq'].apply(lambda x: str(x)[0:10])
        return

    def change_num_day(self):
        """
        同一天(同一时间)变更次数最大值
        同一天(同一时间)变更次数最小值
        :return:
        """
        self.data['index'] = [1] * len(self.data)
        trans_data = self.data[['id', 'bgrq', 'index']].groupby(['id', 'bgrq']).count().reset_index()
        trans_data.drop(['bgrq'], inplace=True, axis=1)
        self.data.drop(['index'], inplace=True, axis=1)
        max_trans_data = trans_data.groupby(['id']).max().reset_index()
        min_trans_data = trans_data.groupby(['id']).min().reset_index()
        max_trans_data.columns = ['id', 'change_max_num_day']
        min_trans_data.columns = ['id', 'change_min_num_day']
        self.data = self.data.merge(max_trans_data, on='id', how='inner')
        self.data = self.data.merge(min_trans_data, on='id', how='inner')
        return

    def identification_num_day(self):
        """
        同一天(同一时间)不同变更代码最大值
        同一天(同一时间)不同变更代码最小值
        :return:
        """
        trans_data = self.data[['id', 'bgrq', 'bgxmdm']].groupby(['id', 'bgrq'])
        trans_data = trans_data.agg({'bgxmdm': pd.Series.nunique}).reset_index()
        trans_data.drop(['bgrq'], inplace=True, axis=1)
        max_trans_data = trans_data.groupby(['id']).max().reset_index()
        min_trans_data = trans_data.groupby(['id']).min().reset_index()
        max_trans_data.columns = ['id', 'identification_max_num_day']
        min_trans_data.columns = ['id', 'identification_min_num_day']
        self.data = self.data.merge(max_trans_data, on='id', how='inner')
        self.data = self.data.merge(min_trans_data, on='id', how='inner')
        return

    # def label_encoder(self):
    #     label_encode = LabelEncoder()
    #     self.data['bgxmdm'] = label_encode.fit_transform(self.data['bgxmdm'])
    #     return

    def total_change_num(self):
        """
        企业变更总次数
        :return:
        """
        self.data['total_change_num'] = [1] * len(self.data)
        trans_data = self.data[['id', 'total_change_num']].groupby(['id']).count().reset_index()
        self.data.drop(['total_change_num'], inplace=True, axis=1)
        self.data = self.data.merge(trans_data, on='id', how='inner')
        return

    def identification_num(self):
        """
        企业不同变更代码个数
        :return:
        """
        trans_data = self.data[['id', 'bgxmdm']].groupby(['id']).agg({'bgxmdm': pd.Series.nunique}).reset_index()
        trans_data.columns = ['id', 'diff_bgxmdm_num']
        self.data = self.data.merge(trans_data, on='id', how='inner')
        return

    def change_before_num(self):
        """
        企业不同bgq个数
        :return:
        """
        trans_data = self.data[['id', 'bgq']].groupby(['id']).agg({'bgq': pd.Series.nunique}).reset_index()
        trans_data.columns = ['id', 'bgq_num']
        self.data = self.data.merge(trans_data, on='id', how='inner')
        return

    def change_after_num(self):
        """
        企业不同bgh个数
        :return:
        """
        trans_data = self.data[['id', 'bgh']].groupby(['id']).agg({'bgh': pd.Series.nunique}).reset_index()
        trans_data.columns = ['id', 'bgh_num']
        self.data = self.data.merge(trans_data, on='id', how='inner')
        return

    def identification_before_num(self):
        """
        企业对应变更代码对应不同bgq平均数
        :return:
        """
        trans_data = self.data[['bgxmdm', 'bgq']].groupby(['bgxmdm'])
        trans_data = trans_data.agg({'bgq': pd.Series.nunique}).reset_index()
        trans_data.columns = ['bgxmdm', 'bgxmdm_bgq_num']
        self.data = self.data.merge(trans_data, on='bgxmdm', how='inner')
        trans_data = self.data[['id', 'bgxmdm_bgq_num']].groupby(['id'])
        trans_data = trans_data.mean().reset_index()
        self.data.drop(['bgxmdm_bgq_num'], inplace=True, axis=1)
        self.data = self.data.merge(trans_data, on='id', how='inner')
        return

    def identification_after_num(self):
        """
        企业对应变更代码对应不同bgh平均数
        :return:
        """
        trans_data = self.data[['bgxmdm', 'bgh']].groupby(['bgxmdm'])
        trans_data = trans_data.agg({'bgh': pd.Series.nunique}).reset_index()
        trans_data.columns = ['bgxmdm', 'bgxmdm_bgh_num']
        self.data = self.data.merge(trans_data, on='bgxmdm', how='inner')
        trans_data = self.data[['id', 'bgxmdm_bgh_num']].groupby(['id'])
        trans_data = trans_data.mean().reset_index()
        self.data.drop(['bgxmdm_bgh_num'], inplace=True, axis=1)
        self.data = self.data.merge(trans_data, on='id', how='inner')
        return

    def before_as_after(self):
        """
        企业bgq作为bgh被使用平均次数
        :return:
        """
        df_bgq = self.data[['bgq']]
        df_bgh = self.data[['bgh']]
        df_bgq.columns = ['change_id']
        df_bgh.columns = ['change_id']

        df_bgq = pd.DataFrame({"change_id": pd.Series(df_bgq['change_id'].unique()).tolist()})
        trans_data = df_bgq.merge(df_bgh, on='change_id', how='inner')
        trans_data['index'] = [1] * len(trans_data)
        trans_data = trans_data.groupby(['change_id']).count().reset_index()
        trans_data.columns = ['bgq', 'bgq_as_bgh_num']
        trans_data = self.data[['id', 'bgq']].merge(trans_data, on='bgq', how='left').fillna(0)

        trans_data.drop(['bgq'], axis=1, inplace=True)
        trans_data = trans_data.groupby(['id']).mean().reset_index()
        trans_data.columns = ['id', 'mean_bgq_as_bgh_num']
        self.data = self.data.merge(trans_data, on='id', how='inner')
        return

    def after_as_before(self):
        """
        企业bgh作为bgq被使用平均次数
        :return:
        """
        df_bgq = self.data[['bgq']]
        df_bgh = self.data[['bgh']]
        df_bgq.columns = ['change_id']
        df_bgh.columns = ['change_id']

        df_bgh = pd.DataFrame({"change_id": pd.Series(df_bgh['change_id'].unique()).tolist()})
        trans_data = df_bgh.merge(df_bgq, on='change_id', how='inner')
        trans_data['index'] = [1] * len(trans_data)
        trans_data = trans_data.groupby(['change_id']).count().reset_index()
        trans_data.columns = ['bgh', 'bgh_as_bgq_num']
        trans_data = self.data[['id', 'bgh']].merge(trans_data, on='bgh', how='left').fillna(0)
        trans_data.drop(['bgh'], axis=1, inplace=True)
        trans_data = trans_data.groupby(['id']).mean().reset_index()
        trans_data.columns = ['id', 'mean_bgh_as_bgq_num']
        self.data = self.data.merge(trans_data, on='id', how='inner')
        return

    def feature_process_v1(self):
        # 行去重
        self.data.drop_duplicates(subset=['id', 'bgxmdm', 'bgq', 'bgh', 'bgrq'], inplace=True)
        # 企业变更次数
        self.total_change_num()
        self.identification_num()
        self.change_before_num()
        self.change_after_num()
        self.identification_before_num()
        self.identification_after_num()
        self.before_as_after()
        self.after_as_before()
        self.change_num_day()
        self.identification_num_day()
        self.convert_data_type(self.data)
        self.data.drop(self.drop_columns, axis=1, inplace=True)
        self.data.drop_duplicates(subset=self.distinct_features, inplace=True)
        return self.data

