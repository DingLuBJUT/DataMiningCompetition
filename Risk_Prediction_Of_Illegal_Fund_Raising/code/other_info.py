# -*- coding:utf-8 -*-
"""
Tools for Risk Prediction Of Illegal Fund Raising

Description:
this file for tools with basic functions.

"""
# 2020/11/07,Junlu,Ding,create


class OtherInfo:
    def __init__(self, data):
        self.data_type = {
            'brand_num': 'int64',
            'legal_judgment_num': 'int64'
        }

        self.useless_columns = [
            'patent_num'
        ]

        self.fill_values = {
            'brand_num': 0,
            'legal_judgment_num': 0
        }

        self.distinct_features = [
            'id',
            'brand_num',
            'legal_judgment_num'
        ]

        self.data = data
        return

    def convert_data_type(self, data_frame):
        for name in self.data_type.keys():
            data_frame[name] = data_frame[name].astype(self.data_type[name])
        return data_frame

    def fill_nan(self, data_frame):
        for name in self.fill_values.keys():
            data_frame[name] = data_frame[name].fillna(self.fill_values[name])
        return data_frame

    def feature_process_v1(self):
        self.data.drop(self.useless_columns, axis=1, inplace=True)
        self.data = self.data.fillna(-1)
        self.data.drop_duplicates(subset=self.distinct_features, inplace=True)
        self.data.drop_duplicates(subset=['id'], inplace=True)
        return self.data
