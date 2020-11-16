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
            'judgment': 'int64'
        }
        self.drop_columns = [
            'patent_num'
        ]

        self.fill_values = {
            'brand_num': 0,
            'judgment': 0
        }

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
        self.data.drop(self.drop_columns, axis=1, inplace=True)
        self.data = self.data.fillna(0)
        return self.data
