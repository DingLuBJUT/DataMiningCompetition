# -*- coding:utf-8 -*-

"""
reccap:实缴资本
    数值信息，空值用-1值填充
    
enttypeminu:企业类型细类
    种类信息，空值用'-1'填充
    
venind:风险行业
    种类信息，空值用'-1'填充, 也可以删除
    
opfrom:经营期限起
opto:经营期限止
    时间信息，数值信息，空置用-1填充
    计算开始year
    计算结束year
    计算时间差
    
compform:组织形式
    删除


enttypeitem:企业类型小类
    种类信息，空值用'-1'填充
    
empnum:从业人数
    数值信息，空值用-1值填充，也可以删除
    
regcap:注册资本（金）
    数值信息，空值用-1值填充
    
industryco:行业细类代码
    种类信息，空值用'-1'填充
    
oploc:经营场所
    种类信息，空值用'-1'填充

oplocdistrict:行政区划代码   
    种类信息，空值用'-1'填充
    
regtype:主题登记类型
    种类信息，空值用'-1'填充
    
townsign:是否城镇
    种类信息，空值用'-1'填充
    
adbusign:是否广告经营
    种类信息，空值用'-1'填充
    
jobid:职位标识
    种类信息，空值用'-1'填充
    
orgid:机构标识
    种类信息，空值用'-1'填充
    
state:状态
    种类信息，空值用'-1'填充
    
opform:经营方式
    种类信息，空值用'-1'填充
    
enttype:企业类型
    种类信息，空值用'-1'填充
    
opscope:经营范围
    包括中文语言特征，暂时直接删除

dom:经营地址
     种类信息，空值用'-1'填充，种类过多，labelEncoder视为数值类型， 也可以删除
     
industryphy:行业类别代码
      种类信息，空值用'-1'填充
      
enttypegb:企业（机构）类型
    种类信息，空值用'-1'填充
    
    
"""

import datetime
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class BaseInfo:
    def __init__(self, data, type='train'):
        self.data = data
        self.data_type = {
            'opfrom': 'time',
            'opto': 'time',
            'reccap': 'int64',
            'enttypeminu': 'category',
            'venind': 'category',
            'enttypeitem': 'category',
            'empnum': 'int64',
            'regcap': 'int64',
            'industryco': 'category',
            'oploc': 'category',
            'oplocdistrict': 'category',
            'regtype': 'category',
            'townsign': 'category',
            'adbusign': 'category',
            'jobid': 'category',
            'orgid': 'category',
            'state': 'category',
            'enttype': 'category',
            'dom': 'category',
            'industryphy': 'category',
            'enttypegb': 'category',
            'opform': 'category'
        }
        self.useless_columns = [
            'ptbusscope',
            'midpreindcode',
            'protype',
            'forreccap',
            'congro',
            'forregcap',
            'exenum',
            'parnum',
            'compform',
            'opscope',
            'id'
        ]

        if type == 'test':
            self.useless_columns.remove('id')
            self.useless_columns.append('score')

        return

    def fill_nan(self, name, value, column_type):
        self.data[name] = self.data[name].fillna(value)
        self.data[name] = self.data[name].astype(column_type)
        return

    def label_encoder(self, name, column_type):
        label_encode = LabelEncoder()
        value_data = self.data[self.data[name].isnull() == 0]
        null_data = self.data[self.data[name].isnull() != 0]
        value_data[name] = label_encode.fit_transform(value_data[name])
        self.data = pd.concat([null_data, value_data])
        self.data[name] = self.data[name].astype(column_type)
        return

    def drop_columns(self, drop_columns):
        self.data.drop(drop_columns, axis=1, inplace=True)
        return

    def unify_time(self, name):
        value_data = self.data[self.data[name].isnull() == 0]
        null_data = self.data[self.data[name].isnull() != 0]
        value_data[name] = value_data[name].apply(
            lambda x: x if len(x) > 10 else (x + " 00:00:00"))
        value_data[name] = value_data[name].apply(
            lambda x: x if x is None else datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        value_data[name] = value_data[name].apply(lambda x: x.year)
        self.data = pd.concat([value_data, null_data])
        return

    def feature_process_v1(self):
        for name in self.data_type.keys():
            if self.data_type.get(name) == 'category':
                self.label_encoder(name, 'category')
            elif self.data_type.get(name) == 'int64':
                mean = self.data[self.data[name].isnull() == 0][name].mean()
                self.fill_nan(name, mean, 'int64')
            elif self.data_type.get(name) == 'time':
                self.unify_time(name)
                mode = self.data[name].mode()[0]
                self.fill_nan(name, mode, 'int64')

        self.data['diff_year'] = (self.data['opto'] - self.data['opfrom']).astype('int64')
        self.drop_columns(self.useless_columns)
        return self.data

    def feature_process_v2(self):
        for name in self.data_type.keys():
            if self.data_type.get(name) == 'category':
                self.fill_nan(name, '-1', 'category')
                # self.label_encoder(name, 'category')
            elif self.data_type.get(name) == 'int64':
                mean = self.data[self.data[name].isnull() == 0][name].mean()
                self.fill_nan(name, mean, 'int64')
            elif self.data_type.get(name) == 'time':
                self.unify_time(name)
                mode = self.data[name].mode()[0]
                self.fill_nan(name, mode, 'int64')

        self.data['diff_year'] = (self.data['opto'] - self.data['opfrom']).astype('int64')
        self.drop_columns(self.useless_columns)
        return self.data








