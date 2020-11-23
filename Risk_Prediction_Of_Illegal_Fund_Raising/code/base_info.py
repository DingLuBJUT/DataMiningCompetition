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


import re
import datetime
import pandas as pd
from itertools import chain
from collections import Counter
from sklearn.preprocessing import LabelEncoder


class BaseInfo:
    def __init__(self, data, ent_prise_info, type='train'):
        self.data = data
        self.ent_prise_info = ent_prise_info

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
            # 'address_index': 'category'
            # 'id_index': 'category',
            # 'scope_num': 'int64',
            # 'is_exist_pos_kw': 'category',
            # 'special_kw_num': 'int64',
            # 'diff_kw_num': 'int64'

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
            'opscope'

            # 'scope_num',
            # 'diff_kw_num',
            # 'is_exist_pos_kw',
            # 'special_kw_num',

            # 'id',
            # 'orgid',
            # 'industryco',
            # 'dom',
            # 'enttypegb',
            # 'enttypeitem',
            # 'opfrom',
            # 'state',
            # 'adbusign',
            # 'jobid',
            # 'enttypegb',
            # 'regtype',
            # 'empnum',
            # 'venind',
            # 'enttypeminu',
            # 'oploc',
            # 'enttype',
            # 'oplocdistrict'

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

    # def split_scope(self, scopes):
    #     scopes = re.sub(u"\\（.*?）|\\{.*?}|\\[.*?]|\\【.*?】", "", scopes)
    #     scopes = re.split(r'[；; ;、, ,，。（）()\s]\s*', scopes)
    #     return scopes

    # def get_pos_corpus(self):
    #     trans_data = self.data[['id', 'opscope']].merge(self.ent_prise_info, on=['id'], how='inner')
    #     pos_scopes = trans_data[trans_data['label'] == 1]['opscope'].tolist()
    #     neg_scopes = trans_data[trans_data['label'] == 0]['opscope'].tolist()
    #
    #     pos_kws = []
    #     for line in pos_scopes:
    #         pos_kws.append(self.split_scope(line))
    #     pos_kws_dict = Counter(list(chain(*pos_kws)))
    #     pos_kws_dict = dict(sorted(pos_kws_dict.items(), key=lambda item: item[1], reverse=True))
    #     pos_kws_dict.pop('')
    #     pos_kws_dict.pop('***')
    #     pos_kws_dict.pop('**')
    #
    #     pos_kws_dict.pop('#')
    #     pos_kws_dict.pop('*******')
    #     pos_kws_dict.pop('***除外')
    #     pos_kws_dict.pop('2')
    #     pos_kws_dict.pop('3')
    #     pos_kws_dict.pop('4')
    #
    #     pos_kws = set(pos_kws_dict.keys())
    #     self.pos_mode_kws = set(dict((k, v) for k, v in pos_kws_dict.items() if v > 0).keys())
    #
    #     neg_kws = []
    #     for line in neg_scopes:
    #         neg_kws.append(self.split_scope(line))
    #     neg_kws = set(chain(*neg_kws))
    #     self.difference_kws = pos_kws - neg_kws
    #     return

    # def intersection_size(self, scopes, type):
    #     scopes = set(self.split_scope(scopes))
    #     if type == "is_exist_pos_kw":
    #         if len(self.difference_kws & scopes) > 0:
    #             return 1
    #         else:
    #             return 0
    #     else:
    #         return len(self.pos_mode_kws & scopes)

    # def op_scope(self):
    #     self.get_pos_corpus()
    #     self.data['scope_num'] = self.data['opscope'].apply(lambda x: len(self.split_scope(x)))
    #     self.data['is_exist_pos_kw'] = self.data['opscope'].apply(
    #         lambda x: self.intersection_size(x, "is_exist_pos_kw"))
    #     self.data['special_kw_num'] = self.data['opscope'].apply(lambda x: self.intersection_size(x, "special_kw_num"))
    #     self.data['diff_kw_num'] = self.data['scope_num'] - self.data['special_kw_num']
    #     return

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

    def get_organization_info(self):
        names = {
            'reccap': 'int64',
            'enttypeminu': 'category',
            'venind': 'category',
            'enttypeitem': 'category',
            'empnum': 'int64',
            'regcap': 'int64',
            'oploc': 'category',
            'regtype': 'category',
            'townsign': 'category',
            'adbusign': 'category',
            'jobid': 'category',
            'orgid': 'category',
            'state': 'category',
            'enttype': 'category',
            'opform': 'category'
        }

        for k, v in names.items():
            if v == 'int64':
                trans_data = self.data[['oplocdistrict', 'industryphy', 'industryco', 'enttypegb', k]].groupby(['oplocdistrict', 'industryphy', 'industryco', 'enttypegb']).max().reset_index()
                trans_data.columns = ['oplocdistrict', 'industryphy', 'industryco', 'enttypegb', 'max_' + k]
            elif v == 'category':
                trans_data = self.data[['oplocdistrict', 'industryphy', 'industryco', 'enttypegb', k]].groupby(['oplocdistrict', 'industryphy', 'industryco', 'enttypegb']).agg({k: pd.Series.nunique}).reset_index()
                trans_data.columns = ['oplocdistrict', 'industryphy', 'industryco', 'enttypegb', 'diff_' + k]
            self.data = self.data.merge(trans_data, on=['oplocdistrict', 'industryphy', 'industryco', 'enttypegb'], how='inner')
        return

    def feature_process_v2(self):
        # self.op_scope()
        # self.data['address_index'] = self.data['dom'].apply(lambda x:x[0:16])
        for name in self.data_type.keys():
            if self.data_type.get(name) == 'category':
                self.fill_nan(name, '-1', 'str')
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
        self.get_organization_info()
        return self.data













