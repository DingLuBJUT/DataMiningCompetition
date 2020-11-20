# -*- coding:utf-8 -*-
"""
Tools for Risk Prediction Of Illegal Fund Raising

Description:
annual_report_info

"""
# 2020/11/20,Junlu,Ding,create
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

"""

 id:企业唯一标识,
 ANCHEYEAR:年度,
 STATE:状态,
 # FUNDAM:资金数额,
 # MEMNUM:成员人数,
 # FARNUM:农民人数,
 # ANNNEWMEMNUM:本年度新增成员人数,
 # ANNREDMEMNUM:本年度退出成员人数,
 EMPNUM:从业人数,
 EMPNUMSIGN:从业人数是否公示,
 BUSSTNAME:经营状态名称,
 COLGRANUM:其中高校毕业生人数经营者,
 RETSOLNUM:其中退役士兵人数经营者,
 DISPERNUM:其中残疾人人数经营者,
 UNENUM:其中下岗失业人数经营者,
 COLEMPLNUM:其中高校毕业生人数雇员,
 RETEMPLNUM:其中退役士兵人数雇员, 
 DISEMPLNUM:其中残疾人人数雇员, 
 UNEEMPLNUM:其中下岗失业人数雇员, 
 WEBSITSIGN:是否有网站标志, 
 FORINVESTSIGN:是否有对外投资企业标志, 
 STOCKTRANSIGN:有限责任公司本年度是否发生股东股权转让标志, 
 PUBSTATE:公示状态：1 全部公示，2部分公示,3全部不公示

"""


class AnnualReportInfo:
    def __init__(self, data):
        self.data = data

        self.useless_columns = [
            'ANCHEYEAR',
            'STATE',
            'FUNDAM',
            'MEMNUM',
            'FARNUM',
            'ANNNEWMEMNUM',
            'ANNREDMEMNUM',
            'EMPNUM',
            'EMPNUMSIGN',
            'BUSSTNAME',
            'COLGRANUM',
            'RETSOLNUM',
            'DISPERNUM',
            'UNENUM',
            'COLEMPLNUM',
            'RETEMPLNUM',
            'DISEMPLNUM',
            'UNEEMPLNUM',
            'WEBSITSIGN',
            'FORINVESTSIGN',
            'STOCKTRANSIGN',
            'PUBSTATE'

            #             'ANCHEYEAR_0',
            #             'ANCHEYEAR_1',
            #             'ANCHEYEAR_2',
            #             'ANCHEYEAR_3',
            #             'STATE_0',
            #             'STATE_1',
            #             'STATE_2',
            #             'EMPNUMSIGN_0',
            #             'EMPNUMSIGN_1',
            #             'EMPNUMSIGN_2',
            #             'BUSSTNAME_0',
            #             'BUSSTNAME_1',
            #             'BUSSTNAME_2',
            #             'BUSSTNAME_3',
            #             'BUSSTNAME_4',
            #             'WEBSITSIGN_0',
            #             'WEBSITSIGN_1',
            #             'WEBSITSIGN_2',
            #             'FORINVESTSIGN_0',
            #             'FORINVESTSIGN_1',
            #             'FORINVESTSIGN_2',
            #             'STOCKTRANSIGN_0',
            #             'STOCKTRANSIGN_1',
            #             'STOCKTRANSIGN_2'
            #             'PUBSTATE_0',
            #             'PUBSTATE_1',
            #             'PUBSTATE_2',
            #             'PUBSTATE_3'
        ]

        self.data_type = {
            'ANCHEYEAR_unique_num': 'int64',
            'STATE_unique_num': 'int64',
            'EMPNUMSIGN_unique_num': 'int64',
            'BUSSTNAME_unique_num': 'int64',
            'WEBSITSIGN_unique_num': 'int64',
            'FORINVESTSIGN_unique_num': 'int64',
            'STOCKTRANSIGN_unique_num': 'int64',
            'PUBSTATE_unique_num': 'int64',
            'EMPNUM_max_num': 'int64',
            'COLGRANUM_max_num': 'int64',
            'RETSOLNUM_max_num': 'int64',
            'DISPERNUM_max_num': 'int64',
            'UNENUM_max_num': 'int64',
            'COLEMPLNUM_max_num': 'int64',
            'RETEMPLNUM_max_num': 'int64',
            'DISEMPLNUM_max_num': 'int64',
            'UNEEMPLNUM_max_num': 'int64'

            #             'ANCHEYEAR_0': 'category',
            #             'ANCHEYEAR_1': 'category',
            #             'ANCHEYEAR_2': 'category',
            #             'ANCHEYEAR_3': 'category',
            #             'STATE_0': 'category',
            #             'STATE_1': 'category',
            #             'STATE_2': 'category'
            #             'EMPNUMSIGN_0': 'category',
            #             'EMPNUMSIGN_1': 'category',
            #             'EMPNUMSIGN_2': 'category',
            #             'BUSSTNAME_0': 'category',
            #             'BUSSTNAME_1': 'category',
            #             'BUSSTNAME_2': 'category',
            #             'BUSSTNAME_3': 'category',
            #             'BUSSTNAME_4': 'category',
            #             'WEBSITSIGN_0': 'category',
            #             'WEBSITSIGN_1': 'category',
            #             'WEBSITSIGN_2': 'category',
            #             'FORINVESTSIGN_0': 'category',
            #             'FORINVESTSIGN_1': 'category',
            #             'FORINVESTSIGN_2': 'category',
            #             'STOCKTRANSIGN_0': 'category',
            #             'STOCKTRANSIGN_1': 'category',
            #             'STOCKTRANSIGN_2': 'category',
            #             'PUBSTATE_0': 'category',
            #             'PUBSTATE_1': 'category',
            #             'PUBSTATE_2': 'category',
            #             'PUBSTATE_3': 'category'

        }
        self.fill_values = {
            'ANCHEYEAR_unique_num': -1,
            'STATE_unique_num': -1,
            'EMPNUMSIGN_unique_num': -1,
            'BUSSTNAME_unique_num': -1,
            'WEBSITSIGN_unique_num': -1,
            'FORINVESTSIGN_unique_num': -1,
            'STOCKTRANSIGN_unique_num': -1,
            'PUBSTATE_unique_num': -1,
            'EMPNUM_max_num': -1,
            'COLGRANUM_max_num': -1,
            'RETSOLNUM_max_num': -1,
            'DISPERNUM_max_num': -1,
            'UNENUM_max_num': -1,
            'COLEMPLNUM_max_num': -1,
            'RETEMPLNUM_max_num': -1,
            'DISEMPLNUM_max_num': -1,
            'UNEEMPLNUM_max_num': -1

            #             'ANCHEYEAR_0': '-1',
            #             'ANCHEYEAR_1': '-1',
            #             'ANCHEYEAR_2': '-1',
            #             'ANCHEYEAR_3': '-1',
            #             'STATE_0': '-1',
            #             'STATE_1': '-1',
            #             'STATE_2': '-1'
            #             'EMPNUMSIGN_0': '-1',
            #             'EMPNUMSIGN_1': '-1',
            #             'EMPNUMSIGN_2': '-1',
            #             'BUSSTNAME_0': '-1',
            #             'BUSSTNAME_1': '-1',
            #             'BUSSTNAME_2': '-1',
            #             'BUSSTNAME_3': '-1',
            #             'BUSSTNAME_4': '-1',
            #             'WEBSITSIGN_0': '-1',
            #             'WEBSITSIGN_1': '-1',
            #             'WEBSITSIGN_2': '-1',
            #             'FORINVESTSIGN_0': '-1',
            #             'FORINVESTSIGN_1': '-1',
            #             'FORINVESTSIGN_2': '-1',
            #             'STOCKTRANSIGN_0': '-1',
            #             'STOCKTRANSIGN_1': '-1',
            #             'STOCKTRANSIGN_2': '-1',
            #             'PUBSTATE_0': '-1',
            #             'PUBSTATE_1': '-1',
            #             'PUBSTATE_2': '-1',
            #             'PUBSTATE_3': '-1'
        }

        self.distinct_features = {
            'ANCHEYEAR_unique_num',
            'STATE_unique_num',
            'EMPNUMSIGN_unique_num',
            'BUSSTNAME_unique_num',
            'WEBSITSIGN_unique_num',
            'FORINVESTSIGN_unique_num',
            'STOCKTRANSIGN_unique_num',
            'PUBSTATE_unique_num',
            'EMPNUM_max_num',
            'COLGRANUM_max_num',
            'RETSOLNUM_max_num',
            'DISPERNUM_max_num',
            'UNENUM_max_num',
            'COLEMPLNUM_max_num',
            'RETEMPLNUM_max_num',
            'DISEMPLNUM_max_num',
            'UNEEMPLNUM_max_num'

            #             'ANCHEYEAR_0',
            #             'ANCHEYEAR_1',
            #             'ANCHEYEAR_2',
            #             'ANCHEYEAR_3',
            #             'STATE_0',
            #             'STATE_1',
            #             'STATE_2'
            #             'EMPNUMSIGN_0',
            #             'EMPNUMSIGN_1',
            #             'EMPNUMSIGN_2',
            #             'BUSSTNAME_0',
            #             'BUSSTNAME_1',
            #             'BUSSTNAME_2',
            #             'BUSSTNAME_3',
            #             'BUSSTNAME_4',
            #             'WEBSITSIGN_0',
            #             'WEBSITSIGN_1',
            #             'WEBSITSIGN_2',
            #             'FORINVESTSIGN_0',
            #             'FORINVESTSIGN_1',
            #             'FORINVESTSIGN_2',
            #             'STOCKTRANSIGN_0',
            #             'STOCKTRANSIGN_1',
            #             'STOCKTRANSIGN_2',
            #             'PUBSTATE_0',
            #             'PUBSTATE_1',
            #             'PUBSTATE_2',
            #             'PUBSTATE_3'
        }

        return

    def drop_columns(self):
        self.data.drop(self.useless_columns, inplace=True, axis=1)
        return

    def convert_data_type(self, data_frame):
        for name in self.data_type.keys():
            data_frame[name] = data_frame[name].astype(self.data_type[name])
        return data_frame

    def fill_nan(self, data_frame):
        for name in self.fill_values.keys():
            data_frame[name] = data_frame[name].fillna(self.fill_values[name])
        return data_frame

    def diff_status_num(self):
        col_names = [
            'ANCHEYEAR',
            'STATE',
            'EMPNUMSIGN',
            'BUSSTNAME',
            'WEBSITSIGN',
            'FORINVESTSIGN',
            'STOCKTRANSIGN',
            'PUBSTATE'
        ]
        for name in col_names:
            trans_data = self.data[['id', name]]
            trans_data = trans_data.groupby(['id']).agg({name: pd.Series.nunique}).reset_index()
            trans_data.columns = ['id', name + '_unique_num']
            self.data = self.data.merge(trans_data, on=['id'], how='inner')
        return

    def max_num(self):
        col_names = [
            'EMPNUM',
            'COLGRANUM',
            'RETSOLNUM',
            'DISPERNUM',
            'UNENUM',
            'COLEMPLNUM',
            'RETEMPLNUM',
            'DISEMPLNUM',
            'UNEEMPLNUM'
        ]
        for name in col_names:
            trans_data = self.data[['id', name]]
            trans_data[name] = trans_data[name].fillna(-1)
            trans_data = trans_data.groupby(['id']).max().reset_index()
            trans_data.columns = ['id', name + '_max_num']
            self.data = self.data.merge(trans_data, on=['id'], how='inner')
        return

    def one_hot(self):

        col_names = [
            'ANCHEYEAR',
            'STATE',
            'EMPNUMSIGN',
            'BUSSTNAME',
            'WEBSITSIGN',
            'FORINVESTSIGN',
            'STOCKTRANSIGN',
            'PUBSTATE'
        ]
        enc = OneHotEncoder()
        for name in col_names:
            self.data[name] = self.data[name].astype('str')
            feature_size = len(self.data[name].unique().tolist())
            enc.fit(self.data[[name]])
            trans_data = pd.DataFrame(enc.transform(self.data[[name]]).toarray())
            trans_data = pd.concat([self.data[['id']], trans_data], axis=1)
            trans_data = trans_data.groupby(['id']).sum().reset_index()
            trans_data_columns = []
            for i in range(feature_size):
                trans_data_columns.append(name + '_' + str(i))
            trans_data.columns = ['id'] + trans_data_columns


            self.data = self.data.merge(trans_data, on=['id'], how='inner')
        return

    def feature_process_v1(self):
        self.diff_status_num()
        self.max_num()
        #         self.one_hot()
        self.drop_columns()
        self.data.drop_duplicates(subset=self.distinct_features, inplace=True)
        return self.data




