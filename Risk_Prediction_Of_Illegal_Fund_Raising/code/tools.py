# -*- coding:utf-8 -*-
"""
Tools for Risk Prediction Of Illegal Fund Raising

Description:
this file for tools with basic functions.

"""
# 2020/11/07,Junlu,Ding,create
from sklearn.preprocessing import LabelEncoder

import datetime
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from bubbly.bubbly import bubbleplot
from plotly.offline import iplot
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import shap



def drop_miss_data(data_frame, miss_threshold):
    """

    :param data_frame:
    :param miss_threshold:
    :return:
    """

    miss_rate = (data_frame.isnull().sum() / len(data_frame)).to_frame().reset_index()
    miss_rate.columns = ['name', 'miss_rate']
    miss_rate = miss_rate.sort_values(by='miss_rate', ascending=False)
    drop_columns = list(miss_rate[miss_rate['miss_rate'] > miss_threshold]['name'])
    print("the drop columns is:")
    print(drop_columns)
    data_frame.drop(drop_columns, inplace=True, axis=1)
    return miss_rate, data_frame


def category_number_info(data_frame):
    """

    :param data_frame:
    :return:
    """
    names = []
    is_nan = []
    is_repeat = []
    unique_num = []

    data_size = len(data_frame)
    for name in data_frame.columns:
        names.append(name)
        unique_num.append(len(data_frame[name].unique()))
        if data_frame[[name]].isnull().sum()[0] > 0:
            is_nan.append('YES')
        else:
            is_nan.append('NO')
        if len(data_frame[name].unique()) == data_size:
            is_repeat.append('NO')
        else:
            is_repeat.append('YES')

    number_info = pd.DataFrame({
        'name': names,
        'is_nan': is_nan,
        'is_repeat': is_repeat,
        'unique_num': unique_num
    })
    return number_info


def label_encoder(data_frame, column_names):
    """

    :param data_frame:
    :param column_names:
    :return:
    """
    label_encode = LabelEncoder()
    for name in column_names:
        value_data = data_frame[data_frame[name].isnull() == 0]
        null_data = data_frame[data_frame[name].isnull() != 0]
        value_data[name] = label_encode.fit_transform(value_data[name])
        data_frame = pd.concat([null_data, value_data])
    return data_frame


def fill_nan(data_frame, column_name, fill_way):
    """

    :param data_frame:
    :param column_name:
    :param fill_way:
    :return:
    """
    if fill_way == "mean":
        mean = data_frame[column_name].mean()
        data_frame[column_name] =data_frame[column_name].fillna(mean)
    elif fill_way == "mode":
        mode = data_frame[column_name].mode()[0]
        data_frame[column_name] = data_frame[column_name].fillna(mode)
    elif fill_way == "max":
        max = data_frame[column_name].max()[0]
        data_frame[column_name] = data_frame[column_name].fillna(max)
    elif fill_way == "min":
        min = data_frame[column_name].min()[0]
        data_frame[column_name] = data_frame[column_name].fillna(min)
    elif fill_way == 'median':
        median = data_frame[column_name].median()
        data_frame[column_name] = data_frame[column_name].fillna(median)
    return data_frame


def predict_result(model, test_data, result_path):
    """

    :param model:
    :param test_data:
    :param result_path:
    :return:
    """

    test_id = test_data['id']
    test_data.drop(['id'], inplace=True, axis=1)
    test_prob = model.predict_proba(test_data)[:, 1]
    result = pd.DataFrame({'id': test_id, 'score': test_prob})
    # result = result.groupby("id").agg('mean').reset_index()
    result.to_csv(result_path, index=False)
    return


def get_balance_data(data_frame):
    """

    :param data_frame:
    :return:
    """
    pos_data = data_frame[data_frame['label'] == 1]
    neg_data = data_frame[data_frame['label'] == 0]
    neg_data = neg_data.sample(n=len(pos_data), axis=0, random_state=2020, replace=True)
    data_frame = pd.concat([neg_data, pos_data])
    return data_frame


def plot_hist(data_frame, feature_name, label_name):
    fg = sns.FacetGrid(data_frame, col=label_name)
    fg.map(plt.hist, feature_name)
    plt.show()
    return


def plot_bubble(data_frame, x_name, y_name, z_name, color_name, size_name):

    data_frame['id'] = list(range(len(data_frame)))
    figure = bubbleplot(dataset=data_frame,
                        x_column=x_name,
                        y_column=y_name,
                        z_column=z_name,
                        bubble_column='id',
                        size_column=size_name,
                        color_column=color_name,
                        x_title=x_name,
                        y_title=y_name,
                        z_title=z_name,
                        title='bubble',
                        x_logscale=False,
                        scale_bubble=0.1,
                        height=600)
    iplot(figure, config={'scrollzoom': True})
    plt.show()
    data_frame.drop(['id'], axis=1, inplace=True)
    return


def plot_kde(data_frame, feature_name, label_name):
    fg = sns.FacetGrid(data_frame, hue=label_name, size=10)
    fg.map(sns.kdeplot, feature_name)
    fg.add_legend()
    plt.show()
    return


def split_data(data_frame, split_ratio):
    label = data_frame[['label']]
    data_frame.drop(['label'], axis=1, inplace=True)
    train_data, val_data, train_label, val_label = train_test_split(data_frame,
                                                                    label,
                                                                    test_size=split_ratio,
                                                                    random_state=2020)
    return train_data, val_data, train_label, val_label


def get_time_year(data_frame, column_name):
    data_frame[column_name] = data_frame[column_name].apply(lambda x: x if len(x) > 10 else (x + " 00:00:00"))
    data_frame[column_name] = data_frame[column_name].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    year = data_frame[column_name].apply(lambda x: x.year)
    return year


def predict_data(data_frame, model):
    label = data_frame['label']
    data_frame.drop(['label'], inplace=True, axis=1)
    prob = pd.Series(model.predict_proba(data_frame)[:, 1], name='prob')
    data_frame = pd.concat([data_frame.reset_index(), label.reset_index(), prob.reset_index()], axis=1)
    pred = pd.Series([1 if prob > 0.5 else 0 for prob in data_frame['prob']], name='pred')
    data_frame = pd.concat([data_frame, pred.reset_index()], axis=1)
    return data_frame


def plot_confusion_matrix(data_frame):
    ax = plt.subplot()

    label = data_frame['label']
    pred = data_frame['pred']
    conf_matrix = confusion_matrix(label, pred)
    print(conf_matrix)

    neg_to_neg = conf_matrix[0, 0]
    pos_to_neg = conf_matrix[0, 1]
    neg_to_pos = conf_matrix[1, 0]
    pos_to_pos = conf_matrix[1, 1]

    pos_acc = pos_to_pos / (pos_to_pos + pos_to_neg)
    neg_acc = neg_to_neg / (neg_to_neg + neg_to_pos)

    print("neg acc is: %f" % (neg_acc))
    print("pos acc is: %f" % (pos_acc))

    sns.heatmap(conf_matrix, annot=True, ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['0', '1'])
    ax.yaxis.set_ticklabels(['0', '1'])
    plt.show()

    return


def plot_SHAP(tree_model, val_data):
    values = shap.TreeExplainer(tree_model).shap_values(val_data)
    shap.summary_plot(values, tree_model)
    return



def main():
    return


if __name__ == '__main__':
    main()