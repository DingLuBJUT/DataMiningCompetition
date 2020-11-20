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


def split_data(data_frame, split_ratio):
    label = data_frame[['label']]
    data_frame.drop(['label'], axis=1, inplace=True)
    train_data, val_data, train_label, val_label = train_test_split(data_frame,
                                                                    label,
                                                                    test_size=split_ratio,
                                                                    random_state=2020)
    return train_data, val_data, train_label, val_label


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


def predict_data(data_frame, model):
    label = data_frame[['label']]
    data_frame.drop(['label'], inplace=True, axis=1)
    prob = pd.DataFrame({"prob": pd.Series(model.predict_proba(data_frame)[:, 1]).tolist()})
    pred = pd.DataFrame({"pred": pd.Series([1 if prob > 0.5 else 0 for prob in prob['prob']]).tolist()})
    data_frame = pd.concat([data_frame, label], axis=1)
    predict_result = pd.concat([label, pred], axis=1)
    return data_frame, predict_result


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


def get_predict_output(model, test_data, result_path):
    """

    :param model:
    :param test_data:
    :param result_path:
    :return:
    """

    test_id = test_data['id']
    test_data.drop(['id','score'], inplace=True, axis=1)
    test_prob = model.predict_proba(test_data)[:, 1]
    result = pd.DataFrame({'id': test_id, 'score': test_prob})
    result.to_csv(result_path, index=False)
    return


def plot_shap(tree_model, val_data):
    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(val_data)
    shap.summary_plot(shap_values, val_data, max_display=100)
    return


def KDEPlot():
    # KDE(核密度估计)
    # 对直方图对一种平滑
    data = load_data()
    fg = sns.FacetGrid(data, hue='Species', size=10)
    fg.map(sns.kdeplot, "PetalLengthCm")
    fg.add_legend()
    plt.show()
    return


def plot_kde(data, column_name):
    x = data[column_name]
    sns.kdeplot(x, shade=True, color="g")
    plt.show()
    return
