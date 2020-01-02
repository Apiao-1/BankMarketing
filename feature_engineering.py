import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
import gc
import os
from datetime import date
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import warnings

warnings.filterwarnings('ignore')
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
pd.set_option('max_colwidth', 200)


# 编码：yes == 1, no == 0
def get_dummy_from_bool(row, column_name):
    return 1 if row[column_name] == 'yes' else 0


# 用平均值替换异常值
def get_correct_values(row, column_name, threshold, df):
    if row[column_name] <= threshold:
        return row[column_name]
    else:
        mean = df[df[column_name] <= threshold][column_name].mean()
        return mean


def encoding(df):
    # 将yes与no的进行编码
    bool_columns = ['default', 'housing', 'loan', 'y']
    for bool_col in bool_columns:
        df[bool_col] = df.apply(lambda row: get_dummy_from_bool(row, bool_col), axis=1)

    # 对类别特征独热编码
    cat_columns = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
    for col in cat_columns:
        df = pd.concat(
            [df.drop(col, axis=1), pd.get_dummies(df[col], prefix=col, prefix_sep='_',
                                                  drop_first=True, dummy_na=False)], axis=1)

    return df


'''
unknown值填充
Percentage of "unknown" in job： 288 / 45211
Percentage of "unknown" in education： 1857 / 45211
Percentage of "unknown" in contact： 13020 / 45211
Percentage of "unknown" in poutcome： 36959 / 45211
'''


def fill_unknown(df, ues_rf_interpolation=True):
    fill_attrs = ['job', 'education', 'contact', 'poutcome']
    # 出现次数少于5%的字段直接删除
    for i in reversed(fill_attrs):
        if df[df[i] == 'unknown']['y'].count() / len(df) < 0.05:
            df = df[df[i] != 'unknown']
            fill_attrs.remove(i)

    # 出现次数少于25%的可以用随机森林插值
    if ues_rf_interpolation:
        for i in fill_attrs:
            if df[df[i] == 'unknown']['y'].count() / len(df) < 0.25:
                df = rf_interpolation(df, i)
                fill_attrs.remove(i)

    # 出现次数大于25%的保留并视做一类新的类别
    print("remain unknown feature: ", fill_attrs)

    return df


def rf_interpolation(df, i):
    tmp = df.copy()
    data = encoding(tmp)

    test_data = data[data[i] == 'unknown']
    test_data.drop(i, inplace=True)
    train_data = data[data[i] != 'unknown']
    trainY = train_data.pop(i)

    test_data[i] = train_predict_unknown(train_data, trainY, test_data)
    data = pd.concat([train_data, test_data])

    return data


def train_predict_unknown(trainX, trainY, testX):
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(trainX, trainY)
    test_predictY = forest.predict(testX).astype(int)
    return pd.DataFrame(test_predictY, index=testX.index)


# 剔除EDA发现的异常值
def drop_incorrect(df):
    df = df[df['campaign'] <= 24]
    df = df[df['previous'] <= 24]
    df = df[df['pdays'] <= 400]
    return df


def feature_count(data, features):
    feature_name = 'count'
    for i in features:
        feature_name += '_' + i
    temp = data.groupby(features).size().reset_index().rename(columns={0: feature_name})
    data = data.merge(temp, 'left', on=features)
    return data, feature_name


def q80(x):
    return x.quantile(0.8)


def q30(x):
    return x.quantile(0.3)


def feature_engineering(df):
    sparse = train.select_dtypes(include='object').columns.tolist()
    dense = train.select_dtypes(include='int').columns.tolist()
    sparse.remove('y')
    print(len(sparse), sparse)
    print(len(dense), dense)

    # 年龄分箱
    df['age_buckets'] = pd.qcut(df['age'], 20, labels=False, duplicates='drop')
    age = df[['age_buckets', 'balance']]
    age = age.groupby(['age_buckets']).agg({'balance': ['mean']}).reset_index()
    age.columns = ['age_buckets', 'age_mean']
    df = pd.merge(df, age, on=['age_buckets'], how='left')

    # 与同龄人balance均值的差值
    df['age_balance'] = df['balance'] - df['age_mean']

    # 贷款、房贷与违约的联合特征
    loan = df[['default', 'housing', 'loan', 'balance']]
    loan = loan.groupby(['default', 'housing', 'loan']).agg({'balance': ['min', 'max', 'mean', 'std', 'skew', 'median',
                                                                         q80, q30, pd.DataFrame.kurt, 'mad',
                                                                         np.ptp]}).reset_index()
    loan.columns = ['default', 'housing', 'loan', 'loan_min', 'loan_max', 'loan_mean', 'loan_std', 'loan_skew',
                    'loan_median', 'loan_q80', 'loan_q30', 'loan_kurt', 'loan_mad', 'loan_ptp']
    df = pd.merge(df, loan, on=['default', 'housing', 'loan'], how='left')

    # 职业与余额的联合特征
    job = df[['job', 'balance']]
    job = job.groupby(['job']).agg({'balance': ['min', 'max', 'mean', 'std', 'skew', 'median', q80,
                                                q30, pd.DataFrame.kurt, 'mad', np.ptp]}).reset_index()
    job.columns = ['job', 'job_min', 'job_max', 'job_mean', 'job_std', 'job_skew',
                   'job_median', 'job_q80', 'job_q30', 'job_kurt', 'job_mad', 'job_ptp']
    df = pd.merge(df, job, on=['job'], how='left')

    # 与同龄人balance均值的差值
    df['job_balance'] = df['balance'] - df['job_mean']

    # 婚姻、教育水平与余额的联合特征
    tmp = df[['marital', 'education', 'balance']]
    tmp = tmp.groupby(['marital', 'education']).agg({'balance': ['min', 'max', 'mean', 'std', 'skew', 'median', q80,
                                                                 q30, pd.DataFrame.kurt, 'mad', np.ptp]}).reset_index()
    tmp.columns = ['marital', 'education', 'balance_min', 'balance_max', 'balance_mean', 'balance_std', 'balance_skew',
                   'balance_median', 'balance_q80', 'balance_q30', 'balance_kurt', 'balance_mad', 'balance_ptp']
    df = pd.merge(df, tmp, on=['marital', 'education'], how='left')

    print("after feature engineering:", df.shape)

    return df


def data_preprocess(df):
    df = drop_incorrect(df)
    df = fill_unknown(df, ues_rf_interpolation=False)
    df = feature_engineering(df)

    # drop irrelevant columns
    # df = df.drop(columns=['pdays'])

    # impute incorrect values and drop original columns
    # df['campaign_cleaned'] = df.apply(lambda row: get_correct_values(row, 'campaign', 34, df), axis=1)
    # df['previous_cleaned'] = df.apply(lambda row: get_correct_values(row, 'previous', 34, df), axis=1)
    # df = df.drop(columns=['campaign', 'previous'])

    df = encoding(df)


    return df


def variance_filter(data):
    X = data.drop(columns='y', inplace=False)
    filter = VarianceThreshold(threshold=0.001)
    mask = filter.fit(X)._get_support_mask()
    origin_columns = X.columns.values
    drop_columns = origin_columns[mask == False]
    print("variance_filter del feature: ", drop_columns)
    data.drop(drop_columns, axis=1, inplace=True)
    print(data.shape)

    return data


def corr_filter(data, plot=False):
    sns.set(rc={'figure.figsize': (25, 25)})
    corr = data.corr()
    if plot:
        plt.figure()
        ax = sns.heatmap(corr, linewidths=.5, annot=True, cmap="YlGnBu", fmt='.1g')
        plt.show()

    # Drop highly correlated features
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if corr.iloc[i, j] >= 0.995:
                if columns[j]:
                    columns[j] = False
    # feature_columns = data.columns[columns == True].values
    drop_columns = data.columns[columns == False].values
    data.drop(drop_columns, axis=1, inplace=True)
    print("corr_filter del feature: ", drop_columns)
    print(data.shape)
    return data


if __name__ == '__main__':
    path = 'process_data/process_data.csv'
    if os.path.exists(path):
        train = pd.read_csv(path)
    else:
        PATH = "data/"
        train = pd.read_csv(PATH + 'bank-full.csv', sep=';')
        print(train.shape)
        train = data_preprocess(train)
        print(train.shape)
        print(train.head(5))

        # feature selection
        variance_data = variance_filter(train)
        del train
        gc.collect()
        corr_data = corr_filter(variance_data, plot=True)
        corr_data.to_csv(path, index=False)

    # df_new = train.copy()
    # introduce new column 'balance_buckets' to  ''
    # df_new['balance_buckets'] = pd.qcut(df_new['balance'], 50, labels=False, duplicates='drop')
    #
    # # group by 'balance_buckets' and find average campaign outcome per balance bucket
    # mean_deposit = df_new.groupby(['balance_buckets'])['y'].mean()
    #
    # # plot
    # plt.plot(mean_deposit.index, mean_deposit.values)
    # plt.title('Mean % subscription depending on account balance')
    # plt.xlabel('balance bucket')
    # plt.ylabel('% subscription')
    # plt.show()
