import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold, chi2
import gc
import os
from datetime import date
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
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

from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype

def reduce_mem_usage(df, use_float16=False):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.
    """

    start_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            continue
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df

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
    sparse_features = train.select_dtypes(include='object').columns.tolist()
    dense_features = train.select_dtypes(include='int').columns.tolist()
    sparse_features.remove('y')
    print(len(sparse_features), sparse_features)
    print(len(dense_features), dense_features)

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
    # 数值型特征归一化
    scaler = StandardScaler()
    df[dense_features] = scaler.fit_transform(df[dense_features])

    return df


'''
unknown值填充
Percentage of "unknown" in job： 288 / 45211
Percentage of "unknown" in education： 1857 / 45211
Percentage of "unknown" in contact： 13020 / 45211
Percentage of "unknown" in poutcome： 36959 / 45211
'''


def fill_unknown(df, ues_rf_interpolation=True, use_knn_interpolation=False):
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

    if use_knn_interpolation:
        for i in fill_attrs:
            if df[df[i] == 'unknown']['y'].count() / len(df) < 0.25:
                df = knn_interpolation(df, i)
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


def knn_interpolation(df, i):
    tmp = df.copy()
    data = encoding(tmp)

    test_data = data[data[i] == 'unknown']
    test_data.drop(i, inplace=True)
    train_data = data[data[i] != 'unknown']
    trainY = train_data.pop(i)

    test_data[i] = train_predict_unknown(train_data, trainY, test_data)
    data = pd.concat([train_data, test_data])

    return data


def train_rf(trainX, trainY, testX):
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(trainX, trainY)
    test_predictY = forest.predict(testX).astype(int)
    return pd.DataFrame(test_predictY, index=testX.index)


def train_knn(trainX, trainY, testX):
    knn = KNeighborsClassifier()
    knn = knn.fit(trainX, trainY)
    test_predictY = knn.predict(testX).astype(int)
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


def mutual_info_classif_filter(data, plot=False):
    train = data.copy()
    y = train.pop('y')
    xbest = SelectKBest(mutual_info_classif, k='all').fit(train, y)

    scores = pd.DataFrame(xbest.scores_)
    columns = pd.DataFrame(data.columns)
    colscores = pd.concat([columns, scores], axis=1)
    colscores.columns = ['col', 'score']
    if plot:
        print(colscores.sort_values(by='score', axis=0, ascending=False))
    # filter = colscores[colscores['score'] > 0]['col'].values
    drop_columns = colscores[colscores['score'] <= 0]['col'].values
    print("mutual_info_classif_filter del feature: ", drop_columns)
    data.drop(drop_columns, axis=1, inplace=True)
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
    # if os.path.exists(path):
    #     train = pd.read_csv(path)
    # else:
    PATH = "data/"
    train = pd.read_csv(PATH + 'bank-full.csv', sep=';')
    # train = reduce_mem_usage(train, use_float16=True)
    print(train.shape)
    train = data_preprocess(train)
    print(train.shape)
    print(train.head(5))

    # feature selection
    train = variance_filter(train)
    train = corr_filter(train, plot=True)
    train = mutual_info_classif_filter(train)
    path = 'process_data/process_data.csv'
    train.to_csv(path, index=False)

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
