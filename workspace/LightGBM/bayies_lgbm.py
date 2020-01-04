import sys

sys.path.append('/home/aistudio/external-libraries')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from bayes_opt import BayesianOptimization
from sklearn.model_selection import StratifiedKFold
import datetime
import gc
import warnings
from sklearn.metrics import roc_auc_score
import feature_engineering
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 200)


def BayesianSearch(clf, params):
    """贝叶斯优化器"""
    # 迭代次数
    num_iter = 10
    init_points = 5
    # 创建一个贝叶斯优化对象，输入为自定义的模型评估函数与超参数的范围
    bayes = BayesianOptimization(clf, params)
    # 开始优化
    bayes.maximize(init_points=init_points, n_iter=num_iter)
    # 输出结果
    params = bayes.res['max']
    print(params['max_params'])

    return params


def GBM_evaluate(min_child_samples, feature_fraction, num_leaves,lambda_l2,bagging_fraction):
    """自定义的模型评估函数"""

    # 模型固定的超参数
    param = {
        'learning_rate': .1,
        'n_estimators': 2000,
        'num_leaves': 50,
        'min_split_gain': 0,
        'min_child_weight': 1e-3,
        'min_child_samples': 21,
        'subsample': .8,
        'colsample_bytree': .8,

        'n_jobs': -1,
        'random_state': 0
    }
    global flag
    if flag:
        find_best_param(param)
        flag = False

    # 贝叶斯优化器生成的超参数
    param['min_child_samples'] = int(min_child_samples)
    param['colsample_bytree'] = float(feature_fraction)
    # param['max_depth'] = int(max_depth)
    param['num_leaves'] = int(num_leaves)
    param['subsample'] = float(bagging_fraction)
    # param['bagging_freq'] = int(bagging_freq)
    param['reg_lambda'] = float(lambda_l2)
    # param['lambda_l1'] = float(lambda_l1)
    # param['min_data_in_leaf'] = int(min_data_in_leaf)

    val = find_best_param(param)

    return val


def find_best_param(params):
    data = pd.DataFrame(y)

    # kFold cv
    models = []
    scores = []

    kf = StratifiedKFold(n_splits=3, random_state=42)
    for i, (tdx, vdx) in enumerate(kf.split(X, y)):
        X_train, X_val, y_train, y_val = X.loc[tdx], X.loc[vdx], y.loc[tdx], y.loc[vdx]
        y_true = y_val
        model = LGBMClassifier().set_params(**params)
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], early_stopping_rounds=50,
                  verbose=False)

        y_pred = model.predict_proba(X_val, num_iteration=model.best_iteration_)[:, 1]
        auc = roc_auc_score(y_true, y_pred)
        # print("AUC score at %d floder: %f" % (i, auc))
        scores.append(auc)
        models.append(model)
        data.loc[vdx, 'y_pred'] = y_pred
        # print(data['y_pred'].value_counts())

    # mean_score = np.mean(scores)
    oof = roc_auc_score(data['y'], data['y_pred'])
    # print("5-floder total mean_score:", mean_score)
    print("5-floder oof auc score:", oof)
    # print("----train %s finish!----" % model.__class__.__name__)


    global best_score, best_param
    if oof >= best_score:
        best_score = oof
        print("update best_score: %f" % best_score)
        best_param = params
        print("update best params: %s" % best_param)
    return oof


best_score = -9999
best_param = {}
flag=True
X, y = None, None

if __name__ == '__main__':
    X, y = feature_engineering.get_train_data(use_over_sampler=True)

    # 调参范围
    adj_params = {
        'min_child_samples': (3, 50),
        'feature_fraction': (0.3, 1),
        # 'max_depth': (4, 15),
        'num_leaves':(30,1300),
        'bagging_fraction': (0.3, 1),
        # 'bagging_freq': (1, 10),
        'lambda_l2': (0.1, 2),
        # 'lambda_l1': (0.1, 1),
        # 'min_data_in_leaf': (1, 150)
    }

    # 调用贝叶斯优化
    BayesianSearch(GBM_evaluate, adj_params)

    print("final best param: %s" % best_param)
    print("final best score: %f" % best_score)
