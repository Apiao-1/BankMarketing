from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler
import feature_engineering

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 200)


def over_sampler(df):
    # col = df.columns.tolist()
    X = df.copy()
    y = X.pop('y')
    print(X.shape, y.shape)
    smo = SMOTE(random_state=42)
    X_smo, y_smo = smo.fit_sample(X, y)
    print(X_smo.shape, y_smo.shape)
    return X_smo, y_smo


def under_sampler(df):
    X = df.copy()
    y = X.pop('y')
    print(X.shape, y.shape)
    under = RandomUnderSampler(random_state=42)
    X_under, y_under = under.fit_sample(X, y)
    print(X_under.shape, y_under.shape)
    return X_under, y_under


def train_lgbm(X, y, plot=False):
    data = pd.DataFrame(y)

    models = []
    scores = []

    kf = StratifiedKFold(n_splits=5, random_state=42)
    for i, (tdx, vdx) in enumerate(kf.split(X, y)):
        print(f'Fold : {i}')
        X_train, X_val, y_train, y_val = X.loc[tdx], X.loc[vdx], y.loc[tdx], y.loc[vdx]
        y_true = y_val

        params = {
            'learning_rate': .05,
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
        model = LGBMClassifier().set_params(**params)
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], early_stopping_rounds=50,
                  verbose=False)

        ## plot feature importance
        if plot:
            fscores = pd.Series(model.feature_importances_, X_train.columns).sort_values(ascending=False)
            fscores.plot(kind='bar', title='Feature Importance %d' % i, figsize=(20, 10))
            plt.ylabel('Feature Importance Score')
            plt.show()

        y_pred = model.predict_proba(X_val, num_iteration=model.best_iteration_)[:, 1]
        auc = roc_auc_score(y_true, y_pred)
        print("AUC score at %d floder: %f" % (i, auc))
        scores.append(auc)
        models.append(model)
        data.loc[vdx, 'y_pred'] = y_pred

    mean_score = np.mean(scores)
    print("5-floder total mean_score:", mean_score)
    print("----train lgbm finish!----")
    print(roc_auc_score(data['y'], data['y_pred']))


if __name__ == '__main__':
    data = pd.read_csv('../process_data/process_data.csv')

    X = data.copy()
    y = X.pop('y')

    # X,y = over_sampler(data)
    # X,y = under_sampler(data)

    # tmp = pd.DataFrame(X)
    # print(tmp.shape)

    train_lgbm(X, y, plot=False)
