from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from utils import sampler

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 200)


def train_cat(plot=False):
    X, y = sampler.get_over_sampler_data()
    data = pd.DataFrame(y)

    # kFold cv
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
            'max_depth': 8,
            'max_bin': 127,
            'reg_lambda': 2,
            'subsample': .7,

            'one_hot_max_size': 2,
            'bootstrap_type': 'Bernoulli',
            'leaf_estimation_method': 'Newton',

            'random_state': 0
        }
        model = CatBoostClassifier().set_params(**params)
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], early_stopping_rounds=50,
                  verbose=False)

        ## plot feature importance
        if plot:
            fscores = pd.Series(model.feature_importances_, X_train.columns).sort_values(ascending=False)
            fscores.plot(kind='bar', title='Feature Importance %d' % i, figsize=(20, 10))
            plt.ylabel('Feature Importance Score')
            plt.show()

        y_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_true, y_pred)
        print("AUC score at %d floder: %f" % (i, auc))
        scores.append(auc)
        models.append(model)
        data.loc[vdx, 'y_pred'] = y_pred

    mean_score = np.mean(scores)
    oof = roc_auc_score(data['y'], data['y_pred'])
    print("5-floder total mean_score:", mean_score)
    print("5-floder oof auc score:", oof)
    print("----train catboost finish!----")

    return data['y_pred']


if __name__ == '__main__':
    train_cat(plot=False)
