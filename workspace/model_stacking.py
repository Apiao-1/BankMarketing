from workspace.LightGBM import LightGBM
from workspace.CatBoost import CatBoost
from workspace.DeepFM import DeepFM
from utils.metric import cal_roc_curve
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import roc_auc_score
import feature_engineering

def model_stacking():
    X, y = feature_engineering.get_train_data(use_over_sampler=True)
    data = pd.DataFrame(y)

    lgbm = LightGBM.train_lgbm(plot=False)
    print(lgbm.shape)
    cat = CatBoost.train_cat(plot=False)
    print(cat.shape)
    deepFM = DeepFM.train_DeepFM()
    print(deepFM.shape)

    X = pd.concat([lgbm, cat, deepFM], axis=1)
    print(X.shape)

    models = []
    scores = []
    checkpoint_predictions = []

    kf = StratifiedKFold(n_splits=5, random_state=42)
    for i, (tdx, vdx) in enumerate(kf.split(X, y)):
        print(f'Fold : {i}')
        X_train, X_val, y_train, y_val = X.loc[tdx], X.loc[vdx], y.loc[tdx], y.loc[vdx]
        y_true = y_val

        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_true, y_pred)
        print("AUC score at %d floder: %f" % (i, auc))
        scores.append(auc)
        data.loc[vdx, 'y_pred'] = y_pred
        # print(data['y_pred'].value_counts())

    mean_score = np.mean(scores)
    oof = roc_auc_score(data['y'], data['y_pred'])
    print("5-floder total mean_score:", mean_score)
    print("5-floder oof auc score:", oof)
    print("----train %s finish!----" % 'Stacking')
    cal_roc_curve(data['y'], data['y_pred'], 'Stacking')

    return data['y_pred']

if __name__ == '__main__':
    stakcing_oof = model_stacking()

'''
(385529,)
(385529, 3)
Fold : 0
AUC score at 0 floder: 0.844169
Fold : 1
AUC score at 1 floder: 0.845082
Fold : 2
AUC score at 2 floder: 0.846117
Fold : 3
AUC score at 3 floder: 0.841820
Fold : 4
AUC score at 4 floder: 0.839050
coupon mean_score: 0.590600092808117
5-floder total mean_score: 0.8483899264561
'''