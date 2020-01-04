from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import feature_engineering
from utils.metric import cal_roc_curve



pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 200)

def train_tree(model):
    X, y = feature_engineering.get_train_data(use_over_sampler=True)
    data = pd.DataFrame(y)
    # kFold cv
    models = []
    scores = []

    kf = StratifiedKFold(n_splits=5, random_state=42)
    for i, (tdx, vdx) in enumerate(kf.split(X, y)):
        print(f'Fold : {i}')
        X_train, X_val, y_train, y_val = X.loc[tdx], X.loc[vdx], y.loc[tdx], y.loc[vdx]
        y_true = y_val

        # model = RandomForestClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict_proba(X_val)[:,1]
        auc = roc_auc_score(y_true, y_pred)
        print("AUC score at %d floder: %f" % (i, auc))
        scores.append(auc)
        models.append(model)
        data.loc[vdx, 'y_pred'] = y_pred
        # print(data['y_pred'].value_counts())

    mean_score = np.mean(scores)
    oof = roc_auc_score(data['y'], data['y_pred'])
    print("5-floder total mean_score:", mean_score)
    print("5-floder oof auc score:", oof)
    print("----train %s finish!----" % model.__class__.__name__)
    cal_roc_curve(data['y'], data['y_pred'], model.__class__.__name__)

    return data['y_pred']

if __name__ == '__main__':
    rf_oof = train_tree(RandomForestClassifier(n_estimators=100,
                 criterion="gini",
                 max_depth=7,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True))
    gbdt_oof = train_tree(GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_decrease=0.))
    et_oof = train_tree(ExtraTreesClassifier(n_estimators=100,
                 criterion="gini",
                 max_depth=7,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto"))
