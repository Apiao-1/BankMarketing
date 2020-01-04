import numpy as np
import pandas as pd
from  sklearn import metrics
import matplotlib.pyplot as plt
import random


# coupon average auc 最终的评价指标对每个优惠券coupon_id单独计算预测结果的AUC值
# def metric_coupon_AUC(test_data, plot=False):
#     coupons = test_data.groupby('Coupon_id').size().reset_index(name='total')
#     aucs = []
#     for _, coupon in coupons.iterrows():
#         if coupon.total > 1:
#             X_test = test_data[test_data.Coupon_id == coupon.Coupon_id].copy()
#             X_test.drop(columns='Coupon_id', inplace=True)
#
#             # 需要去除那些只有一个标签的券，比如某张券全都是1或全都是0，这样的券无法计算AUC值
#             if len(X_test.label.unique()) != 2:
#                 continue
#             aucs.append(roc_auc_score(X_test.label, X_test.y_pred))
#
#     score = np.mean(aucs)
#     if plot:
#         print(score)
#
#     return score

model = []
accuracy_score = []
recall_score = []
precision_score = []
f1_score = []
ROC_AUC_Score = []
FPR = []
TPR = []

def cal_roc_curve(target, oof, name, threshold=0.5):
    clf_fpr, clf_tpr, thresholds = metrics.roc_curve(target, oof)
    precision, recall, thresholds = metrics.precision_recall_curve(target, oof)
    clf_roc_auc = metrics.auc(clf_fpr, clf_tpr)  # 计算auc的值
    plot_pr(clf_roc_auc, precision, recall, name)


    y_pred_clf = np.zeros(len(oof), dtype=int)
    for i in range(len(oof)):
        if oof[i] >= threshold:
            y_pred_clf[i] = 1
        else:
            y_pred_clf[i] = 0

    print('\n>>>在测试集上的表现：', metrics.accuracy_score(target, y_pred_clf))
    print('\n>>>混淆矩阵\n', metrics.confusion_matrix(target, y_pred_clf))
    print('\n>>>分类评价函数\n', metrics.classification_report(target, y_pred_clf))

    model.append(name)
    FPR.append(clf_fpr)
    TPR.append(clf_tpr)
    precision = metrics.precision_score(target, y_pred_clf, average='binary')
    accuracy_score.append(metrics.accuracy_score(target, y_pred_clf))
    recall_score.append(metrics.recall_score(target, y_pred_clf, average='binary'))
    precision_score.append(metrics.precision_score(target, y_pred_clf, average='binary'))
    f1_score.append(metrics.f1_score(target, y_pred_clf, average='binary'))
    ROC_AUC_Score.append(metrics.roc_auc_score(target, oof))



def plot_model_result():
    train = pd.DataFrame()
    train['model'] = model
    train['accuracy_score'] = accuracy_score
    train['recall_score'] = recall_score
    train['precision_score'] = precision_score
    train['f1_score'] = f1_score
    train['ROC_AUC_Score'] = ROC_AUC_Score
    print(train)


def plot_roc_curve():
    lw = 2
    plt.figure(figsize=(10, 10))
    for i in range(model):
        model_name = model[i]
        fpr, tpr = FPR[i], TPR[i]
        auc = ROC_AUC_Score[i]
        plt.plot(fpr, tpr, color='green', lw=lw, label='%s(area = %0.3f)' % (model_name, auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Comparison of classification models')
    plt.legend(loc="lower right")
    plt.show()

def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color

def plot_pr(auc_score, precision, recall, label=None):
    plt.figure(num=None, figsize=(6, 5))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('P/R (AUC=%0.2f) / %s' % (auc_score, label))
    plt.fill_between(recall, precision, alpha=0.2)
    plt.grid(True, linestyle='-', color='0.75')
    plt.plot(recall, precision, lw=1)
    plt.show()