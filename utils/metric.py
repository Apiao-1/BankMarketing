import numpy as np
from sklearn.metrics import roc_auc_score

# coupon average auc 最终的评价指标对每个优惠券coupon_id单独计算预测结果的AUC值
def metric_coupon_AUC(test_data, plot=False):
    coupons = test_data.groupby('Coupon_id').size().reset_index(name='total')
    aucs = []
    for _, coupon in coupons.iterrows():
        if coupon.total > 1:
            X_test = test_data[test_data.Coupon_id == coupon.Coupon_id].copy()
            X_test.drop(columns='Coupon_id', inplace=True)

            # 需要去除那些只有一个标签的券，比如某张券全都是1或全都是0，这样的券无法计算AUC值
            if len(X_test.label.unique()) != 2:
                continue
            aucs.append(roc_auc_score(X_test.label, X_test.y_pred))

    score = np.mean(aucs)
    if plot:
        print(score)

    return score