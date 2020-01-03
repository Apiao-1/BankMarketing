import tensorflow as tf
from utils import inputs, test
from models.DNN import DNN
from layers.Prediction import PredictionLayer


def WDL(linear_feature_columns, dnn_feature_columns, embedding_size=8, dnn_hidden_units=(128, 128), l2_reg_linear=1e-5,
        l2_reg_embedding=1e-5, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0.0, dnn_activation='relu',
        task='binary'):
    features = inputs.build_input_features(linear_feature_columns + dnn_feature_columns)
    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = inputs.input_from_feature_columns(features, dnn_feature_columns,
                                                                                embedding_size, l2_reg_embedding,
                                                                                init_std, seed)
    dnn_input = inputs.combined_dnn_input(sparse_embedding_list, dense_value_list)

    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed)(dnn_input)
    dnn_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(dnn_out)

    linear_logit = inputs.get_linear_logit(features, linear_feature_columns, init_std=init_std, seed=seed,
                                           prefix='linear', l2_reg=l2_reg_linear)

    if len(linear_feature_columns) > 0 and len(dnn_feature_columns) > 0:  # linear + dnn
        final_logit = tf.keras.layers.add([linear_logit, dnn_logit])
    elif len(linear_feature_columns) == 0:
        final_logit = dnn_logit
    elif len(dnn_feature_columns) == 0:
        final_logit = linear_logit
    else:
        raise NotImplementedError

    output = PredictionLayer(task)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model


def WDL_test(sparse_feature_num, dense_feature_num):
    model_name = "WDL"
    sample_size = 8
    x, y, feature_columns = test.get_test_data(
        sample_size, sparse_feature_num, dense_feature_num)

    model = WDL(feature_columns, feature_columns, dnn_hidden_units=[4, 4], dnn_dropout=0.5)
    test.check_model(model, model_name, x, y)


if __name__ == '__main__':
    # WDL_test(1,1)
    WDL_test(2, 2)
