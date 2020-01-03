import tensorflow as tf
from utils import inputs, test
from models.DNN import DNN
from models.FM import FM
from layers.Prediction import PredictionLayer


def DeepFM(linear_feature_columns, dnn_feature_columns, embedding_size=8, dnn_hidden_units=(128, 128),
           l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0.0,
           dnn_activation='relu', dnn_use_bn=False, task='binary'):

    features = inputs.build_input_features(linear_feature_columns + dnn_feature_columns)
    inputs_list = list(features.values())
    sparse_embedding_list, dense_value_list = inputs.input_from_feature_columns(features, dnn_feature_columns,
                                                                                embedding_size, l2_reg_embedding,
                                                                                init_std, seed)

    fm_input = inputs.concat_fun(sparse_embedding_list, axis=1)
    fm_logit = FM()(fm_input)

    dnn_input = inputs.combined_dnn_input(sparse_embedding_list, dense_value_list)
    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed)(dnn_input)
    dnn_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(dnn_out)

    final_logit = tf.keras.layers.add([fm_logit, dnn_logit])

    output = PredictionLayer(task)(final_logit)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model


def DeepFM_test(sparse_feature_num, dense_feature_num):
    model_name = "DeepFM"
    sample_size = 8
    x, y, feature_columns = test.get_test_data(sample_size, sparse_feature_num, dense_feature_num)

    model = DeepFM(feature_columns, feature_columns, dnn_dropout=0.5)
    test.check_model(model, model_name, x, y)


if __name__ == '__main__':
    # DeepFM_test(1,1)
    DeepFM_test(2, 2)
