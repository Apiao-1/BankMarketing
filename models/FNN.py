import tensorflow as tf
from utils import inputs, test
from models import DNN
from layers.Prediction import PredictionLayer


def FNN(linear_feature_columns, dnn_feature_columns, embedding_size=8, dnn_hidden_units=(128, 128),
        l2_reg_embedding=1e-5, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0.0, dnn_activation='relu',
        task='binary'):
    # 将所有特征转化成tensor，返回的是OrderedDict
    features = inputs.build_input_features(linear_feature_columns + dnn_feature_columns)
    # 从OrderedDict中获取所有的value，即获取所有的tensor
    inputs_list = list(features.values())

    # 得到输入dnn的特征list，在fnn中不区分两者，因为dnn_feature_columns = linear_feature_columns
    sparse_embedding_list, dense_value_list = inputs.input_from_feature_columns(
        features, dnn_feature_columns, embedding_size, l2_reg_embedding, init_std, seed)
    dnn_input = inputs.combined_dnn_input(sparse_embedding_list, dense_value_list)

    deep_out = DNN.DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed)(dnn_input)
    deep_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(deep_out)

    output = PredictionLayer(task)(deep_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model


def FNN_test(sparse_feature_num, dense_feature_num):
    model_name = "FNN"

    sample_size = 8
    x, y, feature_columns = test.get_test_data(sample_size, sparse_feature_num, dense_feature_num)

    model = FNN(feature_columns, feature_columns, dnn_hidden_units=[32, 32], dnn_dropout=0.5)
    test.check_model(model, model_name, x, y)


if __name__ == '__main__':
    # FNN_test(1, 1)
    FNN_test(2, 4)
