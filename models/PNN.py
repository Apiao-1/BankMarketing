import tensorflow as tf
from utils import inputs, test
from models.DNN import DNN
from layers.Prediction import PredictionLayer
from layers.Product import InnerProductLayer


def PNN(dnn_feature_columns, embedding_size=8, dnn_hidden_units=(128, 128), l2_reg_embedding=1e-5, l2_reg_dnn=0,
        init_std=0.0001, seed=1024, dnn_dropout=0.0, dnn_activation='relu', task='binary'):
    features = inputs.build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = inputs.input_from_feature_columns(features, dnn_feature_columns,
                                                                                embedding_size, l2_reg_embedding,
                                                                                init_std, seed)
    inner_product = tf.keras.layers.Flatten()(InnerProductLayer()(sparse_embedding_list))

    # ipnn deep input
    linear_signal = tf.keras.layers.Reshape(
        [len(sparse_embedding_list) * embedding_size])(inputs.concat_fun(sparse_embedding_list))

    deep_input = tf.keras.layers.Concatenate()([linear_signal, inner_product])

    dnn_input = inputs.combined_dnn_input([deep_input], dense_value_list)
    deep_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed)(dnn_input)
    deep_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(deep_out)

    output = PredictionLayer(task)(deep_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model


def PNN_test(sparse_feature_num):
    model_name = "PNN"
    sample_size = 8
    x, y, feature_columns = test.get_test_data(sample_size, sparse_feature_num, sparse_feature_num)
    model = PNN(feature_columns, embedding_size=4, dnn_hidden_units=[4, 4], dnn_dropout=0.5)
    test.check_model(model, model_name, x, y)


if __name__ == '__main__':
    # PNN_test(True, True, 2)
    PNN_test(2)
    # PNN_test(True, True, 2)
    # PNN_test(True, True, 2)
