import tensorflow as tf
import datetime
from tensorflow.python.keras.utils import plot_model
from utils.inputs import SparseFeat, DenseFeat
import numpy as np


def gen_sequence(dim, max_len, sample_size):
    return np.array([np.random.randint(0, dim, max_len) for _ in range(sample_size)]), np.random.randint(1, max_len + 1,
                                                                                                         sample_size)


def get_test_data(sample_size=1000, sparse_feature_num=1, dense_feature_num=1, sequence_feature=['sum', 'mean', 'max'],
                  classification=True, include_length=False, hash_flag=False, prefix=''):
    feature_columns = []
    model_input = {}

    if 'weight' in sequence_feature:
        feature_columns.append(SparseFeat(prefix + "weighted_seq_seq_length", 1, embedding=False))

        s_input, s_len_input = gen_sequence(2, 3, sample_size)

        model_input[prefix + "weighted_seq"] = s_input
        model_input[prefix + 'weight'] = np.random.randn(sample_size, 3, 1)
        model_input[prefix + "weighted_seq" + "_seq_length"] = s_len_input
        sequence_feature.pop(sequence_feature.index('weight'))

    for i in range(sparse_feature_num):
        dim = np.random.randint(1, 10)
        feature_columns.append(SparseFeat(prefix + 'sparse_feature_' + str(i), dim, hash_flag, tf.int32))
    for i in range(dense_feature_num):
        feature_columns.append(DenseFeat(prefix + 'dense_feature_' + str(i), 1, tf.float32))

    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            model_input[fc.name] = np.random.randint(0, fc.dimension, sample_size)
        elif isinstance(fc, DenseFeat):
            model_input[fc.name] = np.random.random(sample_size)
        else:
            s_input, s_len_input = gen_sequence(fc.dimension, fc.maxlen, sample_size)
            model_input[fc.name] = s_input
            if include_length:
                feature_columns.append(
                    SparseFeat(prefix + 'sequence_' + str(i) + '_seq_length', 1, embedding=False))
                model_input[prefix + "sequence_" + str(i) + '_seq_length'] = s_len_input

    if classification:
        y = np.random.randint(0, 2, sample_size)
    else:
        y = np.random.random(sample_size)

    return model_input, y, feature_columns


def check_model(model, model_name, x, y):
    # # 设定格式化模型名称，以时间戳作为标记
    # model_name = "test"
    # 设定存储位置，每个模型不一样的路径
    # tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))

    model.compile('adam', 'binary_crossentropy',metrics=['binary_crossentropy'])
    # model.fit(x, y, batch_size=100, epochs=1, validation_split=0.5, callbacks=[tensorboard])
    model.fit(x, y, batch_size=100, epochs=1, validation_split=0.5)

    print(model_name + " test train valid pass!")
    print(model_name + " test pass!")

    start = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    plot_model(model, to_file=model_name + '_' + start + '.png')
