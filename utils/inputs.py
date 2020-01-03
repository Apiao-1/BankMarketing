from collections import OrderedDict, namedtuple

from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import Embedding, Input, Flatten
import tensorflow as tf
from tensorflow.python.keras.regularizers import l2
from layers.Linear import Linear


# 定长特征
# dimension = input type指输入的种类数，输入的字典长度
# embedding表示是否使用嵌入
# use_hash表示是否使用hash编码，设置这个值为true则不需要额外的编码操作，否则需要用LabelEncoder编码
class SparseFeat(namedtuple('SparseFeat', ['name', 'dimension', 'use_hash', 'dtype', 'embedding_name', 'embedding'])):
    __slots__ = ()

    def __new__(cls, name, dimension, use_hash=False, dtype="int32", embedding_name=None, embedding=True):
        if embedding and embedding_name is None:
            embedding_name = name
        return super(SparseFeat, cls).__new__(cls, name, dimension, use_hash, dtype, embedding_name, embedding)

    def __hash__(self):
        return self.name.__hash__()

    def __eq__(self, other):
        if self.name == other.name and self.embedding_name == other.embedding_name:
            return True
        return False

    def __repr__(self):
        return 'SparseFeat:' + self.name


class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype'])):
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype="float32"):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)

    def __hash__(self):
        return self.name.__hash__()

    def __eq__(self, other):
        if self.name == other.name:
            return True
        return False

    def __repr__(self):
        return 'DenseFeat:' + self.name


def build_input_features(feature_columns, prefix=''):
    input_features = OrderedDict()
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            input_features[fc.name] = Input(shape=(1,), name=prefix + fc.name, dtype=fc.dtype)
        elif isinstance(fc, DenseFeat):
            input_features[fc.name] = Input(shape=(fc.dimension,), name=prefix + fc.name, dtype=fc.dtype)
        else:
            raise TypeError("Invalid feature column type,got", type(fc))

    return input_features

# create Embeding layer
def  create_embedding_dict(sparse_feature_columns, embedding_size, init_std, seed, l2_reg, prefix='sparse_'):
    sparse_embedding = {feat.embedding_name:
                            Embedding(feat.dimension, embedding_size, embeddings_initializer=RandomNormal(
                                mean=0.0, stddev=init_std, seed=seed), embeddings_regularizer=l2(l2_reg),
                                      name=prefix + '_emb_' + feat.name) for feat in sparse_feature_columns}
    return sparse_embedding


# 构建所有使用嵌入的sparse特征的embedding层，dict形式
def create_embedding_matrix(feature_columns, l2_reg, init_std, seed, embedding_size, prefix=""):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat) and x.embedding, feature_columns)) if feature_columns else []
    sparse_emb_dict = create_embedding_dict(sparse_feature_columns, embedding_size, init_std, seed,
                                            l2_reg, prefix=prefix + 'sparse')
    return sparse_emb_dict


def get_linear_logit(features, feature_columns, use_bias=False, init_std=0.0001, seed=1024, prefix='linear',l2_reg=0):
    linear_emb_list, dense_input_list = input_from_feature_columns(features, feature_columns, 1, l2_reg, init_std, seed,prefix=prefix)
    if len(linear_emb_list) > 0 and len(dense_input_list) > 0:
        sparse_input = concat_fun(linear_emb_list)
        dense_input = concat_fun(dense_input_list)
        linear_logit = Linear(l2_reg, mode=2, use_bias=use_bias)([sparse_input, dense_input])
    elif len(linear_emb_list) > 0:  # 只有sparse特征
        sparse_input = concat_fun(linear_emb_list)
        linear_logit = Linear(l2_reg, mode=0, use_bias=use_bias)(sparse_input)
    elif len(dense_input_list) > 0:  # 只有dense特征
        dense_input = concat_fun(dense_input_list)
        linear_logit = Linear(l2_reg, mode=1, use_bias=use_bias)(dense_input)
    else:
        raise NotImplementedError

    return linear_logit

def embedding_lookup(sparse_embedding_dict, sparse_input_dict, sparse_feature_columns):
    embedding_vec_list = []
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name

        lookup_idx = sparse_input_dict[feature_name]
        embedding_vec_list.append(sparse_embedding_dict[embedding_name](lookup_idx))

    return embedding_vec_list


def get_dense_input(features, feature_columns):
    dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if feature_columns else []
    dense_input_list = []
    for fc in dense_feature_columns:
        dense_input_list.append(features[fc.name])
    return dense_input_list

# 获得（Embeding后的sparse特征）与dense特征
def input_from_feature_columns(features, feature_columns, embedding_size, l2_reg, init_std, seed, prefix=''):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []

    # 构建所有使用嵌入的sparse特征的embedding层，dict形式{特证名:Embeding layer}
    embedding_dict = create_embedding_matrix(feature_columns, l2_reg, init_std, seed, embedding_size, prefix=prefix)
    sparse_embedding_list = embedding_lookup(embedding_dict, features, sparse_feature_columns)
    dense_value_list = get_dense_input(features, feature_columns)
    return sparse_embedding_list, dense_value_list


def combined_dnn_input(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = Flatten()(concat_fun(sparse_embedding_list))
        dense_dnn_input = Flatten()(concat_fun(dense_value_list))
        return concat_fun([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return Flatten()(concat_fun(sparse_embedding_list))
    elif len(dense_value_list) > 0:
        return Flatten()(concat_fun(dense_value_list))
    else:
        raise NotImplementedError


def concat_fun(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return tf.keras.layers.Concatenate(axis=axis)(inputs)

def get_feature_names(feature_columns):
    features = build_input_features(feature_columns)
    return list(features.keys())
