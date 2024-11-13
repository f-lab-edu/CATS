from collections import OrderedDict, namedtuple
from typing import List, Literal, Union

DEFAULT_GROUP_NAME = "default_group"


class SparseFeat(namedtuple('SparseFeat',
                            ['name', 'vocabulary_size', 'embedding_dim', 'use_hash', 'dtype', 'embedding_name',
                             'group_name'])):
    """
     Returns information about a single categorical data.
     :param name: feature's name
     :param vocabulary_size: input category name
     :param embedding_dim: Converted embedding's dimension
     :param use_hash: whether to use hash
     :param dtype: data's type
     :param embedding_name: embedding's name
     :param group_name: group's name
     """
    __slots__ = ()

    def __new__(cls, name: str, vocabulary_size: int, embedding_dim=4, use_hash=False, dtype="int32", embedding_name=None,
                group_name=DEFAULT_GROUP_NAME):
        if embedding_name is None:
            embedding_name = name
        elif embedding_dim == 'auto':
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        if use_hash:
            raise NotImplementedError("Feature hashing is not supported in PyTorch version. "
                                      "Please use TensorFlow or disable hashing.")
        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, dtype,
                                              embedding_name, group_name)

    def __hash__(self):
        """
         Determines the hash value based on the name.
         :return: self.name's hash
         """
        return self.name.__hash__()


class VarLenSparseFeat(namedtuple('VarLenSparseFeat',
                                  ['sparsefeat', 'maxlen', 'combiner', 'length_name'])):
    __slots__ = ()

    def __new__(cls, sparsefeat: SparseFeat, maxlen: int, combiner: Literal['mean', 'max', 'sum'] = 'mean',
                length_name=None):
        """
         :param sparsefeat: a single categorical data's info namedtuple
         :param maxlen: maximum categories length
         :param combiner: combining method for features ('sum', 'mean', 'max')
         :param length_name: feature length name
         """
        return super(VarLenSparseFeat, cls).__new__(cls, sparsefeat, maxlen, combiner, length_name)

    @property
    def name(self):
        """
         VarLenSparseFeat's name
         :return: sparsefeat.name
         """
        return self.sparsefeat.name

    @property
    def vocabulary_size(self):
        """
        VarLenSparseFeat's vocabulary size
        :return: sparsefeat.vocabulary_size
        """
        return self.sparsefeat.vocabulary_size

    @property
    def embedding_dim(self):
        """
         VarLenSparseFeat's embedding dimension
         :return: sparsefeat.embedding_dim
         """
        return self.sparsefeat.embedding_dim

    @property
    def use_hash(self):
        """
           whether to use hash
          :return: sparsefeat.use_hash
          """
        return self.sparsefeat.use_hash

    @property
    def dtype(self):
        """
        data's type
        :return: sparsefeat.dtype
        """
        return self.sparsefeat.dtype

    @property
    def embedding_name(self):
        """
        embedding's name
        :return: sparsefeat.embedding_name
        """
        return self.sparsefeat.embedding_name

    @property
    def group_name(self):
        """
        group's name
        :return: sparsefeat.group_name
        """
        return self.sparsefeat.group_name

    def __hash__(self):
        """
         Determines the hash value based on the name.
         :return: self.name's hash
         """
        return self.name.__hash__()


class DenseFeat(namedtuple('Dense',
                           ['name', 'dimension', 'dtype'])):
    __slots__ = ()

    def __new__(cls, name: str, dimension=1, dtype="float32"):
        """
          Returns information about a numeric data.
          :param name: numeric data's attribute name
          :param dimension: dimension number
          :param dtype: data's type
        """
        if dimension < 0 and not isinstance(dimension, int):
            raise ValueError("dimension must bigger then 0 and must be integer ")
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)

    def __hash__(self):
        """
          Determines the hash value based on the name.
          :return: self.name's hash
        """
        return self.name.__hash__()


def get_feature_names(feature_columns: List[Union[SparseFeat, DenseFeat, VarLenSparseFeat]]) -> list:
    """
    Get list of feature names
    :param feature_columns: list about feature instances (SparseFeat, DenseFeat, VarLenSparseFeat)
    :return: list about features dictionary's keys
    """
    if feature_columns is None:
        raise ValueError("feature_columns is None. feature_columns must be list")
    if not isinstance(feature_columns, list):
        raise ValueError(f"feature_columns is {type(feature_columns)}, feature_columns must be list.")
    if not all(isinstance(feature, (SparseFeat, DenseFeat, VarLenSparseFeat)) for feature in feature_columns):
        raise TypeError(
            "All elements in feature_columns must be instances of SparseFeat, DenseFeat or VarLenSparseFeat.")
    features = build_input_features(feature_columns)
    return list(features.keys())


def build_input_features(feature_columns: List[Union[SparseFeat, DenseFeat, VarLenSparseFeat]]) -> dict:
    """
    Return an input feature dictionary based on various types of features (SparseFeat, DenseFeat, VarLenSparseFeat).
    input feature dictionary stores the start and end inices of each feature, helping the model identify the location of
    each feature in the input data.
    :param feature_columns: list about feature instances (SparseFeat, DenseFeat, VarLenSparseFeat)
    :return: dictionary about features
    """
    features = OrderedDict()

    curr_features_idx = 0
    for feat in feature_columns:
        feat_name = feat.name
        if feat_name in features:
            continue
        if isinstance(feat, SparseFeat):
            features[feat_name] = (curr_features_idx, curr_features_idx + 1)
            curr_features_idx += 1
        elif isinstance(feat, DenseFeat):
            features[feat_name] = (curr_features_idx, curr_features_idx + feat.dimension)
            curr_features_idx += feat.dimension
        elif isinstance(feat, VarLenSparseFeat):
            features[feat_name] = (curr_features_idx, curr_features_idx + feat.maxlen)
            curr_features_idx += feat.maxlen
            if feat.length_name is not None and feat.length_name not in features:
                features[feat.length_name] = (curr_features_idx, curr_features_idx+1)
                curr_features_idx += 1
        else:
            raise TypeError(f"Invalid feature column type, got {type(feat)}")
    return features

