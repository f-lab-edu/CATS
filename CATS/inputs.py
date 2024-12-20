import itertools
from collections import OrderedDict, defaultdict, namedtuple
from itertools import chain
from typing import DefaultDict, Dict, List, Literal, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

DEFAULT_GROUP_NAME = "default_group"


class SparseFeat(
    namedtuple(
        "SparseFeat",
        [
            "name",
            "vocabulary_size",
            "embedding_dim",
            "use_hash",
            "dtype",
            "embedding_name",
            "group_name",
        ],
    )
):
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

    def __new__(
        cls,
        name: str,
        vocabulary_size: int,
        embedding_dim=4,
        use_hash=False,
        dtype="int32",
        embedding_name=None,
        group_name=DEFAULT_GROUP_NAME,
    ):
        if embedding_name is None:
            embedding_name = name
        elif embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        if use_hash:
            raise NotImplementedError(
                "Feature hashing is not supported in PyTorch version. "
                "Please use TensorFlow or disable hashing."
            )
        return super(SparseFeat, cls).__new__(
            cls,
            name,
            vocabulary_size,
            embedding_dim,
            use_hash,
            dtype,
            embedding_name,
            group_name,
        )

    def __hash__(self):
        """
        Determines the hash value based on the name.
        :return: self.name's hash
        """
        return self.name.__hash__()


class VarLenSparseFeat(
    namedtuple("VarLenSparseFeat", ["sparsefeat", "maxlen", "combiner", "length_name"])
):
    __slots__ = ()

    def __new__(
        cls,
        sparsefeat: SparseFeat,
        maxlen: int,
        combiner: Literal["mean", "max", "sum"] = "mean",
        length_name=None,
    ):
        """
        :param sparsefeat: a single categorical data's info namedtuple
        :param maxlen: maximum categories length
        :param combiner: combining method for features ('sum', 'mean', 'max')
        :param length_name: feature length name
        """
        return super(VarLenSparseFeat, cls).__new__(
            cls, sparsefeat, maxlen, combiner, length_name
        )

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


class DenseFeat(namedtuple("Dense", ["name", "dimension", "dtype"])):
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


def get_feature_names(
    feature_columns: List[Union[SparseFeat, DenseFeat, VarLenSparseFeat]]
) -> list:
    """
    Get list of feature names
    :param feature_columns: list about feature instances (SparseFeat, DenseFeat, VarLenSparseFeat)
    :return: list about features dictionary's keys
    """
    if feature_columns is None:
        raise ValueError("feature_columns is None. feature_columns must be list")
    if not isinstance(feature_columns, list):
        raise ValueError(
            f"feature_columns is {type(feature_columns)}, feature_columns must be list."
        )
    if not all(
        isinstance(feature, (SparseFeat, DenseFeat, VarLenSparseFeat))
        for feature in feature_columns
    ):
        raise TypeError(
            "All elements in feature_columns must be instances of SparseFeat, DenseFeat or VarLenSparseFeat."
        )
    features = build_input_features(feature_columns)
    return list(features.keys())


def build_input_features(
    feature_columns: List[Union[SparseFeat, DenseFeat, VarLenSparseFeat]]
) -> dict:
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
            features[feat_name] = (
                curr_features_idx,
                curr_features_idx + feat.dimension,
            )
            curr_features_idx += feat.dimension
        elif isinstance(feat, VarLenSparseFeat):
            features[feat_name] = (curr_features_idx, curr_features_idx + feat.maxlen)
            curr_features_idx += feat.maxlen
            if feat.length_name is not None and feat.length_name not in features:
                features[feat.length_name] = (curr_features_idx, curr_features_idx + 1)
                curr_features_idx += 1
        else:
            raise TypeError(f"Invalid feature column type, got {type(feat)}")
    return features


def create_embedding_matrix(
    feature_columns: List[Union[SparseFeat, DenseFeat, VarLenSparseFeat]],
    init_std: float = 0.0001,
    linear: bool = False,
    sparse: bool = False,
    device: Literal["cuda", "gpu", "mps"] = "cpu",
) -> nn.ModuleDict:
    """
    Create embedding matrix. return embedding matrix {feature columns name: nn.Embedding}
    :param feature_columns: list about feature instances (SparseFeat, DenseFeat, VarLenSparseFeat)
    :param init_std: initial standard deviation
    :param linear: embedding dimension is 1
    :param sparse: If True, the gradient by the weight matrix will be a sparse tensor.
    :param device: cpu, cuda or mps
    :return: embedding dictionary. {feature columns name: nn.Embedding}
    """
    sparse_feature_columns = [x for x in feature_columns if isinstance(x, SparseFeat)]

    varlen_sparse_feature_columns = [
        x for x in feature_columns if isinstance(x, VarLenSparseFeat)
    ]

    embedding_dict = nn.ModuleDict(
        {
            feat.embedding_name: nn.Embedding(
                feat.vocabulary_size,
                feat.embedding_dim if not linear else 1,
                sparse=sparse,
            )
            for feat in sparse_feature_columns + varlen_sparse_feature_columns
        }
    )

    for tensor in embedding_dict.values():
        nn.init.normal_(tensor.weight, mean=0, std=init_std)

    return embedding_dict.to(device)


def embedding_lookup(
    inputs: torch.Tensor,
    sparse_embedding_dict: Dict[str, nn.Embedding],
    sparse_input_dict: OrderedDict[str:Tuple],
    sparse_feature_columns: List[SparseFeat],
    return_feature_list: list = (),
    to_list: bool = False,
) -> Union[DefaultDict[str, torch.Tensor], List[torch.Tensor]]:
    """
    Converts a sparse matrix to a dense matrix. Uses embedding when converting.
    :param inputs: input Tensor [batch_size x hidden_dim]
    :param sparse_embedding_dict: embedding matrix (nn.Embedding) of sparse embedding's name
    :param sparse_input_dict: sparse feature's indexes
    :param sparse_feature_columns: list about SparseFeat instances
    :param return_feature_list: names of feature to be returned, default () -> return all features
    :param to_list: true or false, convert list
    :return: group_embedding_dict: DefaultDict(list) or if to_list is true, list()
    """
    group_embedding = defaultdict(list)

    if sparse_feature_columns is None:
        raise ValueError(
            "sparse_feature_columns is None. sparse_feature_columns must be list"
        )
    if not isinstance(sparse_feature_columns, list):
        raise ValueError(
            f"sparse_feature_columns is {type(sparse_feature_columns)}, sparse_feature_columns must be list."
        )
    if not all(isinstance(feature, SparseFeat) for feature in sparse_feature_columns):
        raise TypeError(
            "All elements in sparse_feature_columns must be instances of SparseFeat."
        )

    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if len(return_feature_list) == 0 or feature_name in return_feature_list:
            lookup_idx = np.array(sparse_input_dict[feature_name])
            input_tensor = inputs[:, lookup_idx[0] : lookup_idx[1]].long()
            embedding_tensor = sparse_embedding_dict(embedding_name)[input_tensor]
            group_embedding[fc.group_name].append(embedding_tensor)
    if to_list:
        return list(chain.from_iterable(group_embedding.values()))
    return group_embedding


def varlen_embedding_lookup(
    inputs: torch.Tensor,
    varlen_sparse_embedding_dict: Dict[str, nn.Embedding],
    varlen_input_dict: OrderedDict[str:Tuple],
    varlen_sparse_feature_columns: List[VarLenSparseFeat],
) -> DefaultDict[str, torch.Tensor]:
    """
    Converts a variance length sparse matrix to a dense matrix. Uses embedding when converting
    :param inputs: input Tensor [batch_size x hidden_dim]
    :param varlen_sparse_embedding_dict: embedding matrix (nn.Embedding) of variance length sparse embedding's name
    :param varlen_input_dict: variance length sparse feature's indexes
    :param varlen_sparse_feature_columns: list about VarLenSparseFeat instances
    :return: group_embedding_dict: DefaultDict(list)
    """
    varlen_embedding_vec_dict = defaultdict(list)

    if varlen_sparse_feature_columns is None:
        raise ValueError(
            "varlen_sparse_feature_columns is None. varlen_sparse_feature_columns must be list"
        )
    if not isinstance(varlen_sparse_feature_columns, list):
        raise ValueError(
            f"varlen_sparse_feature_columns is {type(varlen_sparse_feature_columns)},"
            f" varlen_sparse_feature_columns must be list."
        )
    if not all(
        isinstance(feature, VarLenSparseFeat)
        for feature in varlen_sparse_feature_columns
    ):
        raise TypeError(
            "All elements in sparse_feature_columns must be instances of SparseFeat."
        )

    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.sparsefeat.embedding_name
        lookup_idx = varlen_input_dict[feature_name]
        input_tensor = inputs[:, lookup_idx[0] : lookup_idx[1]].long()
        varlen_embedding_vec_dict[fc.group_name] = varlen_sparse_embedding_dict[
            embedding_name
        ](input_tensor)
    return varlen_embedding_vec_dict


def get_dense_inputs(
    inputs: torch.Tensor,
    dense_input_dict: OrderedDict[str:Tuple],
    dense_feature_columns: List[DenseFeat],
) -> List[torch.Tensor]:
    """
    Return dense matrix in inputs.
    :param inputs: input Tensor [batch_size x hidden_dim]
    :param dense_input_dict: dense feature's indexes
    :param dense_feature_columns: list about DenseFeat instances
    :return: dense_input_list: list of dense features in inputs
    """
    dense_input_list = list()

    if dense_feature_columns is None:
        raise ValueError(
            "dense_feature_columns is None. dense_feature_columns must be list"
        )
    if not isinstance(dense_feature_columns, list):
        raise ValueError(
            f"dense_feature_columns is {type(dense_feature_columns)}, dense_feature_columns must be list."
        )
    if not all(isinstance(feature, DenseFeat) for feature in dense_feature_columns):
        raise TypeError(
            "All elements in dense_feature_columns must be instances of DenseFeat."
        )

    for fc in dense_feature_columns:
        feature_name = fc.name
        lookup_idx = np.array(dense_input_dict[feature_name])
        input_tensor = inputs[:, lookup_idx[0] : lookup_idx[1]].float()
        dense_input_list.append(input_tensor)
    return dense_input_list
