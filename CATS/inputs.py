from collections import namedtuple
from typing import Literal

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
        if dimension < 0 and not isinstance(dimension, int):
            raise ValueError("dimension must bigger then 0 and must be integer ")
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)

    def __hash__(self):
        return self.name.__hash__()
