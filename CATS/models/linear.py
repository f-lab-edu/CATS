from collections import OrderedDict
from typing import List, Literal, Tuple, Union

import torch
import torch.nn as nn

from ..inputs import (DenseFeat, SparseFeat, VarLenSparseFeat,
                      create_embedding_matrix)


class Linear(nn.Module):
    def __init__(
        self,
        feature_columns: List[Union[SparseFeat, DenseFeat, VarLenSparseFeat]],
        feature_index: OrderedDict[str:Tuple],
        init_std: float = 0.0001,
        device: Literal["cpu", "cuda", "mps"] = "cpu",
    ):
        """
        Linear regression model
        :param feature_columns: list about feature instances (SparseFeat, DenseFeat, VarLenSparseFeat)
        :param feature_index: Start and end index information for each feature instance
        :param init_std: initialize standard deviation
        :param device: target device
        """
        super(Linear, self).__init__()

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

        self.feature_index = feature_index
        self.device = device

        self.sparse_feature_columns = (
            list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))
            if len(feature_columns)
            else []
        )
        self.varlen_sparse_feature_columns = (
            list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns))
            if len(feature_columns)
            else []
        )
        self.dense_feature_columns = (
            list(filter(lambda x: isinstance(x, DenseFeat), feature_columns))
            if len(feature_columns)
            else []
        )

        self.embedding_dict = create_embedding_matrix(
            feature_columns, init_std, linear=True, sparse=False, device=device
        )

        for tensor in self.embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)

        if len(self.dense_feature_columns) > 0:
            self.weight = nn.Parameter(
                torch.Tensor(
                    sum(fc.dimension for fc in self.dense_feature_columns), 1
                ).to(device)
            )
            torch.nn.init.normal_(self.weight, mean=0, std=init_std)
