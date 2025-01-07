from typing import List, Literal, Union

import torch
import torch.nn as nn

from ..inputs import (DenseFeat, SparseFeat, VarLenSparseFeat,
                      create_embedding_matrix)
from ..layers import DNN
from .basemodel import BaseModel


class DNNModel(BaseModel):
    def __init__(
        self,
        linear_feature_columns: List[Union[SparseFeat, DenseFeat, VarLenSparseFeat]],
        dnn_feature_columns: List[Union[SparseFeat, DenseFeat, VarLenSparseFeat]],
        dnn_hidden_units=(128, 128),
        l2_reg_linear: float = 1e-5,
        l2_reg_embedding: float = 1e-5,
        l2_reg_dnn: float = 0,
        init_std: float = 0.0001,
        seed: int = 1024,
        dnn_dropout: float = 0,
        dnn_activation: Union[
            Literal["sigmoid", "relu", "prelu", "identity"], nn.Module
        ] = "relu",
        dnn_use_bn: bool = False,
        task: Literal["binary", "multiclass", "regression"] = "binary",
        device: Literal["cpu", "cuda", "mps"] = "cpu",
    ):
        """
        simple dnn model.
        :param linear_feature_columns: list of features attributes for linear model.
        :param dnn_feature_columns: list of features attributes for dnn model.
        :param dnn_hidden_units: dnn hidden unit's output and input size
        :param l2_reg_linear: L2 regularization for linear features
        :param l2_reg_embedding: L2 regularization for embedding features
        :param l2_reg_dnn: L2 regularization for dnn parameters
        :param init_std: initialize standard deviation
        :param seed: random seed value
        :param dnn_dropout: dnn's dropout rate
        :param dnn_activation: dnn's activation function
        :param dnn_use_bn: if dnn using bn, it's true else false
        :param task: object task
        :param device: target device
        """
        super(DNNModel, self).__init__(
            linear_feature_columns=linear_feature_columns,
            dnn_feature_columns=dnn_feature_columns,
            l2_reg_embedding=l2_reg_embedding,
            init_std=init_std,
            seed=seed,
            task=task,
            device=device,
        )

        self.dnn_hidden_units = dnn_hidden_units
        self.dnn = DNN(
            self._compute_input_dim(dnn_feature_columns),
            dnn_hidden_units,
            activation=dnn_activation,
            use_bn=dnn_use_bn,
            l2_reg=l2_reg_dnn,
            dropout_rate=dnn_dropout,
            init_std=init_std,
            device=device,
        )

        self.sparse_feature_columns = (
            list(filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns))
            if len(dnn_feature_columns)
            else []
        )
        self.dense_feature_columns = (
            list(filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns))
            if len(dnn_feature_columns)
            else []
        )

        self.embedding_dict = create_embedding_matrix(
            dnn_feature_columns, init_std, linear=True, sparse=False, device=device
        )

        dnn_linear_in_feature = dnn_hidden_units[-1]
        self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1, bias=False).to(device)
        self.add_regularization_weight(
            filter(
                lambda x: "weight" in x[0] and "bn" not in x[0],
                self.dnn.named_parameters(),
            ),
            l2=l2_reg_dnn,
        )
        self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_linear)
        self.to(device)
