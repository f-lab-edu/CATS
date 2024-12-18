from typing import List, Literal, Union

import torch
import torch.nn as nn

from ..inputs import (
    DenseFeat,
    SparseFeat,
    VarLenSparseFeat,
    build_input_features,
    create_embedding_matrix,
)

from ..layers import PredictionLayer
from ..callbacks import History


class BaseModel(nn.Module):
    def __init__(
        self,
        linear_feature_columns: List[Union[SparseFeat, DenseFeat, VarLenSparseFeat]],
        dnn_feature_columns: List[Union[SparseFeat, DenseFeat, VarLenSparseFeat]],
        l2_reg_linear: float = 1e-5,
        l2_reg_embedding: float = 1e-5,
        init_std: float = 0.0001,
        seed: int = 1024,
        task: Literal["binary", "multiclass", "regression"] = "binary",
        device: Literal["cpu", "cuda", "mps"] = "cpu",
    ):
        """
        Base model for Machine Learning Models.
        :param linear_feature_columns: list of features attributes for linear model.
        :param dnn_feature_columns: list of features attributes for dnn model.
        :param l2_reg_linear: L2 regularization for linear features
        :param l2_reg_embedding: L2 regularization for embedding features
        :param init_std: initialize standard deviation
        :param seed: random seed value
        :param task: object task
        :param device: target device
        """
        super(BaseModel, self).__init__()
        torch.manual_seed(seed)
        self.dnn_feature_columns = dnn_feature_columns

        self.reg_loss = torch.zeros((1,), device=device)
        self.aux_loss = torch.zeros((1,), device=device)
        self.device = device

        self.feature_index = build_input_features(
            linear_feature_columns + dnn_feature_columns
        )

        self.embedding_dict = create_embedding_matrix(
            dnn_feature_columns, init_std, sparse=False, device=device
        )

        self.linear_model = nn.Linear(
            self._compute_input_dim(linear_feature_columns), 1, bias=False
        ).to(device)

        self.regularization_weight = []

        self.add_regularization_weight(
            self.embedding_dict.parameters(), l2=l2_reg_embedding
        )
        self.add_regularization_weight(self.linear_model.parameters(), l2=l2_reg_linear)

        self.out = PredictionLayer(
            task,
        )
        self.to(device)

        # parameters for callbacks
        self._is_graph_network = True  # used for ModelCheckpoint in tf2
        self._ckpt_saved_epoch = False  # used for EarlyStopping in tf1.14

        self.history = History()

    def _compute_input_dim(
        self,
        feature_columns: List[Union[SparseFeat, DenseFeat, VarLenSparseFeat]],
        include_sparse: bool = True,
        include_dense: bool = True,
        feature_group: bool = False,
    ) -> int:
        """
        Compute length of input dimensions.
        :param feature_columns: list about feature instances (SparseFeat, DenseFeat, VarLenSparseFeat)
        :param include_sparse: true or false, include sparse feature
        :param include_dense: true or false, include dense feature
        :param feature_group:if True, counts sparse features as individual groups (ignoring embedding dimensions);
                             if False, sums up embedding dimensions of sparse features
        :return: number of total input dimensions
        """
        input_dim = 0

        sparse_feature_columns = list(
            filter(
                lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns
            )
            if len(feature_columns)
            else []
        )

        dense_feature_columns = (
            list(filter(lambda x: isinstance(x, DenseFeat), feature_columns))
            if len(feature_columns)
            else []
        )

        dense_input_dim = sum(map(lambda x: x.dimension, dense_feature_columns))
        if feature_group:
            sparse_input_dim = len(sparse_feature_columns)
        else:
            sparse_input_dim = sum(
                feat.embedding_dim for feat in sparse_feature_columns
            )

        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim
        return input_dim
