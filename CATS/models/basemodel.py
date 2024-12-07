from typing import Literal

import torch
import torch.nn as nn

from ..inputs import build_input_features, create_embedding_matrix


class BaseModel(nn.Module):
    def __init__(
        self,
        linear_feature_columns: list,
        dnn_feature_columns: list,
        l2_reg_linear: float = 1e-5,
        l2_reg_embedding: float = 1e-5,
        init_std: float = 0.0001,
        seed: int = 1024,
        task: Literal["binary", "multiclass", "regression"] = "binary",
        device: Literal["cpu", "cuda", "mps"] = "cpu",
        gpus: list = None,
    ):
        """
        Base model for Machine Learning Models.
        :param linear_feature_columns: list of features attributes for linear model.
        :param dnn_feature_columns: list of features attributes for dnn model.
        :param l2_reg_linear: L2 regularization for linear features
        :param l2_reg_embedding: L2 regularization for embeddin features
        :param init_std: initialize standard deviation
        :param seed: random seed value
        :param task: object task
        :param device: target device
        :param gpus: list of gpus id
        """
        super(BaseModel, self).__init__()
        torch.manual_seed(seed)
        self.dnn_feature_columns = dnn_feature_columns

        self.reg_loss = torch.zeros((1,), device=device)
        self.aux_loss = torch.zeros((1,), device=device)
        self.device = device
        self.gpus = gpus
        if gpus and str(self.gpus[0]) not in self.device:
            raise ValueError(f"{gpus[0]} should be the same gpu with {device}")

        self.feature_index = build_input_features(
            linear_feature_columns + dnn_feature_columns
        )
        self.dnn_feature_columns = dnn_feature_columns

        self.embedding_dict = create_embedding_matrix(
            dnn_feature_columns, init_std, sparse=False, device=device
        )

        self.linear_model = nn.Linear(
            self.compute_input_dim(linear_feature_columns), 1, bias=False
        ).to(device)

        self.regularization_weight = []

        self.add_regularization_weight(
            self.embedding_dict.parameters(), l2=l2_reg_embedding
        )
        self.add_regularization_weight(self.linear_model.parameters(), l2=l2_reg_linear)

        self.to(device)

        # parameters for callbacks
        self._is_graph_network = True  # used for ModelCheckpoint in tf2
        self._ckpt_saved_epoch = False  # used for EarlyStopping in tf1.14
