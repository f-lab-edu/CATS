from typing import Callable, List, Literal, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import *

from ..callbacks import History
from ..inputs import (DenseFeat, SparseFeat, VarLenSparseFeat,
                      build_input_features, create_embedding_matrix,
                      embedding_lookup, get_dense_inputs)
from ..layers import PredictionLayer


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

        self.out = PredictionLayer(task)
        self.to(device)

        # parameters for callbacks
        self._is_graph_network = True  # used for ModelCheckpoint in tf2
        self._ckpt_saved_epoch = False  # used for EarlyStopping in tf1.14

        self.history = History()

    def compile(
        self,
        optimizer: Union[
            Literal["sgd", "adam", "adagrad", "rmsprop"], torch.optim.Optimizer
        ],
        loss: Union[
            List[Literal["binary_cross_entropy", "mse_loss", "mae"]],
            Literal["binary_cross_entropy", "mse_loss", "mae"],
            Callable,
        ],
        metrics: List[Literal["log_loss", "auc", "mse", "acc"]],
    ):
        """
        :param optimizer: the optimizer to use for training
        :param loss: the loss function to use for training
        :param metrics: a list of metrics to evaluate during training
        :return:
        """
        self.metrics_names = ["loss"]
        self.optim = self._get_optim(optimizer)
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics)

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

    def _get_optim(
        self,
        optimizer: Union[
            Literal["sgd", "adam", "adagrad", "rmsprop"], torch.optim.Optimizer
        ],
    ) -> torch.optim.Optimizer:
        """
        Get optimizer.
        :param optimizer: optimizer name or optimizer instance
        :return: optim: torch.optim.Optimizer instance
        """
        optim = None
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.parameters(), lr=0.01)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.parameters())
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.parameters())
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.parameters())
            else:
                raise NotImplementedError(f"{optimizer} is not implemented")
        elif isinstance(optimizer, torch.optim.Optimizer):
            optim = optimizer
        return optim

    def _get_loss_func_single(
        self, loss: Literal["binary_cross_entropy", "mse_loss", "mae"]
    ) -> Callable:
        """
        Get single loss function.
        :param loss: str, loss function name in ["binary_cross_entropy", "mse_loss", "mae"]
        :return: loss_func: Callable. loss function
        """
        if loss == "binary_cross_entropy":
            loss_func = F.binary_cross_entropy
        elif loss == "mse_loss":
            loss_func = F.mse_loss
        elif loss == "mae":
            loss_func = F.l1_loss
        else:
            raise NotImplementedError(f"{loss} is not implemented")
        return loss_func

    def _get_loss_func(
        self,
        loss: Union[
            List[Literal["binary_cross_entropy", "mse_loss", "mae"]],
            Literal["binary_cross_entropy", "mse_loss", "mae"],
            Callable,
        ],
    ) -> Union[List[Callable], Callable]:
        """
        Get loss function.
        :param loss: loss function's name or loss function's name list, loss function
        :return: loss_func: loss function or loss functions
        """
        if isinstance(loss, str):
            loss_func = self._get_loss_func_single(loss)
        elif isinstance(loss, list):
            loss_func = [self._get_loss_func_single(loss_name) for loss_name in loss]
        elif callable(loss):
            loss_func = loss
        else:
            raise ValueError(
                "Invalid type for loss. Expected a string, a list of strings, or a callable function."
            )
        return loss_func

    @staticmethod
    def _accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Return accuracy_score function
        :param y_true: numpy array of true target values
        :param y_pred: numpy array of predicted target values
        :return: float representing the accuracy score of the predictions
        """
        return accuracy_score(y_true, np.where(y_pred > 0.5, 1, 0))

    def _get_metrics(
        self, metrics: List[Literal["log_loss", "auc", "mse", "acc"]]
    ) -> dict:
        """
        Get logging metrics dictionary. {dict_name: Callable}
        :param metrics: logging metrics list
        :return: metrics_dict: dictionary for metrics
        """
        metrics_dict = {}
        if metrics:
            for metric in metrics:
                if metric == "log_loss":
                    metrics_dict[metric] = log_loss
                elif metric == "auc":
                    metrics_dict[metric] = roc_auc_score
                elif metric == "mse":
                    metrics_dict[metric] = mean_squared_error
                elif metric == "acc":
                    metrics_dict[metric] = self._accuracy_score
                else:
                    raise NotImplementedError(f"{metric} is not implemented")
                self.metrics_names.append(metric)
        return metrics_dict

    def input_from_feature_columns(
        self, inputs: torch.Tensor, feature_columns: List[Union[SparseFeat, DenseFeat]]
    ) -> Tuple[List, List]:
        """
        Get input data from feature columns.
        :param inputs: input tensor
        :param feature_columns: list about feature instances (SparseFeat, DenseFeat, VarLenSparseFeat)
        :return: sparse embedding value list and dense input value list
        """

        sparse_feature_columns = (
            list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))
            if len(feature_columns)
            else []
        )
        dense_feature_columns = (
            list(filter(lambda x: isinstance(x, DenseFeat), feature_columns))
            if len(feature_columns)
            else []
        )

        sparse_embedding_list = embedding_lookup(
            inputs, self.embedding_dict, self.feature_index, sparse_feature_columns
        )

        dense_value_list = get_dense_inputs(
            inputs, self.feature_index, dense_feature_columns
        )

        return sparse_embedding_list, dense_value_list
