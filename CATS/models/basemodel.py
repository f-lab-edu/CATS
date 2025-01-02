import logging
import time
from typing import Dict, List, Literal, Union

import numpy as np
import torch
import torch.nn as nn
from tensorflow.keras.callbacks import Callback
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..callbacks import History
from ..inputs import (DenseFeat, SparseFeat, VarLenSparseFeat,
                      build_input_features, create_embedding_matrix)
from ..layers import PredictionLayer

try:
    from tensorflow.python.keras.callbacks import CallbackList
except ImportError:
    from tensorflow.python.keras._impl.keras.callbacks import CallbackList


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

    def fit(
        self,
        x: Union[List[np.ndarray], Dict[str, np.ndarray]],
        y: Union[np.ndarray, List[np.ndarray]],
        batch_size: int = 256,
        epochs: int = 1,
        verbose: int = 1,
        initial_epoch: int = 0,
        validation_split: float = 0.0,
        shuffle: bool = True,
        callbacks: List[Callback] = None,
    ) -> History:
        """
        Training Model. Return history about training.
        :param x: numpy array of training data (if the model has a single input), or list of numpy arrays (if the model
            has multiple inputs). If input layers in the model are named, you can also pass a
            dictionary mapping input names to numpy arrays.
        :param y: numpy array of target (label) data or list of numpy arrays
        :param batch_size: Integer. Number of sample per gradient update.
        :param epochs: Integer. Number of epochs to train the model.
        :param verbose: Integer. 0, 1, or 2. verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        :param initial_epoch: Integer. Epoch at which to start training.
        :param validation_split: Float between 0 and 1. rate of validation datasets.
        :param shuffle: Bool. whether to shuffle the order of the batches at the beginning of each epoch.
        :param callbacks: List of `deepctr_torch.callbacks.Callback` instances. [`EarlyStopping` , `ModelCheckpoint`]
        :return: A `Histroy` object.  Its `History.history` attribute is a record of training loss values and metrics
            values at successive epochs, as well as validation loss values and validation metrics values (if applicable)
        """
        # setting train & validation data
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]

        do_validation = False
        if validation_split and 0. < validation_split <= 1.0:
            do_validation = True
            if do_validation:
                if hasattr(x[0], "shape"):
                    split_at = int(x[0].shape[0] * (1. - validation_split))
                else:
                    split_at = int(len(x[0]) * (1. - validation_split))
                x, val_x = x[:split_at], x[split_at:]
                y, val_y = y[:split_at], y[split_at:]
        else:
            val_x = []
            val_y = []

        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        train_tensor_data = TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1)), torch.from_numpy(y)
        )

        model = self.train()
        loss_func = self.loss_func
        optim = self.optim

        # setting dataloader
        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size
        )
        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        # configure callbacks
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, "model"):
            callbacks.__setattr__("model", self)
        callbacks.model.stop_training = False

        # Training
        logging.info(
            "Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
                len(train_tensor_data), len(val_y), steps_per_epoch
            )
        )

        for epoch in range(initial_epoch, epochs):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}
            try:
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    for _, (x_train, y_train) in t:
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()

                        y_pred = model(x).squeeze()

                        optim.zero_grad()
                        if isinstance(loss_func, list):
                            assert (
                                len(loss_func) == self.num_tasks
                            ), "the length of `loss_func` should be equal with `self.num_tasks`"
                            loss = sum(
                                [
                                    loss_func[i](y_pred[:, i], y[:, i], reduction="sum")
                                    for i in range(self.num_tasks)
                                ]
                            )
                        else:
                            loss = loss_func(y_pred, y.squeeze(), reduction="sum")
                        reg_loss = self.get_regularization_loss()

                        total_loss = loss + reg_loss + self.aux_loss

                        loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()
                        total_loss.backward()
                        optim.step()

                        if verbose > 0:
                            for name, metric_fun in self.metrics.items():
                                if name not in train_result:
                                    train_result[name] = []
                                train_result[name].append(
                                    metric_fun(
                                        y.cpu().data.numpy(),
                                        y_pred.cpu().data.numpy().astype("float64"),
                                    )
                                )
            except KeyboardInterrupt:
                t.close()
                raise
            t.close()

            epoch_logs["loss"] = total_loss_epoch / sample_num
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                logging.info("Epoch {0}/{1}".format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"]
                )

                for name in self.metrics:
                    eval_str += " - " + name + ": {0: .4f}".format(epoch_logs[name])

                if do_validation:
                    for name in self.metrics:
                        eval_str += (
                            " - "
                            + "val_"
                            + name
                            + ": {0: .4f}".format(epoch_logs["val_" + name])
                        )
                logging.info(eval_str)
                callbacks.on_epoch_end(epoch, epoch_logs)
                if self.stop_training:
                    break

            callbacks.on_train_end()

            return self.history

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
