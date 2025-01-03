import torch
import torch.nn as nn

from .basemodel import BaseModel
from ..layers import DNN


class DNNModel(BaseModel):
    def __init__(self,
                 linear_feature_columns,
                 dnn_feature_columns,
                 dnn_hidden_units=(128, 128),
                 l2_reg_linear=0.00001,
                 l2_reg_embedding=0.00001,
                 l2_reg_dnn=0,
                 init_std=0.0001,
                 seed=1024,
                 dnn_dropout=0,
                 dnn_activation='relu',
                 dnn_use_bn=False,
                 task='binary',
                 device='cpu'):
        super(DNNModel, self).__init__(linear_feature_columns=linear_feature_columns,
                                       dnn_feature_columns=dnn_feature_columns,
                                       l2_reg_embedding=l2_reg_embedding,
                                       init_std=init_std,
                                       seed=seed,
                                       task=task,
                                       device=device)

        self.dnn_hidden_units = dnn_hidden_units
        self.dnn = DNN(self._compute_input_dim(dnn_feature_columns),
                       dnn_hidden_units,
                       activation=dnn_activation,
                       use_bn=dnn_use_bn,
                       l2_reg=l2_reg_dnn,
                       dropout_rate=dnn_dropout,
                       init_std=init_std,
                       device=device)
        dnn_linear_in_feature = dnn_hidden_units[-1]
        self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1, bias=False).to(device)
        self.to(device)


