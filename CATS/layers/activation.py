import torch
import torch.nn as nn
from typing import Union, Literal


class Identity(nn.Module):

    def __init__(self, **kwargs):
        super(Identity, self).__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        return inputs
        :param inputs: inputs tensor
        :return: inputs
        """
        return inputs


def activation_layer(act_name: Union[Literal['sigmoid', 'relu', 'prelu', 'identity'], nn.Module]) -> nn.Module:
    """
    Get activation layers
    :param act_name: activation function name
    :return: activation layer
    """
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif act_name.lower() == 'prelu':
            act_layer = nn.PReLU()
        elif act_name.lower() == 'identity':
            act_layer = nn.Identity()
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError
    return act_layer
