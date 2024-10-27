import torch
import torch.nn as nn
from typing import Union


def activation_layer(act_name: Union[str, nn.Module]) -> nn.Module:
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
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError
    return act_layer
