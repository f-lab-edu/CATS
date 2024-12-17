from typing import Literal

import torch
import torch.nn as nn


class SequencePoolingLayer(nn.Module):
    def __init__(
        self,
        mode: Literal["mean", "sum", "max"] = "mean",
        supports_masking: bool = False,
        device: Literal["cuda", "gpu", "mps"] = "cpu",
    ):
        """
        apply pooling operation (sum, mean, max) on variable-length sequence feature/multi-value feature.
        :param mode: Pooling operation to be used,can be sum,mean or max.
        :param supports_masking: check to support masking
        :param device: devices. cpu, cuda or mps
        """
        super(SequencePoolingLayer, self).__init__()
        if mode not in ["sum", "mean", "max"]:
            raise ValueError(
                f'parameter mode should in ["sum", "mean", "max"], {mode} is not!!!'
            )

        self.supports_masking = supports_masking
        self.mode = mode
        self.device = device
        self.eps = torch.FloatTensor([1e-8]).to(device)
        self.to(device)
