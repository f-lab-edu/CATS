from typing import Literal, Union

import torch
import torch.nn as nn
from .activation import activation_layer


class DNN(nn.Module):

    def __init__(
        self,
        inputs_dim: int,
        hidden_units: list,
        activation: Union[
            Literal["sigmoid", "relu", "prelu", "identity"], nn.Module
        ] = "relu",
        l2_reg: float = 0,
        dropout_rate: int = 0,
        use_bn: bool = False,
        init_std: float = 0.0001,
        seed: int = 1024,
        device: Literal["cpu", "cuda", "mps"] = "cpu",
    ):
        """
        The multi perceptron layer.
        :param inputs_dim: input feature dimension.
        :param hidden_units: the layer number and units in each layer.
        :param activation: activation function to use.
        :param l2_reg: float between 0 and 1. L2 regularize strength applied to the kernel weights matrix.
        :param dropout_rate: float in [0, 1). Fraction of the units to dropout.
        :param use_bn: Use BatchNormalization before activation or not.
        :param init_std: Init Standard deviation
        :param seed: random seed number
        :param device: 'cpu' or 'gpu'
        """
        super(DNN, self).__init__()

        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn

        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList(
            [
                nn.Linear(hidden_units[i], hidden_units[i + 1])
                for i in range(len(hidden_units) - 1)
            ]
        )

        if self.use_bn:
            self.bn = nn.ModuleList(
                [
                    nn.BatchNorm1d(hidden_units[i + 1])
                    for i in range(len(hidden_units) - 1)
                ]
            )

        self.activation_layers = nn.ModuleList(
            [activation_layer(activation) for i in range(len(hidden_units) - 1)]
        )

        for name, tensor in self.linears.named_parameters():
            if "weight" in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        self.to(device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass
        :param inputs: input tensors
        :return: output tensors
        """
        deep_input = inputs

        for i in range(len(self.linears)):

            fc = self.linears[i](deep_input)

            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)

            fc = self.dropout(fc)
            deep_input = fc

        return deep_input


class PredictionLayer(nn.Module):
    def __init__(
        self,
        task: Literal["binary", "multiclass", "regression"] = "binary",
        use_bias: bool = True,
    ):
        """
        Model output layer
        :param task: model's task in ["binary", "multiclass", "regression"]
        :param use_bias: using bias
        """
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary, multiclass or regression")

        super(PredictionLayer, self).__init__()
        self.use_bias = use_bias
        self.task = task
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros((1,)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param x: input tensors. input tensor's shape is [batch size, classes number].
        :return: output tensors
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected input to be of type torch.Tensor, but got {type(x)}.")
        if not torch.isfinite(x).all():
            raise ValueError("Input tensor contains NaN or Inf values, which are not allowed.")

        inputs = x
        if self.use_bias:
            inputs += self.bias
        if self.task == "binary":
            outputs = torch.sigmoid(inputs)
        elif self.task == "multiclass":
            outputs = torch.softmax(inputs, dim=-1)
        else:
            outputs = inputs
        return outputs


if __name__ == "__main__":
    """Module for Execution in Testing
    python -m CATS.layers.core
    """
    inputs_dim = 10
    hidden_units = [32, 16, 3]
    l2_reg = 0.01
    use_bn = True
    seed = 42
    # device = "mps"  # in mac..
    device = "cuda"

    model = DNN(
        inputs_dim=inputs_dim,
        hidden_units=hidden_units,
        l2_reg=l2_reg,
        use_bn=use_bn,
        seed=seed,
        device=device,
    )

    input_tensor = torch.randn(4, inputs_dim).to(device)
    output = model(input_tensor)
    prediction_layer = PredictionLayer("multiclass")
    output = prediction_layer(output)
    print("Output Tensor Shape:", output.shape)  # 출력 텐서의 형태 확인
    print("Output Tensor:", output)  # 출력 텐서 내용 확인
