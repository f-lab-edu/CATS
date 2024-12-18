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

    def _sequence_mask(
        self,
        lengths: torch.Tensor,
        max_len: int = None,
        dtype: torch.dtype = torch.bool,
    ) -> torch.Tensor:
        """
        Generate sequence mask. sequence mask's length is max_len.
        :param lengths: Tensor containing lengths of features
        :param max_len: maximum length of the sequence. if max_len is None, maximum in lengths
        :param dtype: mask's data type
        :return: mask: sequence mask.
        """
        if max_len is None:
            max_len = lengths.max()
        row_vector = torch.arange(0, max_len, 1).to(lengths.device)
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = row_vector < matrix
        mask = mask.type(dtype)
        return mask

    def forward(self, seq_value_len_list: list) -> torch.Tensor:
        """
        feed forward SequencePoolingLayer. return pooling results tensor sequence.
        :param seq_value_len_list: list in sequence embedding and mask or user_behavior_length
        :return: hist: pooling result sequence
        """
        if self.supports_masking:
            seq_embed_list, mask = seq_value_len_list
            if len(seq_embed_list.shape) != 3:
                raise ValueError(
                    "Expected seq_embed_list tensor to have 3 dimensions, but got " + str(len(seq_embed_list.shape)))
            mask = mask.float()
            user_behavior_length = torch.sum(mask, dim=-1, keepdim=True)
            mask = mask.unsqueeze(2)
        else:
            seq_embed_list, user_behavior_length = seq_value_len_list
            if len(seq_embed_list.shape) != 3:
                raise ValueError(
                    "Expected seq_embed_list tensor to have 3 dimensions, but got " + str(len(seq_embed_list.shape)))
            mask = self._sequence_mask(
                user_behavior_length,
                max_len=seq_embed_list.shape[1],
                dtype=torch.float32,
            )
            mask = torch.transpose(mask, 1, 2)

        embedding_size = seq_embed_list.shape[-1]
        mask = torch.repeat_interleave(mask, embedding_size, dim=2)
        if self.mode == "max":
            hist = seq_embed_list - (1 - mask) * 1e9
            hist = torch.max(hist, dim=1, keepdim=True)[0]
            return hist
        hist = seq_embed_list * mask.float()
        hist = torch.sum(hist, dim=1, keepdim=False)

        if self.mode == "mean":
            self.eps = self.eps.to(user_behavior_length.device)
            hist = torch.div(hist, user_behavior_length.type(torch.float32) + self.eps)
        hist = torch.unsqueeze(hist, dim=1)
        return hist
