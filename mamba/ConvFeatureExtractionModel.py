import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: list[tuple[int, int, int]],
        dropout: float = 0.0,
        conv_bias: bool = False,
        in_dim: int = 13,
        transpose: bool = False
    ):

        super().__init__()
        self.transpose = transpose

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            return nn.Sequential(
                make_conv(),
                nn.Dropout(p=dropout),
                nn.Sequential(
                    TransposeLast(),
                    Fp32LayerNorm(dim, elementwise_affine=True),
                    TransposeLast(),
                ),
                nn.GELU(),
            )

        in_d = in_dim
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):

        # BxT -> BxCxT
        x = x.transpose(-1, -2)
        # x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)

        if self.transpose:
            x = x.transpose(-1, -2)
        return x


class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None, tranpose_dim=-2):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx
        self.tranpose_dim = tranpose_dim

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(self.tranpose_dim, -1)
