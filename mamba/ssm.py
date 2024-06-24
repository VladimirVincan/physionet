import numpy as np
import torch
import torch.nn as nn
from mamba_ssm import Mamba

class StateSpaceModel(nn.Module):
    def __init__(self):
        super().__init__()

        feature_mamba_layers = "[(1024)] * 6"  # TODO
        feature_mamba_layers = eval(feature_mamba_layers)
        self.mamba_encoder = MambaEncoder(feature_mamba_layers)

        self.decoder = nn.Linear(in_features=1024, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, source):
        source = source.squeeze(1)
        features = self.mamba_encoder(source)
        # features = self.layer_norm(features)
        features = self.decoder(features)
        features = self.sigmoid(features)

        return features


class MambaEncoder(nn.Module):
    def __init__(
        self,
        mamba_layers: List[Tuple[int]]
    ):
        super().__init__()

        self.mamba_layers = nn.ModuleList()
        for i, ml in enumerate(mamba_layers):
            # assert len(ml) == 1, "invalid mamba definition: " + str(ml)
            (dim) = ml

            self.mamba_layers.extend(
                [
                    Mamba(
                        d_model=dim,
                        d_state=16,
                        d_conv=4,
                        expand=2,
                    ),
                    LayerNorm(dim)
                ]
            )  # .to('cuda')  # TODO: .to('cuda')

    def forward(self, x):

        for mamba in self.mamba_layers:
            x = mamba(x)

        return x
