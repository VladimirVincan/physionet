import numpy as np
import torch
import torch.nn as nn
from fairseq.modules import LayerNorm
from mamba_ssm import Mamba

from BiMamba import BiMambaEncoder
from ConvFeatureExtractionModel import ConvFeatureExtractionModel


class StateSpaceModel(nn.Module):
    def __init__(self, feature_enc_layers, feature_mamba_layers):
        super().__init__()


        # feature_enc_layers = "[(1024, 4, 1)] + [(1024, 3, 1)] + [(1024,2,1)]"
        # feature_enc_layers = "[(32, 1, 1)]"
        feature_enc_layers = eval(feature_enc_layers)
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            conv_bias=True,
        )

        # TODO: add layer norm and post_extract_proj

        # feature_mamba_layers = "[(32)]"  # TODO
        feature_mamba_layers = eval(feature_mamba_layers)
        self.mamba_encoder = MambaEncoder(feature_mamba_layers)

        self.decoder = nn.Linear(in_features=feature_mamba_layers[-1], out_features=1)
        # Do not put Sigmoid non-linearity, due to BCEWithLogitsLoss
        self.sigmoid = nn.Sigmoid()

    def forward(self, source):
        source = source.squeeze(1)
        features = self.feature_extractor(source)
        features = features.transpose(1, 2)
        features = self.mamba_encoder(features)
        # features = self.layer_norm(features)
        features = self.decoder(features)
        if self.eval:
            features = self.sigmoid(features)

        return features


class MambaEncoder(nn.Module):
    def __init__(
        self,
        mamba_layers: list[tuple[int]]
    ):
        super().__init__()

        self.mamba_layers = nn.ModuleList()
        for i, ml in enumerate(mamba_layers):
            # assert len(ml) == 1, "invalid mamba definition: " + str(ml)
            (dim) = ml

            self.mamba_layers.extend(
                [
                    # Mamba(
                    #     d_model=dim,
                    #     d_state=8,
                    #     d_conv=4,
                    #     expand=2,
                    # ),
                    BiMambaEncoder(
                        d_model=dim,
                        n_state=8,
                    ),
                    LayerNorm(dim)
                ]
            )  # .to('cuda')  # TODO: .to('cuda')


    def forward(self, x):

        for mamba in self.mamba_layers:
            x = mamba(x)

        return x
