import torch
from torch import nn

"""
Copied from here: https://medium.com/@alessandromondin/semantic-segmentation-with-pytorch-u-net-from-scratch-502d6565910a
"""


class CNNBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding='same'):
        super(CNNBlock, self).__init__()

        self.seq_block = nn.Sequential(
            nn.Conv1d(in_channels,
                      out_channels,
                      kernel_size,
                      stride,
                      padding,
                      bias=False), nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.seq_block(x)
        return x


class CNNBlocks(nn.Module):
    """
    Parameters:
    n_conv (int): creates a block of n_conv convolutions
    in_channels (int): number of in_channels of the first block's convolution
    out_channels (int): number of out_channels of the first block's convolution
    expand (bool) : if True after the first convolution of a block the number of channels doubles
    """

    def __init__(self, n_conv, in_channels, out_channels, padding):
        super(CNNBlocks, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(n_conv):

            self.layers.append(
                CNNBlock(in_channels, out_channels, padding=padding))
            # after each convolution we set (next) in_channel to (previous) out_channels
            in_channels = out_channels

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Encoder(nn.Module):
    """
    Parameters:
    in_channels (int): number of in_channels of the first CNNBlocks
    out_channels (int): number of out_channels of the first CNNBlocks
    padding (int): padding applied in each convolution
    maxpool_kernel_sizes (list of ints): the sizes of sliding windows
    downhill (int): number times a CNNBlocks + MaxPool2D it's applied.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 maxpool_kernel_sizes,
                 padding='same',
                 downhill=4):
        super(Encoder, self).__init__()
        self.enc_layers = nn.ModuleList()

        for i in range(downhill):
            self.enc_layers += [
                CNNBlocks(n_conv=2,
                          in_channels=in_channels,
                          out_channels=out_channels,
                          padding=padding),
                nn.MaxPool1d(maxpool_kernel_sizes[i], ceil_mode=True)
            ]

            in_channels = out_channels
            out_channels *= 2
        # doubling the dept of the last CNN block
        self.enc_layers.append(
            CNNBlocks(n_conv=2,
                      in_channels=in_channels,
                      out_channels=out_channels,
                      padding=padding))

    def forward(self, x):
        route_connection = []
        for layer in self.enc_layers:
            if isinstance(layer, CNNBlocks):
                x = layer(x)
                route_connection.append(x)
            else:
                x = layer(x)
        return x, route_connection


class Decoder(nn.Module):
    """
    Parameters:
    in_channels (int): number of in_channels of the first ConvTranspose2d
    out_channels (int): number of out_channels of the first ConvTranspose2d
    padding (int): padding applied in each convolution
    maxpool_kernel_sizes (list of ints): the sizes of sliding windows
    uphill (int): number times a ConvTranspose2d + CNNBlocks it's applied.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 exit_channels,
                 maxpool_kernel_sizes,
                 padding='same',
                 uphill=4):
        super(Decoder, self).__init__()
        self.exit_channels = exit_channels
        self.layers = nn.ModuleList()

        for i in range(uphill):

            self.layers += [
                nn.ConvTranspose1d(in_channels,
                                   out_channels,
                                   kernel_size=maxpool_kernel_sizes[uphill - i - 1],
                                   stride=maxpool_kernel_sizes[uphill - i - 1]),
                CNNBlocks(n_conv=2,
                          in_channels=in_channels,
                          out_channels=out_channels,
                          padding=padding),
            ]
            in_channels //= 2
            out_channels //= 2

        # cannot be a CNNBlock because it has ReLU incorpored
        # cannot append nn.Sigmoid here because you should be later using
        # BCELoss () which will trigger the amp error "are unsafe to autocast".
        self.layers.append(
            nn.Conv1d(in_channels,
                      exit_channels,
                      kernel_size=1,
                      padding=padding), )

    def forward(self, x, routes_connection):
        # pop the last element of the list since
        # it's not used for concatenation
        routes_connection.pop(-1)
        for layer in self.layers:
            if isinstance(layer, CNNBlocks):
                # concatenating tensors channel-wise
                curr_connection = routes_connection.pop(-1)
                curr_connection = curr_connection[:, :, :x.shape[2]]
                x = torch.cat([x, curr_connection], dim=1)
                x = layer(x)
            else:
                x = layer(x)
        return x


class FCNN(nn.Module):
    def __init__(self,
                 in_channels=32,
                 depth=3):
        super(FCNN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers += nn.Linear()



class UNet(nn.Module):

    def __init__(self,
                 in_channels,
                 first_out_channels,
                 exit_channels,
                 maxpool_kernel_sizes,
                 downhill=3,
                 padding='same'):
        super(UNet, self).__init__()
        self.encoder = Encoder(in_channels,
                               first_out_channels,
                               padding=padding,
                               maxpool_kernel_sizes=maxpool_kernel_sizes,
                               downhill=downhill)
        self.decoder = Decoder(first_out_channels * (2**downhill),
                               first_out_channels * (2**(downhill - 1)),
                               exit_channels,
                               padding=padding,
                               maxpool_kernel_sizes=maxpool_kernel_sizes,
                               uphill=downhill)
        self.fcnn = FCNN(exit_channels, 3)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        enc_out, routes = self.encoder(x)
        out = self.decoder(enc_out, routes)
        out = out.permute(0, 2, 1)
        return out
