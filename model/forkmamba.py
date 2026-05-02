import torch
import torch.nn as nn
from utils.amvss import Adap_Multi_VSS, Conv_Free_VSS
from utils.skip_connection import FusionBlock

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        kernel_size = 3
        ):

        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size = kernel_size, padding = 'same')
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU()


    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))

        return 
        
class EncoderBlock(nn.Module):

  def __init__(
    self,
    in_c: int,
    out_c: int
    ):

    super(EncoderBlock, self).__init__()
    self.in_c = in_c
    self.conv = ConvBlock(in_c, out_c)
    self.vss = Adap_Multi_VSS(hidden_dim = in_c)
    self.down = nn.MaxPool2d((2,2))

  def forward(self, x):

    x = self.vss(x)
    skip = self.conv(x)
    x = self.down(skip)

    return x, skip

class Encoder(nn.Module):

    def __init__(
        self,
        in_c = 16,
        depth = 5
        ):

        super(Encoder, self).__init__()
        in_channels = [in_c * 2 ** i for i in range(depth)]

        encoder_blocks = []

        for i in range(depth):
            encoder_blocks.append(EncoderBlock(in_channels[i], 2 * in_channels[i]))

        self.encoder_blocks = nn.ModuleList(encoder_blocks)

    def forward(self, x):
        skip_connections = []

        for encoder_block in self.encoder_blocks:
            x, skip = encoder_block(x)
            skip_connections.append(skip)

        return x, skip_connections

class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_c: int,
        out_c: int
        ):

        super(DecoderBlock, self).__init__()
        self.up = nn.Upsample(scale_factor = 2)
        self.sb = FusionBlock(in_c, in_c // 2)

        self.conv = ConvBlock(in_c, out_c)
        self.vss = Adap_Multi_VSS(hidden_dim = out_c)

    def forward(self, x, skip):

        x = self.up(x)
        x = self.sb(x, skip)
        x = self.conv(x)
        x = self.vss(x)

        return x

class Decoder(nn.Module):

    def __init__(
        self,
        out_c = 16,
        depth = 5
        ):

        super(Decoder, self).__init__()
        in_channels = [out_c * 2 ** (depth - i) for i in range(depth)]

        decoder_blocks = []
        for in_channel in in_channels:
            decoder_blocks.append(DecoderBlock(in_channel, in_channel // 2))

        self.decoder_blocks = nn.ModuleList(decoder_blocks)

    def forward(self, x, skip_features):

        skip_features.reverse()

        for decoder_block, skip in zip(self.decoder_blocks, skip_features):
            x = decoder_block(x, skip)

        return x

class BottleNeck(nn.Module):

    def __init__(self, in_c, branches = 4):
        super().__init__()
        assert in_c % branches == 0, f"hidden_dim must be divisible by {branches}"

        self.dw_blocks = []
        self.vss_blocks = []
        self.branches = branches
        for i in range(branches):
            self.vss_blocks.append(Conv_Free_VSS(hidden_dim = in_c // branches, channel_first = True))
            self.dw_blocks.append(nn.Conv2d(in_c // branches, in_c // branches, kernel_size=3, stride=1, padding = 'same', groups = in_c // branches, dilation = i + 1))
        self.vss_blocks = nn.ModuleList(self.vss_blocks)
        self.dw_blocks = nn.ModuleList(self.dw_blocks)

        self.skip_scale = nn.Parameter(torch.ones(1))
        self.act = nn.ReLU()

    def forward(self, x):

        x_chunks = torch.chunk(x, chunks = self.branches, dim = 1)

        x_mamba = []
        for idx, chunk in enumerate(x_chunks):
            chunk = self.act(self.dw_blocks[idx](chunk))
            chunk = self.vss_blocks[idx](chunk)
            x_mamba.append(chunk)
        del x_chunks
        x = torch.cat(x_mamba, dim = 1) + self.skip_scale * x

        return x

    
class ForkMamba(nn.Module):
    def __init__(
        self,
        in_c = 3,
        out_c = 1,
        depth = 5
        ):

        super(ForkMamba, self).__init__()
        self.in_c = in_c
        self.out_c = out_c

        self.conv_in = nn.Conv2d(in_c, 16, kernel_size=7, padding ='same')
        self.conv_out = nn.Conv2d(16, out_c, kernel_size=1)

        self.En = Encoder(in_c = 16, depth = depth)
        self.BN = BottleNeck(16 * 2 ** depth)
        self.De = Decoder(out_c = 16, depth = depth)

    def forward(self, x):
        assert x.shape[1] == self.in_c, f"Input channels must be {self.in_c}"
        x = self.conv_in(x)
        x, skip_features = self.En(x)
        x = self.BN(x)
        x = self.De(x, skip_features)
        x = self.conv_out(x)
        return x