import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

from tools import TransformerBlock


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()

        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):

        x = self.down(x)

        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):

        x = self.up(x)

        return x


class CasConvFeatureAggregation(nn.Module):
    def __init__(self, in_channels, n=6):
        super(CasConvFeatureAggregation, self).__init__()

        self.n = n

        self.in_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
        )

        self.conv_list = nn.ModuleList()
        self.conv_list.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'))
        for i in range(1, self.n):
            self.conv_list.append(nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            ))

        self.fuse_conv_list = nn.ModuleList()
        for i in range(self.n - 1):
            self.fuse_conv_list.append(nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
                nn.ReLU(),
            ))

        self.final_fuse = nn.Sequential(
            nn.Conv2d(in_channels * (self.n - 1), in_channels * (self.n - 1), kernel_size=3, stride=1, padding=1, groups=in_channels, padding_mode='reflect'),
            nn.Conv2d(in_channels * (self.n - 1), in_channels, kernel_size=1),
        )

    def forward(self, x):

        res = x
        x = self.in_conv(x)

        fea_list = []
        for i in range(self.n):
            x = self.conv_list[i](x)
            fea_list.append(x)

        fuse_list = []
        for i in range(self.n - 1):
            x = torch.cat([fea_list[i], fea_list[i + 1]], dim=1)
            x = self.fuse_conv_list[i](x)
            fuse_list.append(x)

        x = torch.cat(fuse_list, dim=1)

        # channel shuffle
        B, C, H, W = x.shape
        x = x.view(B, -1, self.n - 1, H, W).transpose(1, 2).contiguous().view(B, -1, H, W)

        x = self.final_fuse(x)

        return x + res



class ANet(nn.Module):
    def __init__(self, in_channels):
        super(ANet, self).__init__()

        self.dw_layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=1, padding=1, groups=in_channels, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, padding_mode='reflect'),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
        )


    def forward(self, x):

        x = self.dw_layer(x)
        x = self.gap(x)
        x = self.mlp(x)

        return x


class TNet(nn.Module):
    def __init__(self, in_channels):
        super(TNet, self).__init__()

        self.transformer = nn.Sequential(*[TransformerBlock(dim=in_channels, num_heads=8, ffn_expansion_factor=2) for _ in range(10)])
        self.bn = nn.BatchNorm2d(in_channels)  # norm

    def forward(self, x):

        x = self.transformer(x) + x
        x = self.bn(x)

        return x


class InverseAtmosphericScatteringModel(nn.Module):
    def __init__(self, in_channels):
        super(InverseAtmosphericScatteringModel, self).__init__()

        self.in_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
        )

        self.estimate_A = ANet(in_channels)
        self.estimate_T = TNet(in_channels)

        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

    def forward(self, x):

        in_x = self.in_conv(x)
        a = self.estimate_A(in_x)
        t = self.estimate_T(in_x)

        x = (a - x) * (1 - t)
        x = self.out_conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, n):
        super(Encoder, self).__init__()

        self.ccfa = CasConvFeatureAggregation(in_channels, n)

    def forward(self, x):

        x = self.ccfa(x)

        return x


class Latent(nn.Module):
    def __init__(self, in_channels):
        super(Latent, self).__init__()

        self.transformer = nn.Sequential(*[TransformerBlock(dim=in_channels, num_heads=8, ffn_expansion_factor=2) for _ in range(10)])

    def forward(self, x):

        x = self.transformer(x) + x

        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n):
        super(Decoder, self).__init__()

        self.ccfa = CasConvFeatureAggregation(in_channels, n)
        self.iasm = InverseAtmosphericScatteringModel(in_channels)

    def forward(self, x):

        x = self.ccfa(x)
        x = self.iasm(x)

        return x


class Generator(nn.Module):
    def __init__(self, base_channels=32):
        super(Generator, self).__init__()

        self.inconv = nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.inconv_2 = nn.Conv2d(3, base_channels * 2, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.inconv_3 = nn.Conv2d(3, base_channels * 4, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

        self.outconv1 = nn.Conv2d(base_channels * 4, 3, kernel_size=1)
        self.outconv2 = nn.Conv2d(base_channels * 2, 3, kernel_size=1)
        self.outconv = nn.Conv2d(base_channels, 3, kernel_size=1)

        self.down_1 = Down(base_channels, base_channels * 2)
        self.down_2 = Down(base_channels * 2, base_channels * 4)
        self.up_1 = Up(base_channels * 4, base_channels * 2)
        self.up_2 = Up(base_channels * 2, base_channels)


        self.encoder_1 = Encoder(base_channels, n=6)
        self.encoder_2 = Encoder(base_channels * 2, n=6)
        self.encoder_3 = Encoder(base_channels * 4, n=6)
        self.latent = Latent(base_channels * 4)
        self.decoder_1 = Decoder(base_channels * 4, n=6)
        self.decoder_2 = Decoder(base_channels * 2, n=6)
        self.decoder_3 = Decoder(base_channels, n=6)

    def forward(self, input):

        input_2 = F.interpolate(input, scale_factor=0.5)
        input_4 = F.interpolate(input, scale_factor=0.25)
        x_2 = self.inconv_2(input_2)
        x_4 = self.inconv_3(input_4)

        x = self.inconv(input)           # 32 * hw
        x = self.encoder_1(x)
        skip1 = x

        x = self.down_1(x) + x_2        # 64 * hw/2
        x = self.encoder_2(x)
        skip2 = x

        x = self.down_2(x) + x_4        # 128 * hw/4
        x = self.encoder_3(x)

        x = self.latent(x)

        x = self.decoder_1(x)
        out_4 = self.outconv1(x)
        x = self.up_1(x) + skip2         # 64 * hw/2

        x = self.decoder_2(x)
        out_2 = self.outconv2(x)
        x = self.up_2(x) + skip1         # 32 * hw

        x = self.decoder_3(x)
        out = self.outconv(x)

        return [input_4 + out_4, input_2 + out_2, input + out]


if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Generator().to(device)


    params = sum([param.numel() for param in model.parameters()])
    params_m = params / 1000000
    print(f'params:{params_m:.2f}M')


    input_tensor = torch.randn(1, 3, 256, 256).to(device)
    flops, _ = profile(model, inputs=(input_tensor,))
    flops_g = flops / 1000000000
    print(f'FLOPs:{flops_g:.2f}G')