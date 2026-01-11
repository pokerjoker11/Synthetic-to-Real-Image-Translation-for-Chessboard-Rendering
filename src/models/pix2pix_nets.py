# src/models/pix2pix_nets.py
from __future__ import annotations

import torch
import torch.nn as nn


# ----------------------------
# Helpers
# ----------------------------
def init_weights(net: nn.Module) -> None:
    """
    Pix2Pix-style init: normal(0, 0.02) for conv/linear; BN gamma normal(1,0.02), beta=0.
    """
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.normal_(m.weight, 1.0, 0.02)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


class UNetDown(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, normalize: bool = True, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not normalize)
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetUp(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        # concat skip connection
        return torch.cat([x, skip], dim=1)


# ----------------------------
# Generator: U-Net (Pix2Pix)
# ----------------------------
class UNetGenerator(nn.Module):
    """
    U-Net generator as in Pix2Pix.
    Input/Output: RGB in [-1,1]
    """
    def __init__(self, in_ch: int = 3, out_ch: int = 3, base: int = 64):
        super().__init__()
        # Encoder (down)
        self.d1 = UNetDown(in_ch, base, normalize=False)         # 256 -> 128
        self.d2 = UNetDown(base, base * 2)                       # 128 -> 64
        self.d3 = UNetDown(base * 2, base * 4)                   # 64 -> 32
        self.d4 = UNetDown(base * 4, base * 8)                   # 32 -> 16
        self.d5 = UNetDown(base * 8, base * 8)                   # 16 -> 8
        self.d6 = UNetDown(base * 8, base * 8)                   # 8 -> 4
        self.d7 = UNetDown(base * 8, base * 8)                   # 4 -> 2
        self.d8 = UNetDown(base * 8, base * 8, normalize=False)  # 2 -> 1 (bottleneck)

        # Decoder (up)
        self.u1 = UNetUp(base * 8, base * 8, dropout=0.5)        # 1 -> 2
        self.u2 = UNetUp(base * 16, base * 8, dropout=0.5)       # 2 -> 4
        self.u3 = UNetUp(base * 16, base * 8, dropout=0.5)       # 4 -> 8
        self.u4 = UNetUp(base * 16, base * 8)                    # 8 -> 16
        self.u5 = UNetUp(base * 16, base * 4)                    # 16 -> 32
        self.u6 = UNetUp(base * 8, base * 2)                     # 32 -> 64
        self.u7 = UNetUp(base * 4, base)                         # 64 -> 128

        self.final = nn.Sequential(
            nn.ConvTranspose2d(base * 2, out_ch, kernel_size=4, stride=2, padding=1),  # 128 -> 256
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)

        u1 = self.u1(d8, d7)
        u2 = self.u2(u1, d6)
        u3 = self.u3(u2, d5)
        u4 = self.u4(u3, d4)
        u5 = self.u5(u4, d3)
        u6 = self.u6(u5, d2)
        u7 = self.u7(u6, d1)
        out = self.final(u7)
        return out


# ----------------------------
# Discriminator: PatchGAN
# ----------------------------
class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator as in Pix2Pix.
    Takes concatenated (A,B) => predicts patch realism map.
    """
    def __init__(self, in_ch: int = 3, base: int = 64):
        super().__init__()
        # input is concatenation => 2*in_ch channels
        c = in_ch * 2

        def block(in_f, out_f, normalize=True):
            layers = [nn.Conv2d(in_f, out_f, 4, 2, 1, bias=not normalize)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(c, base, normalize=False),     # 256 -> 128
            *block(base, base * 2),               # 128 -> 64
            *block(base * 2, base * 4),           # 64 -> 32
            *block(base * 4, base * 8, normalize=True),  # 32 -> 16
            nn.Conv2d(base * 8, 1, 4, 1, 1)       # 16 -> 15 (patch map)
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x = torch.cat([a, b], dim=1)
        return self.model(x)
