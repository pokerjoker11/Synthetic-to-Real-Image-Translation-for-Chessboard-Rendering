# src/models/pix2pix_nets.py
from __future__ import annotations

import torch
import torch.nn as nn


# ----------------------------
# Helpers
# ----------------------------
def get_norm(norm: str, ch: int) -> nn.Module:
    """
    Normalization layer factory.
    - "batch": BatchNorm2d
    - "instance": InstanceNorm2d (recommended for batch_size=1)
    - "none": Identity
    """
    norm = (norm or "none").lower()
    if norm == "batch":
        return nn.BatchNorm2d(ch)
    if norm == "instance":
        # affine=True lets the layer learn scale/shift
        # track_running_stats=False avoids eval-time surprises
        return nn.InstanceNorm2d(ch, affine=True, track_running_stats=False)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm='{norm}'. Use: batch|instance|none")


def init_weights(net: nn.Module) -> None:
    """
    Pix2Pix-style init:
      - Conv/Linear: N(0, 0.02)
      - Norm gamma: N(1, 0.02), beta: 0
    """
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

        # BatchNorm / InstanceNorm
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            if getattr(m, "weight", None) is not None:
                nn.init.normal_(m.weight, 1.0, 0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0.0)


# ----------------------------
# Building blocks
# ----------------------------
class UNetDown(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        normalize: bool = True,
        dropout: float = 0.0,
        norm: str = "instance",
    ):
        super().__init__()
        # If we apply norm, we can safely disable conv bias.
        use_norm = normalize and norm.lower() != "none"
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not use_norm)]
        if use_norm:
            layers.append(get_norm(norm, out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetUp(nn.Module):
    """
    Two upsampling options:
      - up_mode="deconv": ConvTranspose2d (classic pix2pix)
      - up_mode="upsample": Upsample + Conv2d (B3: reduces checkerboard artifacts)
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dropout: float = 0.0,
        norm: str = "instance",
        up_mode: str = "upsample",
    ):
        super().__init__()
        up_mode = up_mode.lower()
        use_norm = norm.lower() != "none"

        layers = []
        if up_mode == "deconv":
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not use_norm))
        elif up_mode == "upsample":
            layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=not use_norm))
        else:
            raise ValueError(f"Unknown up_mode='{up_mode}'. Use: deconv|upsample")

        if use_norm:
            layers.append(get_norm(norm, out_ch))
        layers.append(nn.ReLU(inplace=True))
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
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        base: int = 64,
        norm: str = "instance",     # batch_size=1 => instance is usually better
        up_mode: str = "upsample",  # "upsample" implements B3; "deconv" is classic
    ):
        super().__init__()
        # Encoder (down)
        self.d1 = UNetDown(in_ch, base, normalize=False, norm=norm)          # 256 -> 128
        self.d2 = UNetDown(base, base * 2, norm=norm)                        # 128 -> 64
        self.d3 = UNetDown(base * 2, base * 4, norm=norm)                    # 64 -> 32
        self.d4 = UNetDown(base * 4, base * 8, norm=norm)                    # 32 -> 16
        self.d5 = UNetDown(base * 8, base * 8, norm=norm)                    # 16 -> 8
        self.d6 = UNetDown(base * 8, base * 8, norm=norm)                    # 8 -> 4
        self.d7 = UNetDown(base * 8, base * 8, norm=norm)                    # 4 -> 2
        self.d8 = UNetDown(base * 8, base * 8, normalize=False, norm=norm)   # 2 -> 1 (bottleneck)

        # Decoder (up)
        self.u1 = UNetUp(base * 8, base * 8, dropout=0.5, norm=norm, up_mode=up_mode)        # 1 -> 2
        self.u2 = UNetUp(base * 16, base * 8, dropout=0.5, norm=norm, up_mode=up_mode)       # 2 -> 4
        self.u3 = UNetUp(base * 16, base * 8, dropout=0.5, norm=norm, up_mode=up_mode)       # 4 -> 8
        self.u4 = UNetUp(base * 16, base * 8, dropout=0.0, norm=norm, up_mode=up_mode)       # 8 -> 16
        self.u5 = UNetUp(base * 16, base * 4, dropout=0.0, norm=norm, up_mode=up_mode)       # 16 -> 32
        self.u6 = UNetUp(base * 8, base * 2, dropout=0.0, norm=norm, up_mode=up_mode)        # 32 -> 64
        self.u7 = UNetUp(base * 4, base, dropout=0.0, norm=norm, up_mode=up_mode)            # 64 -> 128

        # Final up to 256
        up_mode = up_mode.lower()
        if up_mode == "deconv":
            self.final = nn.Sequential(
                nn.ConvTranspose2d(base * 2, out_ch, kernel_size=4, stride=2, padding=1),
                nn.Tanh(),
            )
        elif up_mode == "upsample":
            self.final = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(base * 2, out_ch, kernel_size=3, stride=1, padding=1),
                nn.Tanh(),
            )
        else:
            raise ValueError(f"Unknown up_mode='{up_mode}'")

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
    def __init__(self, in_ch: int = 3, base: int = 64, norm: str = "instance"):
        super().__init__()
        # input is concatenation => 2*in_ch channels
        c = in_ch * 2
        norm = (norm or "none").lower()

        def block(in_f: int, out_f: int, normalize: bool = True):
            use_norm = normalize and norm != "none"
            layers = [nn.Conv2d(in_f, out_f, kernel_size=4, stride=2, padding=1, bias=not use_norm)]
            if use_norm:
                layers.append(get_norm(norm, out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(c, base, normalize=False),     # 256 -> 128
            *block(base, base * 2, normalize=True),      # 128 -> 64
            *block(base * 2, base * 4, normalize=True),  # 64 -> 32
            *block(base * 4, base * 8, normalize=True),  # 32 -> 16
            nn.Conv2d(base * 8, 1, kernel_size=4, stride=1, padding=1)       # 16 -> 15 (patch map)
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x = torch.cat([a, b], dim=1)
        return self.model(x)
