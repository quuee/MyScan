# unet.py
import torch
import torch.nn as nn
import torch.nn.init as init


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()
        self.conv_down1 = double_conv(in_channels, 64)
        self.conv_down2 = double_conv(64, 128)
        self.conv_down3 = double_conv(128, 256)
        self.conv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = double_conv(256 + 512, 256)
        self.conv_up2 = double_conv(128 + 256, 128)
        self.conv_up1 = double_conv(64 + 128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, 1)

        # 🔥 关键：手动初始化 final_conv，防止输出饱和
        init.xavier_uniform_(self.final_conv.weight)
        init.constant_(self.final_conv.bias, 0)

        # 权重初始化
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        c1 = self.conv_down1(x)     # 64
        p1 = self.maxpool(c1)

        c2 = self.conv_down2(p1)    # 128
        p2 = self.maxpool(c2)

        c3 = self.conv_down3(p2)    # 256
        p3 = self.maxpool(c3)

        c4 = self.conv_down4(p3)    # 512

        u3 = self.upsample(c4)
        u3 = torch.cat([u3, c3], dim=1)
        u3 = self.conv_up3(u3)

        u2 = self.upsample(u3)
        u2 = torch.cat([u2, c2], dim=1)
        u2 = self.conv_up2(u2)

        u1 = self.upsample(u2)
        u1 = torch.cat([u1, c1], dim=1)
        u1 = self.conv_up1(u1)

        outputs = self.final_conv(u1)
        return torch.sigmoid(outputs)  # 输出 [0,1]
        # return torch.tanh(outputs)  # ✅ 输出范围 [-1, 1]
