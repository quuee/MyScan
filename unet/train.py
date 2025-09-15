# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from unet import UNet
from dataset import DenoisingDataset

# 自定义加权 L1 损失函数
def weighted_l1_loss(pred, target, foreground_weight=10.0, background_weight=1.0):
    """
    pred: 网络输出 [B, 3, H, W]，值范围 [0, 1]
    target: 真实 clean 图 [B, 3, H, W]，值范围 [0, 1]
    foreground_weight: 文字（暗区）权重
    background_weight: 背景（亮区）权重
    """
    # 计算绝对误差
    l1_loss = torch.abs(pred - target)  # [B, 3, H, W]

    # 创建权重矩阵：文字区域权重高，背景权重低
    weights = torch.where(
        target < 0.2,           # 如果 target 像素 < 0.2 → 很可能是文字（黑）
        foreground_weight,      # 高权重
        background_weight       # 否则是背景 → 低权重
    )

    # 加权损失
    weighted_loss = l1_loss * weights
    return weighted_loss.mean()

# 参数设置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 40
BATCH_SIZE = 4  # 根据显存调整（RTX 3090 可用 8）
LR = 1e-4
IMG_SIZE = 512  # 统一分辨率

noisy_dir = "./data/noisy"
clean_dir = "./data/clean"
model_save_path = "./checkpoints/unet_denoise_rgb.pth"

# 数据集
dataset = DenoisingDataset(noisy_dir, clean_dir, image_size=IMG_SIZE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

print(f"   图像数量: {len(dataset)}")
if len(dataset) == 0:
    raise ValueError("❌ 数据集为空！请检查目录中是否有图片文件")

# 模型
model = UNet(in_channels=3, out_channels=3).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.L1Loss()  # 推荐用于图像重建

# 训练
model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    with tqdm(dataloader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}/{EPOCHS}")

        for noisy_batch, clean_batch in tepoch:
            noisy_batch = noisy_batch.to(DEVICE)  # [B, 3, H, W]
            clean_batch = clean_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(noisy_batch)

            loss = weighted_l1_loss(outputs, clean_batch, foreground_weight=10.0, background_weight=1.0)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            tepoch.set_postfix(loss=loss.item())

            # 调试：检查输出范围
            if epoch == 0 and loss.item() < 1e-5:
                print("\n⚠️  损失过小，可能数据相同或加载错误")
            if torch.isnan(loss):
                print("\n💥 损失为 NaN！梯度爆炸")
                break

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Average Loss: {avg_loss:.6f}")

    # 每10轮保存一次
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"./checkpoints/unet_epoch_{epoch+1}.pth")

# 保存最终模型
torch.save(model.state_dict(), model_save_path)
print(f"\n✅ 模型已保存: {model_save_path}")
