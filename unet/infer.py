# infer.py 推理
import torch
from PIL import Image
import torchvision.transforms as T
from unet import UNet
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./unet/checkpoints/unet_denoise_rgb.pth"
IMAGE_SIZE = 512

# 加载模型
model = UNet(in_channels=3, out_channels=3).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# 读图
input_path = "./unet/test_img/0002.jpg"  # 改成你的测试图
original_img = Image.open(input_path).convert("RGB")
orig_size = original_img.size

# 预处理
transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor()  # [0,1]
])
to_pil = T.ToPILImage()

input_tensor = transform(original_img).unsqueeze(0).to(DEVICE)  # [1,3,512,512]

with torch.no_grad():
    output_tensor = model(input_tensor)
    output_tensor = torch.clamp(output_tensor, 0, 1)  # 必须！

    # 🔥 关键调试：打印输出张量信息
    print("\n🔍 推理输出张量分析:")
    print(f"  shape: {output_tensor.shape}")
    print(f"  min: {output_tensor.min().item():.6f}")
    print(f"  max: {output_tensor.max().item():.6f}")
    print(f"  mean: {output_tensor.mean().item():.6f}")
    print(f"  std: {output_tensor.std().item():.6f}")
    print(f"  NaN: {torch.isnan(output_tensor).any().item()}")
    print(f"  Inf: {torch.isinf(output_tensor).any().item()}")

    # 安全处理
    if torch.isnan(output_tensor).any():
        print("⚠️ 检测到 NaN，用零填充")
        output_tensor = torch.zeros_like(output_tensor)
    if torch.isinf(output_tensor).any():
        print("⚠️ 检测到 Inf，裁剪")
        output_tensor = torch.clamp(output_tensor, -10, 10)

    output_tensor = torch.clamp(output_tensor, 0, 1)  # 确保 [0,1]

# 转为 PIL 并恢复原始尺寸
output_pil = to_pil(output_tensor.cpu().squeeze(0))
output_pil = output_pil.resize(orig_size, Image.BILINEAR)
output_pil.save("./unet/test_img/debug_output.png")
print("\n✅ 已保存: debug_output.png")
