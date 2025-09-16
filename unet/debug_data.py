# debug_data.py 判断noisy clean数据源是否一致，一致会导致训练结果无效
from PIL import Image
import numpy as np
import os

noisy_path = "./data/noisy/2.png"
clean_path = "./data/clean/2.png"

noisy = np.array(Image.open(noisy_path).convert("RGB")).astype(np.float32)
clean = np.array(Image.open(clean_path).convert("RGB")).astype(np.float32)

diff = np.mean(np.abs(noisy - clean))
print(f"像素平均差异: {diff:.4f}")

if diff < 5.0:
    print("⚠️  警告：两张图几乎一样！可能是同一张图")
else:
    print("✅ 差异正常")
