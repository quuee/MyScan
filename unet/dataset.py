# dataset.py
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms

class DenoisingDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, image_size=512):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.image_size = image_size
        self.noisy_images = sorted(os.listdir(noisy_dir))
        self.clean_images = sorted(os.listdir(clean_dir))

        # 确保文件名一一对应
        assert len(self.noisy_images) == len(self.clean_images), "Noisy 和 Clean 图片数量不一致"
        for n, c in zip(self.noisy_images, self.clean_images):
            assert n == c, f"文件未对齐: {n} != {c}"

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.BILINEAR),
            transforms.ToTensor()  # 自动归一化到 [0,1]
        ])

    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, idx):
        noisy_path = os.path.join(self.noisy_dir, self.noisy_images[idx])
        clean_path = os.path.join(self.clean_dir, self.clean_images[idx])

        try:
            noisy_img = Image.open(noisy_path).convert("RGB")
            clean_img = Image.open(clean_path).convert("RGB")
        except Exception as e:
            print(f"加载图片失败: {noisy_path} 或 {clean_path}")
            raise e

        noisy_tensor = self.transform(noisy_img)
        clean_tensor = self.transform(clean_img)

        return noisy_tensor, clean_tensor