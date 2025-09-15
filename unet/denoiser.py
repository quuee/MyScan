# 给外部使用
# denoiser.py
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from unet.unet import UNet

import os


class DocumentDenoiser:
    """
    文档去噪器：输入/输出均为 numpy array
    输入: np.ndarray (H, W, 3), dtype=uint8 [0,255] 或 float32 [0,1]
    输出: np.ndarray (H, W, 3), dtype=float32, range [0,1]
    """

    def __init__(self, model_path, img_size=512, device=None):
        self.img_size = img_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 构建模型
        self.model = UNet(in_channels=3, out_channels=3)
        self.model.to(self.device)
        self.model.eval()

        # 加载权重
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"✅ 模型已加载: {model_path} (device={self.device})")
        except Exception as e:
            raise RuntimeError(f"❌ 加载模型失败: {e}")

        # 预处理 transform
        self.transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),  # 自动归一化到 [0,1]
        ])

    def _numpy_to_pil(self, image: np.ndarray) -> Image.Image:
        if image.dtype == np.uint8:
            pass
        elif image.dtype in [np.float32, np.float64]:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)  # 假设已经是 [0,255]
        else:
            raise TypeError(f"不支持的数据类型: {image.dtype}")

        return Image.fromarray(image)

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        将 [1,3,H,W] tensor 转为 (H, W, 3) numpy array
        """
        # 移除 batch 维度并转到 cpu
        tensor = tensor.squeeze(0).cpu().clamp(0, 1)  # [3,H,W]
        # 转为 numpy (H, W, 3)
        array = tensor.permute(1, 2, 0).numpy()  # [H,W,3]
        return array.astype(np.float32)

    @torch.no_grad()
    def denoise(self, image: np.ndarray, resize_to_original=True) -> np.ndarray:
        """
        对单张 numpy 图像去噪
        :param image: np.ndarray, shape=(H, W, 3), dtype=uint8 或 float32
        :param resize_to_original: 是否恢复原始尺寸
        :return: np.ndarray, shape=(H, W, 3), dtype=float32, range [0,1]
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("输入必须是 numpy.ndarray")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"输入必须是 (H, W, 3) 的图像，当前 shape: {image.shape}")

        original_h, original_w = image.shape[:2]

        # 转为 PIL 进行统一预处理
        pil_image = self._numpy_to_pil(image)

        # 预处理：调整大小 + 转为 tensor
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)  # [1,3,512,512]

        # 推理
        output_tensor = self.model(input_tensor) # [1,3,512,512]
        print(f"📌 模型原始输出: min={output_tensor.min().item():.4f}, "
            f"mean={output_tensor.mean().item():.4f}, max={output_tensor.max().item():.4f}")

        # 后处理：裁剪异常值
        output_tensor = torch.clamp(output_tensor, 0, 1)

        # 转回 numpy
        output_array = self._tensor_to_numpy(output_tensor)  # [H,W,3], float32, [0,1]

        # 是否恢复原始分辨率？
        if resize_to_original and (output_array.shape[0] != original_h or output_array.shape[1] != original_w):
            # 使用 scipy 或 cv2 更高效，这里用 PIL 简单实现
            pil_out = Image.fromarray((output_array * 255).astype(np.uint8))
            pil_out = pil_out.resize((original_w, original_h), Image.BILINEAR)
            output_array = np.array(pil_out).astype(np.float32) / 255.0  # [H,W,3], float32, [0,1]

        return output_array

    @torch.no_grad()
    def denoise_batch(self, images):
        """
        批量去噪（逐张处理）
        :param images: list of np.ndarray
        :return: list of np.ndarray
        """
        results = []
        for img in images:
            try:
                result = self.denoise(img)
                results.append(result)
            except Exception as e:
                print(f"⚠️ 去噪失败: {e}")
                results.append(None)
        return results

    def set_device(self, device):
        self.device = device
        self.model.to(device)
        print(f"🔧 设备已切换至: {device}")
