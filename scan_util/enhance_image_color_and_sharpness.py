import cv2
import numpy as np


def simple_enhance(image: np.ndarray,)-> np.ndarray:
    # 1. 锐化
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    # 2. 对比度和亮度调整
    enhanced = cv2.convertScaleAbs(sharpened, alpha=1.5, beta=50)
    return enhanced

def enhance_image_color_and_sharpness(
    image: np.ndarray,
    sharpen_strength: float = 1.0,
    saturation_factor: float = 1.5,
    contrast_clip_limit: float = 2.0,
) -> np.ndarray:
    """
    对图像进行锐化和颜色深化处理（增强饱和度和对比度），并保存结果。

    参数：
        image (np.ndarray): 输入的BGR图像
        output_dir (str): 输出目录路径
        sharpen_strength (float): 锐化强度（默认1.0，可调整核权重）
        saturation_factor (float): 饱和度增强倍数（建议1.3~2.0）
        contrast_clip_limit (float): CLAHE对比度限制（建议1.5~3.0）
        save_prefix (str): 保存文件名前缀

    返回：
        np.ndarray: 增强后的图像
    """

    # 1. 锐化：使用拉普拉斯卷积核
    kernel = np.array([
        [0, -1 * sharpen_strength, 0],
        [-1 * sharpen_strength, 4 * sharpen_strength + 1, -1 * sharpen_strength],
        [0, -1 * sharpen_strength, 0]
    ])
    sharpened = cv2.filter2D(image, -1, kernel)

    # 2. 深化颜色：仅对彩色图像处理
    if len(sharpened.shape) == 3 and sharpened.dtype == np.uint8:
        hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # 增强饱和度
        s = cv2.multiply(s, saturation_factor)
        s = np.clip(s, 0, 255).astype(np.uint8)

        # 使用CLAHE增强明度通道（V）的局部对比度
        clahe = cv2.createCLAHE(clipLimit=contrast_clip_limit, tileGridSize=(8, 8))
        v = clahe.apply(v)

        # 合并通道并转回BGR
        enhanced_hsv = cv2.merge([h, s, v])
        enhanced = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
    else:
        # 若为灰度图，仅做对比度/亮度增强
        enhanced = cv2.convertScaleAbs(sharpened, alpha=1.5, beta=30)

    return enhanced

