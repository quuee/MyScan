# 去除噪点
# noise_removal.py

import cv2
import numpy as np

def remove_noise(img: np.ndarray, method='morphological') -> np.ndarray:
    """
    去除图像中的小面积噪点
    """
    if method == 'morphological':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
        return cleaned

    elif method == 'dnn':  # 可选：使用轻量级 DNN 去噪（如 OpenCV 的 denoising_DCT）
        import cv2.dnn_superres
        # 示例：可用预训练模型，但此处简化为非DNN版本
        pass

    return img
