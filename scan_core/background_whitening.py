# 背景漂白
# background_whitening.py
import cv2
import numpy as np

def whitening_background(img: np.ndarray) -> np.ndarray:
    """
    使用自适应光照校正和形态学操作进行背景漂白，同时尽量避免字迹被腐蚀。
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 减少模糊强度以保护字迹
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=5, sigmaY=5)  # 调整sigma值减少模糊

    # 光照校正，避免除以零
    blurred = np.where(blurred == 0, 1, blurred)  # 修改此处避免除以0错误
    divided = np.divide(gray.astype(np.float32), blurred.astype(np.float32))
    normalized = np.uint8(cv2.normalize(divided, None, 0, 255, cv2.NORM_MINMAX))

    # 使用OTSU阈值法自动决定阈值，二值化后反色，使文字为黑，背景为白
    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary
