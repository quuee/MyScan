# 背景漂白
# background_whitening.py
import cv2
import numpy as np

def whitening_background(img: np.ndarray) -> np.ndarray:

    return whitening1(img)


def whitening1(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    """
    使用自适应光照校正和形态学操作进行背景漂白，同时尽量避免字迹被腐蚀。
    """
    # 减少模糊强度以保护字迹
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=5, sigmaY=5)  # 调整sigma值减少模糊

    # 光照校正，避免除以零
    blurred = np.where(blurred == 0, 1, blurred)  # 修改此处避免除以0错误
    divided = np.divide(gray.astype(np.float32), blurred.astype(np.float32))
    normalized = np.uint8(cv2.normalize(divided, None, 0, 255, cv2.NORM_MINMAX))

    # 使用OTSU阈值法自动决定阈值，二值化后反色，使文字为黑，背景为白
    # _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 局部自适应阈值，保留细节更多
    binary = cv2.adaptiveThreshold(normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return binary

def whitening2(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    """
    保留文字色彩的同时对背景进行漂白。
    """
    # 使用边缘检测算法找到文字区域
    edges = cv2.Canny(gray, 50, 150)

    # 膨胀操作，连接靠近的文字边缘
    kernel = np.ones((5, 5), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)

    # 对原图进行高斯模糊，为后续除法做准备
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=5, sigmaY=5)

    # 避免除以零
    blurred = np.where(blurred == 0, 1, blurred)

    # 图像除法去除光照影响
    divided = np.divide(gray.astype(np.float32), blurred.astype(np.float32))
    normalized = np.uint8(cv2.normalize(divided, None, 0, 255, cv2.NORM_MINMAX))

    # 应用局部自适应阈值处理，得到二值化图像
    binary = cv2.adaptiveThreshold(normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 将二值图像反色，使文字变为白色，背景为黑色
    inverted_binary = cv2.bitwise_not(binary)

    # 将文字区域与原图合并，保证文字颜色不变
    mask = cv2.bitwise_and(inverted_binary, edges_dilated)
    white_bg_img = cv2.addWeighted(img, 0.5, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.5, 0)

    # 对于背景部分，使用白色填充
    background_mask = cv2.bitwise_not(mask)
    # white_background = np.ones_like(img) * 255
    white_filled = cv2.bitwise_or(white_bg_img, cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR))

    return white_filled

def whitening3(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    # 使用自适应阈值处理找到文字和背景
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # 对binary图像进行膨胀操作，以确保文字区域被完全覆盖
    kernel = np.ones((3,3), np.uint8)
    dilated_binary = cv2.dilate(binary, kernel, iterations=2)

    # 创建白色背景
    white_bg = np.ones_like(img) * 255

    # 使用dilated_binary作为掩膜，保护文字区域
    result = img.copy()
    for c in range(3):
        result[:,:,c] = np.where(dilated_binary == 0, white_bg[:,:,c], img[:,:,c])

    return result
