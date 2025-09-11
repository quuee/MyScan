import cv2
import numpy as np

# 透视变换
def warp_document(image, quad, width=None, height=None)-> np.ndarray:
    """
    将四边形区域矫正为矩形
    自动计算高分辨率输出尺寸，保持长宽比
    """
    # 计算原始四边形的宽度和高度（像素距离）
    (tl, tr, br, bl) = quad

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 如果未指定，则使用检测到的实际尺寸
    if width is None:
        width = maxWidth
    if height is None:
        height = maxHeight

    # 限制最大尺寸防止超内存（可选）
    max_output_size = 2000
    if width > max_output_size or height > max_output_size:
        scale = max_output_size / max(width, height)
        width = int(width * scale)
        height = int(height * scale)

    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(quad, dst_pts)
    warped = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_CUBIC)
    return warped
