
# 表格/边框/线条保留
# table_preservation.py
import cv2
import numpy as np

def preserve_table_lines(img: np.ndarray,min_length=50, thickness=1) -> np.ndarray:
    """
    方案一
    提取水平和垂直线条以保留表格结构
    """
    # 输入应为二值图
    binary = img.copy()
    if binary.dtype != np.uint8:
        binary = np.uint8(binary)

    # 定义结构元素
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    # 提取水平线
    hor_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    # 提取垂直线
    ver_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    # 合并线条
    table_mask = cv2.addWeighted(hor_lines, 1, ver_lines, 1, 0.0)

    # 将线条加回原图（避免被误判为噪声）
    result = cv2.addWeighted(binary, 1, table_mask, 1, 0.0)
    return result

def enhance_table_lines(image, min_length=60, thickness=2):
    """
    增强图像中的表格线。

    :param image: 输入图像（灰度图）
    :param min_length: 线条检测的最小长度
    :param thickness: 输出线条的厚度
    :return: 强调了表格线的新图像
    """
    # 使用Canny边缘检测来寻找图像中的边缘
    # 使用Canny边缘检测来寻找图像中的边缘
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # 使用Hough变换来检测直线
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=min_length, maxLineGap=10)

    # 创建一个新的图像用于绘制线条，确保它是三通道以便于可视化（如果需要）
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 在line_image上画线，颜色为白色，厚度由thickness参数控制
            cv2.line(line_image, (x1, y1), (x2, y2), 255, thickness)

    return line_image
