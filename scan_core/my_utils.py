import cv2
import numpy as np

# 多边形 → 四边形近似
def mask_to_quad(mask, epsilon_factor=0.02)-> np.ndarray:
    """
    将分割轮廓近似为四边形
    mask: (n, 2) 数组，来自 results[0].masks.xy[0]
    epsilon_factor: 控制近似精度，越小越接近原形
    """
    # 转为整数点
    pts = mask.astype(np.float32)

    # 计算周长并近似为多边形
    peri = cv2.arcLength(pts, closed=True)
    epsilon = epsilon_factor * peri
    approx = cv2.approxPolyDP(pts, epsilon, closed=True)

    # 如果近似为 4 个点，直接返回
    if len(approx) == 4:
        return approx.reshape(4, 2)

    # 如果不是 4 个点，强制取凸包并近似为四边形
    hull = cv2.convexHull(approx.reshape(-1, 2))
    peri = cv2.arcLength(hull, closed=True)
    epsilon = epsilon_factor * peri
    approx = cv2.approxPolyDP(hull, epsilon, closed=True)

    if len(approx) == 4:
        return approx.reshape(4, 2)
    else:
        # 强制返回外接矩形的四个角（最后手段）
        x, y, w, h = cv2.boundingRect(hull)
        return np.array([[x,y], [x+w,y], [x+w,y+h], [x,y+h]], dtype=np.float32)


# 四个点排序：左上、右上、右下、左下
def order_points(pts)-> np.ndarray:
    """
    输入 4 个点，输出顺时针排序：[左上, 右上, 右下, 左下]
    """
    rect = np.zeros((4, 2), dtype="float32")

    # 按 x+y 排序，最小的是左上，最大的是右下
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]      # 左上 = x+y 最小
    rect[2] = pts[np.argmax(s)]      # 右下 = x+y 最大
    rect[1] = pts[np.argmin(diff)]   # 右上 = x-y 最小
    rect[3] = pts[np.argmax(diff)]   # 左下 = x-y 最大

    return rect


