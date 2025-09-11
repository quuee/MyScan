# 文字增强与保护
# text_preservation.py
import cv2
import numpy as np
import pytesseract

def enhance_text_structure(img: np.ndarray) -> np.ndarray:
    """
    利用 OCR 辅助判断文字区域，并保护这些区域不被过度处理
    """
    # 方案一
    # # 获取文本轮廓位置
    # h, w = img.shape[:2]
    # boxes = pytesseract.image_to_boxes(img, config='--psm 6')

    # mask = np.zeros((h, w), dtype=np.uint8)

    # for box in boxes.splitlines():
    #     b = box.split(' ')
    #     if len(b) >= 6:
    #         x, y, w1, h1 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    #         y = h - y
    #         h1 = h - h1
    #         cv2.rectangle(mask, (x, h1), (w1, y), 255, -1)

    # # 在原始图像上保留文字区域
    # result = img.copy()
    # result[mask == 255] = 0  # 强化文字为黑色
    # return result

    #方案二
    # result = img.copy()
    # # h, w = img.shape[:2]

    # # 使用 pytesseract 获取文字位置
    # data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    # n_boxes = len(data['level'])

    # for i in range(n_boxes):
    #     if int(data['conf'][i]) > 50:  # 置信度高于50%
    #         (x, y, w_box, h_box) = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
    #         roi = result[y:y+h_box, x:x+w_box]
    #         # 在该区域内增强对比度或锐化
    #         sharpened = cv2.filter2D(roi, -1, kernel=np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]))
    #         result[y:y+h_box, x:x+w_box] = np.minimum(sharpened, 255)

    # return result

    """
    方案三
    使用 OCR 定位文字区域，并进行局部图像增强（不重绘文字）
    目标：提升清晰度，但保留原始字体、粗细、手写风格
    """
    result = img.copy()
    h, w = img.shape[:2]

    # 高精度 OCR 定位（不关心识别内容，只关心位置）
    data = pytesseract.image_to_data(
        img,
        config='--psm 6',  # 假设单块文本
        output_type=pytesseract.Output.DICT
    )
    n_boxes = len(data['level'])

    for i in range(n_boxes):
        conf = int(data['conf'][i])
        if conf > 30:  # 置信度阈值
            x, y, w_box, h_box = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

            # 过滤太小或无效区域
            if w_box < 5 or h_box < 5:
                continue

            # 限制 ROI 在图像范围内
            x, y, w_box, h_box = max(x,0), max(y,0), min(w_box, w-x), min(h_box, h-y)

            # 提取 ROI
            roi = result[y:y+h_box, x:x+w_box].copy()

            # 方法1️⃣：局部对比度拉伸（CLAHE）
            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            # enhanced_roi = clahe.apply(roi)

            # 方法2️⃣：轻微锐化（保护边缘）
            kernel = np.array([[0, -0.5, 0],[-0.5, 3, -0.5],[0, -0.5, 0]])
            enhanced_roi = cv2.filter2D(roi, -1, kernel)

            # 写回原图
            result[y:y+h_box, x:x+w_box] = enhanced_roi

    return result
