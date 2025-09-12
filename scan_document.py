import cv2
import os
import numpy as np
from scan_core import yolov8Detect
from scan_core import warp_document
from scan_core import background_whitening
from scan_core import enhance_image_color_and_sharpness
from scan_core import table_preservation
from scan_core import noise_removal
from scan_core import text_preservation

from datetime import datetime

input_image_path = './test_image/0080.jpg'
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

# 读取图片
image = cv2.imread(input_image_path)
orig = image.copy()

if image is None:
    raise FileNotFoundError(f"❌ 无法读取图像：{input_image_path}")

now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# yolo检测 返回四个角坐标
ordered_quad = yolov8Detect.yolov8Detect(orig)

# 绘制检测结果（可选）
debug_img = orig.copy()
pts = ordered_quad.reshape(-1, 1, 2).astype(int)
cv2.polylines(debug_img, [pts], True, (0, 255, 0), 8)
cv2.imwrite(f"{output_dir}/01_detected_{now}.jpg", debug_img)
print("✅ 已保存：01_detected_{now}.jpg")

# 透视变换
warped = warp_document.warp_document(orig, ordered_quad, width=None, height=None)
cv2.imwrite(f"{output_dir}/02_warped_{now}.jpg", warped)
print("✅ 已保存：02_warped_{now}.jpg")

# 增强文字结构 ocr辅助  TODO 没用
# text_enhance = text_preservation.enhance_text_structure(warped)
# cv2.imwrite(f"{output_dir}/03_text_enhance_{now}.jpg", text_enhance)
# print("✅ 已保存：03_text_enhance_{now}.jpg")

# 保留表格线 TODO 没用
# with_table = table_preservation.preserve_table_lines(text_enhance)
# cv2.imwrite(f"{output_dir}/04_with_table_{now}.jpg", with_table)
# print("✅ 已保存：04_with_table_{now}.jpg")


enhanced = enhance_image_color_and_sharpness.simple_enhance(warped)
cv2.imwrite(f"{output_dir}/03_enhanced_{now}.jpg", enhanced)
print("✅ 已保存：03_enhanced_{now}.jpg")

# 背景漂白
white_bg = background_whitening.whitening_background(enhanced)
cv2.imwrite(f"{output_dir}/05_white_bg_{now}.jpg", white_bg)
print("✅ 已保存：05_white_bg_{now}.jpg")

# 去除噪点 TODO 没用
# removeNoise = noise_removal.remove_noise(white_bg)
# cv2.imwrite(f"{output_dir}/06_removeNoise_{now}.jpg", white_bg)
# print("✅ 已保存：06_removeNoise_{now}.jpg")



# chazhi = cv2.resize(enhanced,None,fx=2,fy=2,interpolation=cv2.INTER_LINEAR)
# cv2.imwrite(f"{output_dir}/08_INTER_NEAREST_{now}.jpg", chazhi)
# print("✅ 已保存：08_INTER_NEAREST_{now}.jpg")


