import cv2
import numpy as np
import os
from ultralytics import YOLO
from datetime import datetime
from scan_core import my_utils
from scan_core import warp_document
from scan_core import background_whitening
from scan_core import enhance_image_color_and_sharpness
from unet.denoiser import DocumentDenoiser


# ----------------------------
# 加载模型 & 打开摄像头
# ----------------------------
model = YOLO('./model/yolov8s-seg-document.pt')

denoiser = DocumentDenoiser(
        model_path="./unet/checkpoints/unet_denoise_rgb.pth",
        device="cpu"  # 或 "cpu"
    )

stream_url = "http://192.168.2.104:4747/video" # 手机ip相机
cap = cv2.VideoCapture(stream_url)

output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

# ----------------------------
# 主循环
# ----------------------------
prev_quad = None
prev_conf = 0.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    orig_frame = frame.copy()
    display = frame.copy()
    h, w = frame.shape[:2]

    # 推理
    results = model(frame, conf=0.4, imgsz=640,classes=[0],max_det=1)

    current_quad = None
    current_conf = 0.0

    if results[0].masks is not None:
        masks = results[0].masks.xy
        boxes = results[0].boxes
        confs = boxes.conf.cpu().numpy()

        # 选置信度最高的
        best_idx = np.argmax(confs)
        best_conf = confs[best_idx]

        if best_conf > 0.4:
            mask = masks[best_idx]
            quad = my_utils.mask_to_quad(mask, epsilon_factor=0.015)  # 调整精度

            # 排序四个点
            ordered_quad = my_utils.order_points(quad)

            current_quad = ordered_quad
            current_conf = best_conf

    # 平滑策略：延续上一帧结果
    final_quad = current_quad if current_quad is not None else prev_quad
    prev_conf = current_conf if current_quad is not None else prev_conf * 0.9

    # 绘制当前或缓存的 quad
    if final_quad is not None:
        # 绘制四边形
        pts = final_quad.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(display, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

        # 填充浅色
        overlay = display.copy()
        cv2.fillPoly(overlay, [pts], color=(200, 200, 255))
        cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)

        # 实时透视矫正预览（小窗口）
        # 实时预览小图（可缩小用于显示，但不影响保存）
        try:
            # 用于显示的小图（缩放）
            # 缩略图
            warped_small = cv2.resize(final_warped, (300, 200), interpolation=cv2.INTER_AREA)
            # 将缩略图复制到展示图像上
            # 高度方向上从20像素开始,宽度方向
            display[20:220, w-320:w-20] = warped_small
            # 画矩形
            cv2.rectangle(display, (w-320, 20), (w-20, 220), (0,255,0), 2)
        except:
            pass

    # 更新缓存
    if current_quad is not None:
        prev_quad = current_quad

    # 显示
    cv2.imshow('Detection', display)


    key = cv2.waitKey(1) & 0xFF
    # 'q' 退出
    if key == ord('q'):
        break
    # 按 's' 保存矫正图
    elif key == ord('s') and final_quad is not None:
        # ✅ 使用动态高分辨率进行透视变换
        final_warped = warp_document.warp_document(orig_frame, final_quad, width=None, height=None)  # 自动计算尺寸

        # enhance = enhance_image_color_and_sharpness.enhance_image_color_and_sharpness(final_warped)

        # 漂白
        # white_bg = background_whitening.whitening_background(final_warped)

        # 漂白 去噪
        denoise_res = denoiser.denoise(final_warped)
        result_uint8 = (denoise_res * 255).astype(np.uint8)  # [0,1] → [0,255]
        result_bgr = cv2.cvtColor(result_uint8, cv2.COLOR_RGB2BGR)

        now = datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")

        # 保存高清结果
        cv2.imwrite(f'{output_dir}/camera_{formatted_datetime}.jpg', result_bgr)


cap.release()
cv2.destroyAllWindows()
