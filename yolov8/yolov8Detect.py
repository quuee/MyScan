from ultralytics import YOLO
import numpy as np
from scan_util import my_utils

model = YOLO('./yolov8/yolov8s-seg-document.pt')

def yolov8Detect(image: np.ndarray) -> np.ndarray :
  # 推理
  results = model(image, conf=0.4, imgsz=640, classes=[0], max_det=1)
  if results[0].masks is None:
    print("❌ 未检测到文档！")
    return

  masks = results[0].masks.xy
  boxes = results[0].boxes
  confs = boxes.conf.cpu().numpy()

  best_idx = np.argmax(confs)
  mask = masks[best_idx]
  quad = my_utils.mask_to_quad(mask, epsilon_factor=0.015)
  ordered_quad = my_utils.order_points(quad)

  # 返回四个角坐标
  return ordered_quad
