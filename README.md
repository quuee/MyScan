## 依赖

`pip install opencv-python pytesseract ultralytics`

额外安装 `sudo apt install tesseract-ocr` 才能使用 pytesseract（目前不会用，效果没有）

## 入口

- scan_document.py 单张图片
- camera_scan_document.py 连接手机相机

## TODO

- 1、目标检测精度勉强够用
- 2、背景漂白时，总把文字、线框给腐蚀掉（有孔洞），不够清晰，需要增强清晰度（最近邻插值，只是放大了）；采用 unet 进行去噪
- 3、透视变换不够智能，矫正后的图像包含了原始图像的边缘（如纸张边框、阴影、污渍），希望 只保留内容区域，去除边缘，得到一个“纯白色背景 + 内容居中”的干净矩形图
- 4、需要提取文字、水印原本的颜色回显
- 5、添加自定义水印

## 实现方式

- 目标检测 yolov8-seg 返回四角坐标
- opencv 透视变换（不够智能）
- 语义分割模型 https://github.com/milesial/Pytorch-UNet 进行去噪
