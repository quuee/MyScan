## 依赖

`pip install opencv-python pytesseract ultralytics`

额外安装 `sudo apt install tesseract-ocr` 才能使用 pytesseract（目前不会用，效果没有）

## 入口

- scan_document.py 单张图片
- camera_scan_document.py 连接手机相机

## TODO

- 1、目标检测精度勉强够用
- 2、背景漂白时，总把文字、线框给腐蚀掉（有孔洞），不够清晰，需要增强清晰度
  采用 unet 进行去噪（目前只是训练黑白图、清晰度还不够，没有粗体、表格、多种语言和字体，样本太少了）
- 3、透视变换不够智能，矫正后的图像包含了原始图像的边缘（如纸张边框、阴影、污渍），希望 只保留内容区域，去除边缘，得到一个“纯白色背景 + 内容居中”的干净矩形图
- 4、需要提取文字、水印原本的颜色回显（unet 1、需权重基于“是否为文字”，2、制作彩色 clean 图，然后重新训练）
- 5、添加自定义水印（opencv 即可实现）

## 实现方式

- 目标检测 yolov8-seg 返回四角坐标
- opencv 透视变换（不够智能）
- 图像分割网络 UNet 进行漂白去噪
