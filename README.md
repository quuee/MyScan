## 依赖

pip install opencv pytesseract ultralytics

额外安装 sudo apt install tesseract-ocr 才能使用 pytesseract（目前不会用，效果没有）

## 入口

scan_document.py 单张图片
camera_scan_document.py 连接手机相机

## TODO

1、目标检测精度勉强够用
2、背景漂白时，总把文字、线框给腐蚀掉（有孔洞），不够清晰，需要增强清晰度
3、背景漂白时，假如像 0060.jpg 这张图，会将顶部轮廓识别成边缘，实际应该生成正四方白底，没有顶部这条边缘线
4、透视变换不够智能，不能将目标区域文字投影到正中间
5、需要提取文字、水印原本的颜色回显
