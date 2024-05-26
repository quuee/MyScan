import cv2
import numpy as np
from scan_calc import image_resize

def scan(img):
    screenCnt = None

    ratio = img.shape[0] / 500.0
    orig = img.copy()
    
    image2 = image_resize(orig,height = 500)
    
	# 预处理
    gray = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY) # 转换为灰度图像
    gray = cv2.GaussianBlur(gray,(5,5),0) # 高斯滤波
    edged = cv2.Canny(gray,60,180) # 边缘检测算法

    print("step 1: 边缘检测")
    # cv2.imshow("image",img)
    # cv2.imshow("edged",edged)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
	# 轮廓检测 找出所有轮廓点
    cnts = cv2.findContours(edged.copy(),
				       cv2.RETR_LIST,
					   cv2.CHAIN_APPROX_SIMPLE)
    
    # 根据轮廓面积排序
    cnts = sorted(cnts,key=cv2.contourArea,reverse=True)[:5]
    
	#遍历轮廓
    for c in cnts:
        # 计算轮廓近似
        # 周长（True表示合并）
        peri = cv2.arcLength(c,True) 
        # c表示输入的点集，epsilon表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数
        # True表示封闭的
        epsilon = 0.02*peri
        approx = cv2.approxPolyDP(c,epsilon,True)

        # print(len(approx))
        if len(approx) == 4:
            screenCnt = approx
            break
    
    # return orig,ratio,screenCnt
    print("step 2: 获取轮廓")
    cv2.drawContours(image2,[screenCnt],-1,(0,255,0),2)
    cv2.imshow("outline",image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()