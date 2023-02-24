import cv2
import numpy as np
from scan import resize

def detection(img):
    screenCnt = None
    temp_contours_merge = []

    ratio = img.shape[0] / 500.0
    orig = img.copy()
    
    image2 = resize(orig,height = 500)
    
	# 预处理
    gray = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY) # 图像二值化
    blur = cv2.GaussianBlur(gray,(5,5),0) # 高斯滤波
    edged = cv2.Canny(blur,60,150) # 边缘检测

    # 找出所有轮廓
    cnts,hierarchy=cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        # x,y,w,h=cv2.boundingRect(c)
        # cv2.rectangle(image2,(x,y),(x+w,y+h),(255,0,0),2) #blue
            
        temp_contours_merge.append(c)

    contours_merge = np.vstack([temp_contours_merge[0],temp_contours_merge[1]])
    for i in range(2,len(temp_contours_merge)):
        contours_merge = np.vstack([contours_merge,temp_contours_merge[i]])

    rect2 = cv2.minAreaRect(contours_merge)
    box2 = cv2.boxPoints(rect2)
    box2 = np.int0(box2)
    merge = cv2.drawContours(image2,[box2],-1,(0,255,0),2)

    cv2.imshow("merge",merge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


if __name__ == '__main__':
    img = cv2.imread('test2.jpg')
    detection(img)

