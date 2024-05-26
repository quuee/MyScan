import cv2
import numpy as np
import imutils

# https://blog.csdn.net/qq_34406071/article/details/109129495

'''
图像预处理

return:
  image 原始图像
  gray 灰度图
  edged 边缘图
'''
def getOutline(input_path):
    image = cv2.imread(input_path)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsvImg = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsvImg)

    # blur = cv2.GaussianBlur(gray,(5,5),0,0)
    # blur = cv2.bilateralFilter(gray,7,70,70)
    ret, thresImg= cv2.threshold(S, 138, 255, cv2.THRESH_BINARY)
    blur = cv2.medianBlur(thresImg,5)
    edged = cv2.Canny(blur,30,120)
    return image,hsvImg,edged


def get_cnt(edged):
    contours,hierarchy = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    return contours


if __name__ == "__main__":
    image,gray,edged = getOutline("./test_image/test1.jpg")
    cv2.namedWindow("scanned",cv2.WINDOW_FREERATIO)
    # cv2.resizeWindow("scanned",1920,1080)
    cv2.imshow("scanned",edged)

    cv2.namedWindow("gray",cv2.WINDOW_FREERATIO)
    # cv2.resizeWindow("gray",1920,1080)
    cv2.imshow("gray",gray)
    cv2.waitKey(0) # 等待键盘操作
    cv2.destroyAllWindows() # 销毁所有窗口
    # cnts = get_cnt(edged)
    # print(cnts)


