import cv2
import numpy as np


def detection1(img):
    screenCnt = None

    ratio = img.shape[0] / 500.0
    orig = img.copy()
    
    image2 = resize(orig,height = 500)
    
	# 预处理
    gray = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY) # 图像二值化
    blur = cv2.GaussianBlur(gray,(5,5),0) # 高斯滤波
    edged = cv2.Canny(blur,75,200) # 边缘检测算法
    
	# 轮廓检测 找出所有轮廓点
    cnts = cv2.findContours(edged.copy(),
				       cv2.RETR_LIST,
					   cv2.CHAIN_APPROX_SIMPLE)[0]
    # 根据轮廓面积排序
    cnts = sorted(cnts,key=cv2.contourArea,reverse=True)
    
	#遍历轮廓
    for c in cnts:
        # 周长（True表示合并）
        peri = cv2.arcLength(c,True) 
        # c表示输入的点集，epsilon表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数
        epsilon = 0.02*peri
        approx = cv2.approxPolyDP(c,epsilon,True) # 近似轮廓

        # print(len(approx))
        if len(approx) == 4:
            screenCnt = approx
            break
    
    return orig,ratio,screenCnt

# 如果找不到外围最大的轮廓，直接将整张图片当成最大的一个轮廓
# 或者将小轮廓合并成一个大轮廓
def detection2(img):
    screenCnt = None
    temp_contours_merge = []

    ratio = img.shape[0] / 500.0
    orig = img.copy()
    
    image2 = resize(orig,height = 500)
    
	# 预处理
    gray = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY) # 图像二值化
    blur = cv2.GaussianBlur(gray,(5,5),0) # 高斯滤波
    edged = cv2.Canny(blur,75,200) # 边缘检测算法
    
	# 轮廓检测，找出所有轮廓
    cnts,hierarchy = cv2.findContours(edged.copy(),
				       cv2.RETR_LIST,
					   cv2.CHAIN_APPROX_SIMPLE)
    
	#遍历轮廓
    for c in cnts:
        temp_contours_merge.append(c)
            
    contours_merge = np.vstack([temp_contours_merge[0],temp_contours_merge[1]])
    for i in range(2,len(temp_contours_merge)):
        contours_merge = np.vstack([contours_merge,temp_contours_merge[i]])
    rect2 = cv2.minAreaRect(contours_merge)
    box2 = cv2.boxPoints(rect2)
    screenCnt = box2

    return orig,ratio,screenCnt

# 还有矩形轮廓和外界园


# 对图像进行统一的resize
def resize(image,width=None,height=None,inter=cv2.INTER_LINEAR):
    dim = None
    (h,w)=image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w*r),height)
    else:
        r = width / float(w)
        dim = (width,int(h*r))
	
    resized = cv2.resize(image,dim,interpolation=inter)
    return resized
    
def order_points(pts):
    rect = np.zeros((4, 2), dtype='float32')
    # 按顺序找到对应的坐标0123 分别是左上，右上，右下，左下
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts,axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect
    
# 透视变换
def four_point_transform(image,pts):
    # 获取输入坐标
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt((br[0]-bl[0])**2 + (br[1]-bl[1])**2)
    widthB = np.sqrt((tr[0]-tl[0])**2 + (tr[1]-tl[1])**2)
    maxWidth = max(int(widthA),int(widthB))
    
    heightA = np.sqrt((tr[0]-br[0])**2 + (tr[1]-br[1])**2)
    heightB = np.sqrt((tl[0]-bl[0])**2 + (tl[1]-bl[1])**2)
    maxHeight = max(int(heightA),int(heightB))
    
    dst = np.array([
        [0,0],
        [maxWidth-1,0],
        [maxWidth-1,maxHeight-1],
        [0,maxHeight-1]],
        dtype='float32'
	)
    
    M = cv2.getPerspectiveTransform(rect,dst)
    #透视变换
    warped = cv2.warpPerspective(image,M,(maxWidth,maxHeight))
    return warped
    



    