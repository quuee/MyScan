import cv2
import numpy as np

# https://cloud.tencent.com/developer/article/1909041

'''
OpenCV将不同轮廓合并成一个轮廓
'''
def scan(img):
    # screenCnt = None
    # temp_contours_merge = []

    # ratio = img.shape[0] / 500.0
    # orig = img.copy()
    
    # image2 = image_resize(orig,height = 500)
    
	# # 预处理
    # gray = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY) # 图像二值化
    # blur = cv2.GaussianBlur(gray,(5,5),0) # 高斯滤波
    # edged = cv2.Canny(blur,75,200) # 边缘检测算法
    
	# # 轮廓检测，找出所有轮廓
    # cnts,hierarchy = cv2.findContours(edged.copy(),
	# 			       cv2.RETR_LIST,
	# 				   cv2.CHAIN_APPROX_SIMPLE)
    
	# #遍历轮廓
    # for c in cnts:
    #     temp_contours_merge.append(c)
            
    # contours_merge = np.vstack([temp_contours_merge[0],temp_contours_merge[1]])
    # for i in range(2,len(temp_contours_merge)):
    #     contours_merge = np.vstack([contours_merge,temp_contours_merge[i]])
    # rect2 = cv2.minAreaRect(contours_merge)
    # box2 = cv2.boxPoints(rect2)
    # screenCnt = box2

    # return orig,ratio,screenCnt


    split_res = img.copy()#显示每个轮廓结构
    merge_res = img.copy()#显示合并后轮廓结构

    # 记录开始时间
    start = cv2.getTickCount()
    hsvImg = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsvImg)
    ret, thresImg= cv2.threshold(S, 138, 255, cv2.THRESH_BINARY)
    cv2.imshow('threshold', thresImg)
    blurImg = cv2.medianBlur(thresImg,5)
    cv2.imshow('blur', blurImg)
    
    contours,hierarchy = cv2.findContours(blurImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    merge_list = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        split_res = cv2.drawContours(split_res,[box],0,(0,0,255),2)
        merge_list.append(cnt)
    cv2.imshow('split_res', split_res)
    cv2.imwrite('split_res.jpg', split_res)

    contours_merge = np.vstack([merge_list[0],merge_list[1]])
    for i in range(2, len(merge_list)):
        contours_merge = np.vstack([contours_merge,merge_list[i]])

    rect2 = cv2.minAreaRect(contours_merge)
    box2 = cv2.boxPoints(rect2)
    box2 = np.int0(box2)
    merge_res = cv2.drawContours(merge_res,[box2],0,(0,255,0),2)
    cv2.imshow('merge_res', merge_res)
    cv2.imwrite('merge_res.jpg', merge_res)

    # 记录结束时间    
    end = cv2.getTickCount()
    # 运行耗时
    use_time = (end - start) / cv2.getTickFrequency()
    print('use-time: %.3fs' % use_time)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print ('finish')
