import cv2
import numpy as np


# 对图像进行统一的resize
def image_resize(image,width=None,height=None,inter=cv2.INTER_LINEAR):
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
    '''Rearrange coordinates to order:
      top-left, top-right, bottom-right, bottom-left'''
    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    s = pts.sum(axis=1)
    # Top-left point will have the smallest sum.
    rect[0] = pts[np.argmin(s)]
    # Bottom-right point will have the largest sum.
    rect[2] = pts[np.argmax(s)]
 
    diff = np.diff(pts, axis=1)
    # Top-right point will have the smallest difference.
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left will have the largest difference.
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    # return rect.astype('int').tolist()
    return rect.astype('int')


def find_dest(pts):
    (tl, tr, br, bl) = pts
    # Finding the maximum width.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
 
    # Finding the maximum height.
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # Final destination co-ordinates.
    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
 
    return order_points(destination_corners)

# 透视变换
# def four_point_transform(image,pts):
#     # 获取输入坐标
#     rect = order_points(pts)
#     (tl, tr, br, bl) = rect
#     widthA = np.sqrt((br[0]-bl[0])**2 + (br[1]-bl[1])**2)
#     widthB = np.sqrt((tr[0]-tl[0])**2 + (tr[1]-tl[1])**2)
#     maxWidth = max(int(widthA),int(widthB))
    
#     heightA = np.sqrt((tr[0]-br[0])**2 + (tr[1]-br[1])**2)
#     heightB = np.sqrt((tl[0]-bl[0])**2 + (tl[1]-bl[1])**2)
#     maxHeight = max(int(heightA),int(heightB))
    
#     dst = np.array([
#         [0,0],
#         [maxWidth-1,0],
#         [maxWidth-1,maxHeight-1],
#         [0,maxHeight-1]],
#         dtype='float32'
# 	)
    
#     M = cv2.getPerspectiveTransform(rect,dst)
#     #透视变换
#     warped = cv2.warpPerspective(image,M,(maxWidth,maxHeight))
#     return warped
    



    