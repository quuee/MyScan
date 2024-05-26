import cv2
import numpy as np
from scan_calc import order_points,find_dest


def scan(orig_img):
    # Resize image to workable size
    dim_limit = 1080
    max_dim = max(orig_img.shape)
    print(f'max_dim:{max_dim}')
    resize_scale = 0.6
    if max_dim > dim_limit:
        resize_scale = dim_limit / max_dim
        print(f'resize_scale:{resize_scale}')
        resize_img = cv2.resize(orig_img, None, fx=resize_scale, fy=resize_scale)
    # Create a copy of resized original image for later use
    resize_img2 = resize_img.copy()
    # Repeated Closing operation to remove text from the document.
    kernel = np.ones((5, 5), np.uint8)
    resize_img = cv2.morphologyEx(resize_img, cv2.MORPH_CLOSE, kernel, iterations=3)
    # GrabCut
    mask = np.zeros(resize_img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (20, 20, resize_img.shape[1] - 20, resize_img.shape[0] - 20)
    cv2.grabCut(resize_img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    resize_img = resize_img * mask2[:, :, np.newaxis]
 
    gray = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)

    ## GaussianBlur Canny 参数不同，扫描结果也不同
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    # Edge Detection.
    canny = cv2.Canny(gray, 30, 120)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
 
    # Finding contours for the detected edges.
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Keeping only the largest detected contour.
    page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
 
    # Detecting Edges through Contour approximation.
    # Loop over the contours.
    if len(page) == 0:
        return resize_img2
    for c in page:
        # Approximate the contour.
        epsilon = 0.02 * cv2.arcLength(c, True)
        corners = cv2.approxPolyDP(c, epsilon, True)
        # If our approximated contour has four points.
        if len(corners) == 4:
            break
    # Sorting the corners and converting them to desired shape.
    print(f'corners before sort:{corners}')
    corners = sorted(np.concatenate(corners).tolist())
    print(f'corners sorted:{corners}')
    # For 4 corner points being detected.
    corners = order_points(corners)
    print(f'corners order_points:{corners}')
 
    destination_corners = find_dest(corners)
    print(f'destination_corners:{destination_corners}')
 
    # 放大原来的图片
    corners = np.float32(corners.reshape(4,2) / resize_scale)
    destination_corners = np.float32(destination_corners.reshape(4,2) / resize_scale)
    # h, w = resize_img2.shape[:2]
    # Getting the homography.
    M = cv2.getPerspectiveTransform(corners, destination_corners)
    # Perspective transform using homography.
    final = cv2.warpPerspective(orig_img, M, (int(destination_corners[2][0]), int(destination_corners[2][1])),
                                flags=cv2.INTER_LINEAR)

    return final

if __name__ == "__main__":
    img = cv2.imread("./test_image/test2.jpg")
    final_img = scan(img)
    (h,w)=final_img.shape[:2]
    cv2.namedWindow("final",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("final",w,h)
    cv2.imshow("final",final_img)
    cv2.waitKey(0) # 等待键盘操作
    cv2.destroyAllWindows() # 销毁所有窗口