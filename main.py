from scan import detection1,detection2,four_point_transform
import cv2
import datetime

if __name__ == '__main__':
    img = cv2.imread('./test_image/test3.png')
    # detection(img)
    orig,ratio,screen_cnt = detection2(img)
    warped = four_point_transform(orig,screen_cnt.reshape(4,2)*ratio)

	# 二值化
    gray = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
    # 调整输出图像
    thresh = cv2.threshold(gray,120,250,cv2.THRESH_BINARY)[1]
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_name = "./result_image/scan_"+time+'.jpg'
    print(result_name)
    cv2.imwrite(result_name,thresh)
    cv2.imshow('scanned',thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()