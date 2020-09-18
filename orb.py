import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob

path = glob.glob("./removed/*.jpg")
path.sort()

for idx in range(len(path)):
    img1 = cv2.imread(path[idx])
    img1 = cv2.resize(img1, (400, 500))
    img2 = cv2.imread(path[idx+1])
    img2 = cv2.resize(img2, (400, 500))
    
    img_hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    img_hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        # threshold값 조정하여 노란색 범위 지정
    lower_yellow = (15, 45, 100)
    upper_yellow = (30, 240, 255)
    img_mask1 = cv2.inRange(img_hsv1, lower_yellow, upper_yellow)
    img_mask2 = cv2.inRange(img_hsv2, lower_yellow, upper_yellow)
    
        # img_color==>노란색만 추출 ==>img_result
    img_result1 = cv2.bitwise_and(img1, img1, mask=img_mask1)
    img_result2 = cv2.bitwise_and(img2, img2, mask=img_mask2)
    
    gray1 = cv2.cvtColor(img_result1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img_result2, cv2.COLOR_BGR2GRAY)
    
    orb = cv2.ORB_create(
        nfeatures=500,
        scaleFactor=1.5,
        nlevels=5,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=100,
    )
    kp1, des1 = orb.detectAndCompute(gray1,None)
    kp2, des2 = orb.detectAndCompute(gray2,None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:500],None,flags=2)
    title = path[idx][:-4]+'corners.jpg'
    cv2.imshow("orb_matching", img3)
    cv2.imwrite(title, img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#    img2 = cv2.drawKeypoints(img1, kp1, img2, (0,0,255), flags=0)
#    cv2.imshow("orb",img2)
#    title = image[:-4]+'corners.jpg'
#
##    cv2.imwrite(title, img2)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
