import cv2
import numpy as np
import glob

path = glob.glob("./images/*.jpg")
res_path = './noise_removed'


from collections import defaultdict


def get_crosspt(x11,y11, x12,y12, x21,y21, x22,y22):
    if x12==x11 or x22==x21:
        print('delta x=0')
        return None
    m1 = (y12 - y11) / (x12 - x11)
    m2 = (y22 - y21) / (x22 - x21)
    if m1==m2:
        print('parallel')
        return None
    print(x11,y11, x12, y12, x21, y21, x22, y22, m1, m2)
    cx = (x11 * m1 - y11 - x21 * m2 + y21) / (m1 - m2)
    cy = m1 * (cx - x11) + y11

    return cx, cy
        
def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle
    to segment `k` angles inside `lines`.
    """
    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    
    return segmented
def intersection(line1, line2):
    #Finds the intersection of two lines given in Hesse normal form.
    #Returns closest integer pixel locations.
    #See https://stackoverflow.com/a/383527/5087436

    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]

    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


def segmented_intersections(lines):
    #Finds the intersections between groups of lines.

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2))
    return intersections
    ##########아래는 허프라인p 사용시#############

def find_intersection(line1, line2):
    # extract points
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]
    # compute determinant
    Px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4))/  \
        ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    Py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))/  \
        ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    return Px, Py

def segment_lines(lines, delta):
    h_lines = []
    v_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(x2-x1) < delta: # x-values are near; line is vertical
                v_lines.append(line)
            elif abs(y2-y1) < delta: # y-values are near; line is horizontal
                h_lines.append(line)
    return h_lines, v_lines

def cluster_points(points, nclusters):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)
    _, _, centers = cv2.kmeans(points, nclusters, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    return centers

########################################################

for image in path:
    print(image)
    res = res_path+'/'+image
    img_origin = cv2.imread(image)

    copy_img = img_origin.copy()
    img_color = cv2.resize(copy_img, (400, 500))

    height, width = img_color.shape[:2]
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
        # threshold값 조정하여 노란색 범위 지정
    lower_yellow = (15, 45, 100)
    upper_yellow = (30, 240, 255)
    img_mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
        # img_color==>노란색만 추출 ==>img_result
    img_result = cv2.bitwise_and(img_color, img_color, mask=img_mask)
    

    size = 15
    # generating the kernel
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    

    #노란길 필터링 한 후
    gray = cv2.cvtColor(img_result,cv2.COLOR_BGR2GRAY) #gray 변환
    ret, img_result = cv2.threshold(gray, 130, 255, 0) #노란색 부분 날리기 위함
    

    sharpening_1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img_result = cv2.filter2D(img_result, -1, sharpening_1)

    img_result = cv2.bilateralFilter(img_result,3,75,75) #엣지를 제외한 노이즈 제거
    img_result = cv2.filter2D(img_result, -1, kernel_motion_blur) #수평 블러링


    cv2.imshow('img_result0', img_result)
    cv2.waitKey(0)
    

    # applying the kernel to the input image
    output = cv2.filter2D(img_result, -1, kernel_motion_blur)

    cv2.imshow('Motion Blur', output)
    cv2.waitKey(0)

    #sharpening_1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    #img_result = cv2.filter2D(output, -1, sharpening_1)
    #cv2.imshow('Sharpening1', img_sharp)
    #cv2.waitKey(0)

    


    #ret, img_result = cv2.threshold(img_sharp, 70, 230, 0)


    img_result = cv2.bilateralFilter(img_result,9,75,75)
    cv2.imshow('Result', img_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




    #gray = cv2.cvtColor(img_result,cv2.COLOR_BGR2GRAY)
    #edges = cv2.Canny(img_result,50,150,apertureSize = 3)
    #img_sobel_x = cv2.Scharr(img_result, cv2.CV_64F, 1,0)
    img_sobel_x = cv2.Sobel(img_result, cv2.CV_64F, 1, 0, ksize=3)
    img_sobel_x = cv2.convertScaleAbs(img_sobel_x)

    #img_sobel_y = cv2.Scharr(img_result, cv2.CV_64F, 0,1)
    img_sobel_y = cv2.Sobel(img_result, cv2.CV_64F, 0, 1, ksize=3)
    img_sobel_y = cv2.convertScaleAbs(img_sobel_y)


    edges = cv2.addWeighted(img_sobel_x, 1, img_sobel_y, 1, 0);

    cv2.namedWindow("edge first", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("edge first", edges)
    cv2.waitKey(0)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for a in range(len(contours)):
        hull = contours[a]
        area = cv2.contourArea(hull)
        x, y, w, h = cv2.boundingRect(hull)
        rect_area = w * h
#        if rect_area >=1000:
        if rect_area < 3000:   
            cv2.rectangle(img_result, (x, y), (x + w, y + h), (0, 0, 0), cv2.FILLED)
    
    title = image[:-4]+'_removed_color.jpg'
    cv2.namedWindow("erase small boxes", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("erase small boxes", img_result)
    #cv2.imwrite(title, img_result)
    cv2.waitKey(0)
    
    
    title = image[:-4]+'_removed_edge.jpg'

    edge3 = cv2.Canny(img_result,100,150,apertureSize = 3)
    cv2.namedWindow("final edge", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("final edge", edge3)
    #cv2.imwrite(title, edge3)
    cv2.waitKey(0)
    img2= None
    try:
        corners = cv2.goodFeaturesToTrack(edge3,30,0.01,20)
        corners = np.int0(corners)
        for i in corners:
            x,y = i.ravel()
            cv2.circle(img_result,(x,y),3,255,-1)
#        orb = cv2.ORB_create(
#            nfeatures=30,
#            scaleFactor=1.5,
#            nlevels=8,
#            edgeThreshold=31,
#            firstLevel=0,
#            WTA_K=2,
#            scoreType=cv2.ORB_HARRIS_SCORE,
#            patchSize=31,
#            fastThreshold=20,
#        )
#        kp1, des1 = orb.detectAndCompute(edge3,None)
#
#
#        img2 = cv2.drawKeypoints(img_color, kp1, img2, (0,0,255), flags=0)
        
        
        title = image[:-4]+'corners.jpg'
        cv2.namedWindow("corner", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("corner", img_result)
#        cv2.imwrite(title, img_color)
        cv2.waitKey(0)
    except:
        pass
    
    try:
            lines = cv2.HoughLines(edge3,1,np.pi/90,80)
            segmented = segment_by_angle_kmeans(lines)
            intersections = segmented_intersections(segmented)
            line_equa=[]
            
            for i in range(len(lines)):
                for rho, theta in lines[i]:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0+1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 -1000*(a))

                    cv2.line(img_result,(x1,y1),(x2,y2),(0,0,255),2)
                    line_equa.append([x1,y1,x2,y2])
            #line equation x1,y1, x2,y2
            
            hor_line= []
            ver_line = []
            
            for line in line_equa:
                if line[0] != line[2]:
                    if abs(line[3]-line[1]/line[2]-line[0]) >= 1:
                        ver_line.append(line)
                elif line[0] == line[2]:
                    ver_line.append(line)
                elif line[1] != line[3]:
                    if abs(line[3]-line[1]/line[2]-line[0]) < 1:
                        hor_line.append(line)
                elif line[1] == line[3]:
                    hor_line.append(line)
            
            print("ver_lines:", ver_line)
            vx, vy  = get_crosspt(ver_line[0][0],ver_line[0][1], ver_line[0][2],ver_line[0][3], ver_line[1][0],ver_line[1][1], ver_line[1][2],ver_line[1][3])
            print("vx,vy:", vx,vy)
            
            width = abs(ver_line[1][0] - ver_line[0][0])
            cv2.line(img_color,(vx,vy),(ver_line[0][0]-width,ver_line[0][1]),(0,0,255),2)
            cv2.imshow('v_pt',img_color)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            title = image[:-4]+'houghlines.jpg'
            cv2.namedWindow("hough", cv2.WINDOW_AUTOSIZE)
            cv2.imshow('hough',img_result)
            #cv2.imwrite(title, img_result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            
            for point in intersections:
                for cx, cy in point:
                    cv2.circle(img_result, (cx, cy), radius=4, color=[255,0,255], thickness=-1) # -1: filled circle
            
            title = image[:-4]+'intersections.jpg'
            cv2.namedWindow("intersections", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("intersections", img_result)
            #cv2.imwrite(title, img_result)
            cv2.waitKey(0)
#            cv2.imwrite('corners.png', img_result)
    except:
        pass
        
        
    delta = 30
    
#    try:
#        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, maxLineGap=20, minLineLength=50)
#        h_lines, v_lines = segment_lines(lines, delta)
#        print(h_lines);print(v_lines)
#        # draw the segmented lines
#        houghimg = img_result.copy()
#        for line in h_lines:
#            for x1, y1, x2, y2 in line:
#                color = [0,0,255] # color hoz lines red
#                cv2.line(houghimg, (x1, y1), (x2, y2), color=color, thickness=1)
#        for line in v_lines:
#            for x1, y1, x2, y2 in line:
#                color = [255,0,0] # color vert lines blue
#                cv2.line(houghimg, (x1, y1), (x2, y2), color=color, thickness=1)
#
#        cv2.imshow("Segmented Hough Lines", houghimg)
#        cv2.waitKey(0)
#        cv2.imwrite('hough.png', houghimg)
#
#        # find the line intersection points
#        Px = []
#        Py = []
#        for h_line in h_lines:
#            for v_line in v_lines:
#                px, py = find_intersection(h_line, v_line)
#                Px.append(px)
#                Py.append(py)
#
#        # draw the intersection points
#        intersectsimg = img_result.copy()
#        for cx, cy in zip(Px, Py):
#            cx = np.round(cx).astype(int)
#            cy = np.round(cy).astype(int)
#            color = np.random.randint(0,255,3).tolist() # random colors
#            cv2.circle(intersectsimg, (cx, cy), radius=2, color=color, thickness=-1) # -1: filled circle
#
#        cv2.imshow("Intersections", intersectsimg)
#        cv2.waitKey(0)
#        cv2.imwrite('intersections.png', intersectsimg)
#
#        # use clustering to find the centers of the data clusters
#        P = np.float32(np.column_stack((Px, Py)))
#        nclusters = 4
#        centers = cluster_points(P, nclusters)
#        print(centers)
#
#        # draw the center of the clusters
#        for cx, cy in centers:
#               cx = np.round(cx).astype(int)
#               cy = np.round(cy).astype(int)
#               cv2.circle(img_result, (cx, cy), radius=4, color=[0,0,255], thickness=-1) # -1: filled circle
#
#        cv2.imshow("Center of intersection clusters", img_result)
#        cv2.waitKey(0)
#        cv2.imwrite('corners.png', img_result)
#    except:
#        pass

