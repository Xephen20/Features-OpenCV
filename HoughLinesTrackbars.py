import cv2
import numpy as np
import math

def nothing(x):
    pass

def calculateDistance(x1,y1,x2,y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

def midpoint(x1, y1, x2, y2):
    return ((x1 + x2)/2, (y1 + y2)/2)

def angle_of_line(x1, y1, x2, y2):
    angle = math.degrees(math.atan2(-(y2-y1), x2-x1))
    if angle <0:
        angle = angle + 180
    return angle

cv2.namedWindow("HSVTrackbars")
cv2.createTrackbar("L - H", "HSVTrackbars", 153, 179, nothing)
cv2.createTrackbar("L - S", "HSVTrackbars", 96, 255, nothing)
cv2.createTrackbar("L - V", "HSVTrackbars", 175, 255, nothing)
cv2.createTrackbar("U - H", "HSVTrackbars", 156, 179, nothing)
cv2.createTrackbar("U - S", "HSVTrackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "HSVTrackbars", 255, 255, nothing)

cv2.namedWindow("CannyTrackbars")
cv2.createTrackbar("Thresh1", "CannyTrackbars", 190, 1000, nothing)
cv2.createTrackbar("Thresh2", "CannyTrackbars", 135, 1000, nothing)

cv2.namedWindow("HoughLinesTrackbars")
cv2.createTrackbar("hThresh", "HoughLinesTrackbars", 19, 400, nothing)
cv2.createTrackbar("hMinLine", "HoughLinesTrackbars", 20, 100, nothing)
cv2.createTrackbar("hMaxGap", "HoughLinesTrackbars", 1, 100, nothing)

cv2.namedWindow("LinesTrackbars")
cv2.createTrackbar("minLen", "LinesTrackbars", 1, 1000, nothing)
cv2.createTrackbar("maxLen", "LinesTrackbars", 1000, 1000, nothing)
cv2.createTrackbar("minAngle", "LinesTrackbars", 0, 360, nothing)
cv2.createTrackbar("maxAngle", "LinesTrackbars", 360, 360, nothing)

cv2.namedWindow("VisibilityTrackbars")
cv2.createTrackbar("showLen", "VisibilityTrackbars", 1, 2, nothing)
cv2.createTrackbar("showAngle", "VisibilityTrackbars", 1, 2, nothing)

kernel = np.ones((3,3), np.uint8) 

while True:

    l_h = cv2.getTrackbarPos("L - H", "HSVTrackbars")
    l_s = cv2.getTrackbarPos("L - S", "HSVTrackbars")
    l_v = cv2.getTrackbarPos("L - V", "HSVTrackbars")
    u_h = cv2.getTrackbarPos("U - H", "HSVTrackbars")
    u_s = cv2.getTrackbarPos("U - S", "HSVTrackbars")
    u_v = cv2.getTrackbarPos("U - V", "HSVTrackbars")
    
    Thresh1 = cv2.getTrackbarPos("Thresh1", "CannyTrackbars")
    Thresh2 = cv2.getTrackbarPos("Thresh2", "CannyTrackbars")
    
    hThresh = cv2.getTrackbarPos("hThresh", "HoughLinesTrackbars")
    hMinLine = cv2.getTrackbarPos("hMinLine", "HoughLinesTrackbars")
    hMaxGap = cv2.getTrackbarPos("hMaxGap", "HoughLinesTrackbars")
    
    minLen = cv2.getTrackbarPos("minLen", "LinesTrackbars")
    maxLen = cv2.getTrackbarPos("maxLen", "LinesTrackbars")
    
    minAngle = cv2.getTrackbarPos("minAngle", "LinesTrackbars")
    maxAngle = cv2.getTrackbarPos("maxAngle", "LinesTrackbars")
    
    showLen = cv2.getTrackbarPos("showLen", "VisibilityTrackbars")
    showAngle = cv2.getTrackbarPos("showAngle", "VisibilityTrackbars")
    
    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    
    images = [
        cv2.imread('box.jpeg', cv2.IMREAD_COLOR)
        #can be multiple
    ]
    black = np.zeros((50,440))

    for i, img in enumerate(images):
        black = np.zeros((50,440))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        result = img
        #use mask
        mask = cv2.dilate(mask, kernel, iterations=1)

        lines = cv2.Canny(result, threshold1=Thresh1, threshold2=Thresh2)
        img_dilation = cv2.dilate(lines, kernel, iterations=1) 
        lines[mask>0]=0
        
        cv2.putText(img,'Min Len: %s'%minLen, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, 2)
        cv2.putText(img,'Max Len: %s'%maxLen, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, 2)
        cv2.putText(img,'Min Angle: %s'%minAngle, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, 2)
        cv2.putText(img,'Max Angle: %s'%maxAngle, (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, 2)
        
        HoughLines = cv2.HoughLinesP(img_dilation, 1, np.pi/180, threshold = hThresh, minLineLength = hMinLine, maxLineGap = hMaxGap)
        sum_lines = []
        if HoughLines is not None:
            for line in HoughLines:
                coords = line[0]
                length = calculateDistance(coords[0], coords[1], coords[2], coords[3])
                angle = angle_of_line(coords[0], coords[1], coords[2], coords[3])
                x,y = midpoint(coords[0], coords[1], coords[2], coords[3])
                if length > minLen and length < maxLen:
                    if angle > minAngle and angle < maxAngle:
                        cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [0,0,255], 3)
                        if showLen == 1 or showLen == 2:
                            cv2.putText(img,'%s'%int(length), (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, 2)
                        if showAngle == 1 or showAngle == 2:  
                            cv2.putText(img,'%s'%int(angle), (int(x)+20,int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, 2)
                            sum_lines.append(angle)
     
        cv2.putText(img,'Found: %s'%len(sum_lines)+" lines", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, 2)
      
        #cv2.imshow(f"Mask{i}", mask)
        cv2.imshow(f"Original{i}", img)
        #cv2.imshow(f"Result{i}", result)
        #cv2.imshow(f"Lines{i}", lines)
        cv2.imshow(f"Dilation{i}", img_dilation)
        #cv2.imshow(f"blacknWhite{i}", black)
        cv2.waitKey(1)
