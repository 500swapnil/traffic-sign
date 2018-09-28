import cv2
import numpy as np
import time
import os
import tensorflow as tf
import pandas as pd
import sermanet as classifier
import lenet as detector

candidates = []
windows = []
min_area = 600
max_area = 12000

# Define the codec and create VideoWriter object
frame_width = 1280
frame_height = 720
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Classification Dictionary
sign = {0:'Advance Left',
        1:'Advance Right',
        2:'Breaker',
        3:'Compulsory Ahead',
        4:'Compulsory Left',
        5:'Compulsory Right',
        6:'Stop',
        7:'Traffic Signal',
        }

def morphOps(mask):
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (2,2))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (10,10))
    
    morph = cv2.erode(mask, erode_kernel, iterations = 3)
    morph = cv2.dilate(morph, dilate_kernel, iterations = 2)
    return morph


def DetectContours(frame,bin_image,color):
    _, contours, _ = cv2.findContours( bin_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset = (0,0))
        
    shape_detected = False
    for contour in contours:
        approx = cv2.approxPolyDP(contour,0.2*cv2.arcLength(contour,True),True)
        area = cv2.contourArea(contour)
        # Check whether area of contour is within range
        if area > max_area or area < min_area:
            continue
        
        ## For red signs, may be of 2 shapes - Triangle or Octogon (May be detected as circle)
        if color == 'red':
            if len(approx) == 3:
                shape_detected = True
                # get the bounding rect
                x, y, w, h = cv2.boundingRect(contour)
                
            if len(approx) in [2,8]:
                x, y, w, h = cv2.boundingRect(contour)
                if abs(1 - w/h) <= 0.1 and abs(1 - np.pi*(w/2)**2):
                    shape_detected = True
        
        ## For blue signs, only 1 possible shape - Circle (May be detected as Octogon)
        if color == 'blue':
            if len(approx) in [2,8]:
                x, y, w, h = cv2.boundingRect(contour)
                if abs(1 - w/h) <= 0.1 and abs(1 - np.pi*(w/2)**2):
                    shape_detected = True
        
        if shape_detected == True:
            w *= 0.9
            h *= 0.9
            # x += 0.1*w
            # y += 0.1*h
            x,y,w,h = int(x),int(y),int(w),int(h)
            candidates.append((x,y,w,h))


def ColourSegment(frame):
        bilateral_filtered_frame = cv2.bilateralFilter(frame, 1, 40, 40)

        # Convert BGR to HSV
        hsv = cv2.cvtColor(bilateral_filtered_frame, cv2.COLOR_BGR2HSV)

        #### For RED Signs ####
        # define range of red color in HSV
        lower_red_1 = np.array([0,38,60])
        upper_red_1 = np.array([8,255,255])

        lower_red_2 = np.array([160,40,0])
        upper_red_2 = np.array([179,200,255])

        # Threshold the HSV image to get only red colors
        mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

        red_mask = np.bitwise_or(mask1, mask2)
        
        red_morph = morphOps(red_mask)
        _, red_bin = cv2.threshold(red_morph ,0, 255, cv2.THRESH_BINARY_INV)

        # find red contours and add to candidates list
        DetectContours(frame,red_bin,'red')

        ################################

        #### For BLUE Signs ####
        # define range of blue color in HSV
        lower_blue = np.array([84,100,20])
        upper_blue = np.array([124,255,255])

        # Threshold the HSV image to get only blue colors
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        blue_morph = morphOps(blue_mask)
        _, blue_bin = cv2.threshold(blue_morph ,0, 255, cv2.THRESH_BINARY_INV)
        
        # find blue contours and add to candidates list
        DetectContours(frame,blue_bin,'blue')   

        ################################

def getArea(item):
    return item[2]*item[3]


directory = 'videos/'
for video in os.listdir(directory):
    k = 0
    print(video)
    cap = cv2.VideoCapture(directory+video)
    out = cv2.VideoWriter('outputs/'+video[:-4]+'.avi', fourcc, 30.0, (frame_width,frame_height))
    start = time.time()

    while cap.isOpened(): #and k < 60:
        ret, frame = cap.read()
        if ret==True:
            

            # We resize the frame to our output frame width and height irrespective of input
            resized_frame = cv2.resize(frame, (frame_width, frame_height))

            ColourSegment(resized_frame)

            # Check Overlapping segments here

            candidates.sort(key=getArea, reverse = True)

            for i in candidates:

                if i[2]*i[3]>100:
                    flag = 1
                    for j in windows:
                        if( (i[0]>=j[0]) and (i[1]>=j[1]) and (i[0]+i[2]<=j[0]+j[2]) and (i[1]+i[3]<=j[1]+j[3]) ):
                            flag = 0
                            break
                    if flag == 1:
                        windows.append(i)

            
            signBoards = []
            labels = []
            checkSign = []
            for window in windows:
                x, y, w, h = window
                signBoards.append(cv2.resize(np.array(resized_frame[y:y+h,x:x+w]),(32,32)))

            if len(signBoards) > 0:
                signBoards = np.array(signBoards)
                # checkSign = detector.predict(signBoards)
                checkSign = detector.confidence(signBoards)
                pred_no = classifier.predict(signBoards)   
                labels = np.vectorize(sign.get)(pred_no)
            
            if len(checkSign) == len(windows):
                for i in range(len(checkSign)):
                    # draw a green rectangle to visualize the candidates
                    
                    if checkSign[i][1]<6:
                        continue
                    x, y, w, h =  windows[i]
                    cv2.putText(resized_frame, str(labels[i]), (x, y - 20),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.rectangle(resized_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # cv2.imshow('Candidates',resized_frame)
            candidates = []
            windows = []
            
            # write the output resized_frame
            out.write(resized_frame)

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
            k += 1
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    avg_FPS = k/(time.time() - start)
    print("FPS:",avg_FPS)
cv2.destroyAllWindows()
