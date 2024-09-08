'''
Reference:
Color masking: https://stackoverflow.com/questions/30331944/finding-red-color-in-image-using-python-opencv
Find largest: https://stackoverflow.com/questions/44588279/find-and-draw-the-largest-contour-in-opencv-on-a-specific-color-python
Circle fitting : https://stackoverflow.com/questions/55621959/opencv-fitting-a-single-circle-to-an-image-in-python
Lucas-Kanade optical flow: https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
Bilateral Filtering: https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed
'''

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

'''
* apply_mask: apply a mask to the first frame of the video, 
            returns the coordinate of the center of the red dot
* image: the first frame of the video
* x, y: coordinates of the center of the red dot
'''
def apply_mask(image):
    result = image.copy()

    # Filter
    image = cv.bilateralFilter(image, 8, 75, 75)
    cv.imshow('filt', image)
    cv.waitKey()


    # Upper + lower range mask
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    lower1 = np.array([155,50,50])
    upper1 = np.array([179,255,255])

    lower2 = np.array([0,50,50])
    upper2 = np.array([10,255,255])

    mask1 = cv.inRange(image, lower1, upper1)
    mask2 = cv.inRange(image, lower2, upper2)
    mask = mask1 + mask2
    result = cv.bitwise_and(result, result, mask=mask)

    cv.imshow('mask', mask)
    # cv.imshow('result', result)
    cv.waitKey()

    # Find contours
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        # # Filtering small contours
        # for cnts in contours:
        #     area = cv.contourArea(cnts)
        #     if area > 50:
        #         # Draw all found contours in blue
        #         cv.drawContours(result, cnts, -1, (255,255,255), 3)
        
        # Find the biggest countour by the area
        c = max(contours, key = cv.contourArea)
        cv.drawContours(result, c, -1, 255, 3)
        
        # Draw the fitted ellipse of the biggest contour (c) in green
        # ellipse: center_coordinates, axesLength, angle
        # ellipse = cv.fitEllipse(c)
        # cv.ellipse(result, ellipse, (0, 255, 0), 2)
        # print("ellipse loc: x={x}, y={y}".format(x = ellipse[0][0], y = ellipse[0][1]))

        # Draw the fitted circle of the biggest contour (c) in red
        # circle: center_coordinates, radius, 
        (x,y),radius = cv.minEnclosingCircle(c)
        center = (int(x),int(y))
        radius = int(radius)
        cv.circle(result, center, radius, (0,0,255), 2)
        print("circle loc: x={x}, y={y}".format(x = x, y = y))

    # Show the images
    cv.imshow("Result", result)
    cv.waitKey(0)
    return x, y

if __name__=="__main__":
    cap = cv.VideoCapture('dot_track/test2.mp4')


    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15, 15),
                    maxLevel = 2,
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    # Take first frame
    ret, init_frame = cap.read()
    old_gray = cv.cvtColor(init_frame, cv.COLOR_BGR2GRAY)
    x, y = apply_mask(init_frame)

    p0 = np.array([[[x, y]]], dtype='float32')

    # Create a mask image for drawing purposes
    mask = np.zeros_like(init_frame)
    count = 0
    img_array = []
    x_cord = []
    y_cord = []
    dist = 0

    while(1):
        ret, frame = cap.read()
        if not ret:
            print('No frames grabbed!')
            break

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            x_cord.append(int(a))
            y_cord.append(int(b))
            dist += ((a-c)**2+(b-d)**2)**2
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), [0, 255, 0], 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, [0, 255, 0], -1)
        img = cv.add(frame, mask)
        img_array.append(img)
        height, width, layers = img.shape
        size = (width,height)

        cv.imshow('frame', img)
        
        
        count += 1
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    out = cv.VideoWriter('/dot_track/track_output.mp4',cv.VideoWriter_fourcc(*'DIVX'), 30, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    x_cord = np.array(x_cord)
    y_cord = np.array(y_cord)

    print("Distance traveled in pixels: " + str(dist))
    plt.plot(x_cord, y_cord)
    plt.gca().invert_yaxis()
    plt.show()

    cv.destroyAllWindows()
