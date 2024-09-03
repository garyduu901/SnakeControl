import numpy as np
import cv2 as cv
import math

# Use calibrated image as reference
caliResult = cv.imread('caliResult2.jpg')
chessBoardSize = (9, 9)  # Size of the chessboard
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Get corners from calibrated images
ret, corners = cv.findChessboardCorners(caliResult, chessBoardSize, None)
gray = cv.cvtColor(caliResult, cv.COLOR_BGR2GRAY)
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
sharpened = cv.filter2D(gray, -1, kernel)
corners = cv.cornerSubPix(sharpened, corners, (11, 11), (-1, -1), criteria)
cv.drawChessboardCorners(caliResult, chessBoardSize, corners, ret)
cv.imshow('img', caliResult)
cv.waitKey(3000)

# Acquire side length
side_length_x = []
side_length_y = []

for i in range(chessBoardSize[1]):
    row_side_length = []
    for j in range(chessBoardSize[0] - 1):
        curr_corner = corners[j+i*9][0]
        next_corner = corners[j+i*9+1][0]
        pixel_num = math.sqrt((curr_corner[0]-next_corner[0])**2 + (curr_corner[1]-next_corner[1])**2)
        row_side_length.append(pixel_num)
    side_length_x.append(row_side_length)

for i in range(chessBoardSize[1] - 1):
    row_side_length = []
    for j in range(chessBoardSize[0]):
        curr_corner = corners[j+i*9][0]
        next_corner = corners[j+(i+1)*9][0]
        pixel_num = math.sqrt((curr_corner[0]-next_corner[0])**2 + (curr_corner[1]-next_corner[1])**2)
        row_side_length.append(pixel_num)
    side_length_y.append(row_side_length)


side_length_x = np.array(side_length_x)
side_length_y = np.array(side_length_y)

avg_side_x = sum(np.mean(side_length_x, axis=1))/chessBoardSize[1]
print(avg_side_x)

avg_side_y = sum(np.mean(side_length_y, axis=1))/chessBoardSize[1]
print(avg_side_y)

Fx = 1.73836169e+04 # Focal length (calculated constant)
Fy = 1.73836169e+04 # Focal length (calculated constant)


# For objects have different z values
W_2x = 15            # Object physical length (mm)
P_2x = pixel_num     # Measured pixel number from image
D_2x = W_2x * Fx / P_2x # Distance between camera and object

W_2y = 15            # Object physical length (mm)
P_2y = pixel_num     # Measured pixel number from image
D_2y = W_2y * Fy / P_2y # Distance between camera and object

# Measure unknown distance (movements)
P_3x = pixel_num     # Measured pixel number from image
P_3y = pixel_num     # Measured pixel number from image

W_movx = D_2x*P_3x/Fx 
W_movy = D_2y*P_3y/Fy 

