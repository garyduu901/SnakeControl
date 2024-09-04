import numpy as np
import cv2 as cv
import glob
import math

# Chessboard size
chessBoardSize = (9, 9)  # Size of the chessboard
frameSize = (1280, 720)   # Size of the images

# Termination criteria 
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points like (0,0,0), (1,0,0), ..., (8,7,0)
objp = np.zeros((chessBoardSize[0] * chessBoardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessBoardSize[0], 0:chessBoardSize[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all images
objPoints = []  # 3D point in real world space 
imgPoints = []  # 2D point in image space 

# Read images from the captured_images folder
images = glob.glob('calibration/captured_images/*.jpg')

# Debug: Check if images are found
if not images:
    print("No images found in the 'captured_images' folder. Check the folder path and file extensions.")
else:
    print("Found {} images.".format(len(images)))

for image in images:
    print("Processing:", image)
    img = cv.imread(image)
    if img is None:
        print("Failed to load image {}".format(image))
        continue
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # NEW: sharpening the image may gives more allowance for blurriness
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
    sharpened = cv.filter2D(gray, -1, kernel)

    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(sharpened, chessBoardSize, None)

    # If found, add object points and image points (after refining them)
    if ret:
        objPoints.append(objp)
        corners2 = cv.cornerSubPix(sharpened, corners, (11, 11), (-1, -1), criteria)
        imgPoints.append(corners2)

        # Draw and display the corners 
        cv.drawChessboardCorners(img, chessBoardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(3000)
    else:
        print("Chessboard corners not found in image {}".format(image))

cv.destroyAllWindows()

# Check if we have enough points for calibration
if len(objPoints) == 0 or len(imgPoints) == 0:
    print("No valid points found for calibration. Please check your images and chessboard pattern.")
else:
    # Perform calibration
    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, frameSize, None, None)
    print("Camera Calibrated:", ret)
    print("\nCamera Matrix:\n", cameraMatrix)
    # print("\nDistortion Parameters:\n", dist)
    # print("\nRotation Vectors:\n", rvecs)
    # print("\nTranslation Vectors:\n", tvecs)

fx = cameraMatrix[0][0]
fy = cameraMatrix[1][1]
W = 0.5

ref = cv.imread('pixel_to_dist/diff_z.jpg')
chessBoardSize = (9, 9)  # Size of the chessboard
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Get corners from calibrated images
gray = cv.cvtColor(ref, cv.COLOR_BGR2GRAY)
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
sharpened = cv.filter2D(gray, -1, kernel)
ret, corners = cv.findChessboardCorners(sharpened, chessBoardSize, None)
corners = cv.cornerSubPix(sharpened, corners, (11, 11), (-1, -1), criteria)

# Acquire side length
side_length = []

for i in range(chessBoardSize[1]):
    for j in range(chessBoardSize[0] - 1):
        curr_corner = corners[j+i*9][0]
        next_corner = corners[j+i*9+1][0]
        pixel_num = math.sqrt((curr_corner[0]-next_corner[0])**2 + (curr_corner[1]-next_corner[1])**2)
        side_length.append(pixel_num)
    

P = np.average(side_length)
D = W * fx / P
W_prime = D * P * 2 / fx

print(W_prime)