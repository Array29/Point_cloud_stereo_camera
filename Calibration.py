import cv2
import numpy as np
import cv2 as cv
from Camera import Camera

camera = Camera()
color, depth = camera.start_stream()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
objpoints = []
imgpoints = []

photos = camera.calibration_photos(color)

for i, photo in enumerate(photos):
    gray = cv.cvtColor(photo, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)
    print(f"Image {i}: Chessboard corners found: {ret}")

    if ret:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        key = cv2.waitKey(1)
        done = False
        while not done:
            key = cv2.waitKey(1)
            # Draw and display the corners
            cv.drawChessboardCorners(photo, (7, 6), corners2, ret)
            cv.imshow('img', photo)
            if key == 27:
                cv.destroyAllWindows()
                ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
                print(mtx)
                done = True
    else:
        print(f"Image {i}: Chessboard corners not found")

cv.destroyAllWindows()

# camera.stop_stream(color)
# camera.stop_stream(depth)

