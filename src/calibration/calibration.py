import cv2

# assert cv2.__version__[0] >= '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import os
import glob
import fnmatch

CHECKERBOARD = (6, 6)
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_S1_S2_S3_S4 + cv2.CALIB_FIX_K6

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
_img_shape = None
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

matches = []
for root, dirnames, filenames in os.walk('chessboard'):
    for filename in fnmatch.filter(filenames, '*.jpg'):
        matches.append(os.path.join(root, filename))
print(matches)

for fname in matches:
    img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    if _img_shape is None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."

    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(img, CHECKERBOARD,
                                             cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_CB_ADAPTIVE_THRESH +
                                             cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    # If found, add object points, image points (after refining them)
    if ret:
        print(fname)
        objpoints.append(objp)
        cv2.cornerSubPix(img, corners, (3, 3), (-1, -1), subpix_criteria)
        imgpoints.append(corners)
        N_OK = len(objpoints)

# K = np.zeros((3, 3))
# D = np.zeros((4, 1))
# rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
# tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
ret, mtx, dist, rvecs, tvecs = \
    cv2.calibrateCamera(
        objpoints,
        imgpoints,
        img.shape[::-1],
        None,
        None,
        None,
        None,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )

print("Found " + str(N_OK) + " valid images for calibration")
print("DIM =" + str(_img_shape[::-1]))
print("K = np.array(" + str(mtx.tolist()) + ")")
print("D = np.array(" + str(dist.tolist()) + ")")
print("rvecs = np.array(" + str(rvecs) + ")")
print("tvecs = np.array(" + str(tvecs) + ")")
