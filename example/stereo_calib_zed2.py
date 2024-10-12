import cv2 as cv
import glob
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import numpy as np

import argparse
import rosbag
import rospy

def stereo_calibrate(mtx1, dist1, mtx2, dist2, frames_folder):
    #read the synched frames
    images_names = glob.glob(frames_folder)
    images_names = sorted(images_names)

 
    c1_images = []
    c2_images = []

    bridge = CvBridge()
    bagIn = rosbag.Bag(frames_folder, "r")
    count = 0
    for topic, msg, t in bagIn.read_messages(topics=["/zed2/zed_node/left/image_rect_color"]):
        cv_img = np.array(bridge.imgmsg_to_cv2(msg))
        cv_img = cv_img[:,:,0:3]
        count += 1
        if(count%5!=0):
            continue
        c1_images.append(cv_img)
    
    count = 0
    for topic, msg, t in bagIn.read_messages(topics=["/zed2/zed_node/right/image_rect_color"]):
        cv_img = np.array(bridge.imgmsg_to_cv2(msg))
        cv_img = cv_img[:,:,0:3]
        count += 1
        if(count%5!=0):
            continue
        c2_images.append(cv_img)

    #change this if stereo calibration not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
 
    rows = 6 #number of checkerboard rows.
    columns = 7 #number of checkerboard columns.
    world_scaling = 0.05 #change this to the real world square size. Or not.
 
    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
 
    #frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]
 
    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []

    pattern = (6, 7)

    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
 
    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, pattern, None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, pattern, None)
 
        if c_ret1 == True and c_ret2 == True:
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
 
            #print(corners1)
            cv.drawChessboardCorners(gray1, pattern, corners1, c_ret1)
            cv.imshow('img', gray1)
 
            cv.drawChessboardCorners(gray2, pattern, corners2, c_ret2)
            cv.imshow('img2', gray2)
            k = cv.waitKey(100)
 
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
 
    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
                                                                 mtx2, dist2, (width, height), criteria = criteria, flags = stereocalibration_flags)
 
    print(ret)
    return R, T


mtx1 = np.array([ 
    [ 537.062880969855, 0.0, 545.2841926994336],
    [ 0.0, 537.4913086024168, 322.48637729774714],
    [ 0.0, 0.0, 1.0]
])
dist1 = np.array( [-0.0011421544278270694, 0.0001897652124749736, 0.0006611587633698628, -0.002061749881492359] )

mtx2 = np.array([
    [536.9785094834107, 0.0, 545.2494322809677], 
    [0.0, 537.4881426327885, 321.8942875071399],
    [ 0.0, 0.0, 1.0]
])
dist2 = np.array( [0.0005093854804305595, -0.002645308988378109, 0.0006469607047076213, -0.002117898296552639])
#camera matrix, distortion coefficients
R, T = stereo_calibrate(mtx1, dist1, mtx2, dist2, 'run2.bag')
print("R:\n", R) 
print("T:\n", T)


# 0.2991762245813406
# R:
#  [[ 9.99999798e-01 -1.52354287e-04 -6.16545148e-04]
#  [ 1.52555518e-04  9.99999935e-01  3.26350516e-04]
#  [ 6.16495388e-04 -3.26444508e-04  9.99999757e-01]]
# T:
#  [[-1.19507285e-01]
#  [ 8.17796703e-05]
#  [ 5.07066585e-04]]

# .2711038898282269
# R:
#  [[ 9.99999986e-01 -7.76970383e-05 -1.48012268e-04]
#  [ 7.77443177e-05  9.99999946e-01  3.19449601e-04]
#  [ 1.47987440e-04 -3.19461104e-04  9.99999938e-01]]
# T:
#  [[-1.20065732e-01]
#  [ 1.17439830e-04]
#  [ 5.46892788e-04]]

