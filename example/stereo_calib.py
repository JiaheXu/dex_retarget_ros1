import cv2 as cv
import glob
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import numpy as np

import argparse
import rosbag
import rospy

def stereo_calibrate(mtx1, dist1, mtx2, dist2, frames_folder):

    c1_images = []
    c2_images = []

    bridge = CvBridge()
    bagIn = rosbag.Bag(frames_folder, "r")
    count = 0
    for topic, msg, t in bagIn.read_messages(topics=["/cam1/rgb/image_raw"]):
        cv_img = np.array(bridge.imgmsg_to_cv2(msg))
        cv_img = cv_img[:,:,0:3]
        count += 1
        if(count%5!=0):
            continue
        c1_images.append(cv_img)
    
    count = 0
    for topic, msg, t in bagIn.read_messages(topics=["/cam2/rgb/image_raw"]):
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
    print("error: ", sqrt(ret/( len(imgpoints_left)*2) )) 
    return R, T


mtx1 = np.array([ 
    [ 611.9021606445312, 0.0, 637.0317993164062],
    [ 0.0, 611.7799682617188, 369.0512390136719],
    [ 0.0, 0.0, 1.0]
])
dist1 = np.array( [0.5463702082633972, -2.601414203643799, 0.0008451102185063064, -0.0003721700340975076, 1.4684650897979736] )

#   // For "rational_polynomial", the 8 parameters are: (k1, k2, p1, p2, k3, k4, k5, k6).
#   camera_info.D = {parameters->param.k1, parameters->param.k2, parameters->param.p1, parameters->param.p2,
#                    parameters->param.k3, parameters->param.k4, parameters->param.k5, parameters->param.k6};

# distCoeffs1 output vector of distortion coefficients [k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,taux,tauy] of 4, 5, 8, 12 or 14 elements. The output vector length depends on the options.
# D: [0.5463702082633972, -2.601414203643799, 0.0008451102185063064, -0.0003721700340975076, 1.4684650897979736, 0.42450839281082153, -2.430366039276123, 1.4001946449279785]
# k1 k2 p1 p2 k3
mtx2 = np.array([
    [607.1500244140625, 0.0, 641.7113647460938], 
    [0.0, 607.0665893554688, 365.9603576660156],
    [ 0.0, 0.0, 1.0]
])
dist2 = np.array([0.4385905861854553, -2.6185202598571777, -0.00028256000950932503, -0.00051872682524845, 1.5916898250579834] )
# D: [0.4385905861854553, -2.6185202598571777, -0.00028256000950932503, -0.00051872682524845, 1.5916898250579834, 0.3232973515987396, -2.449460506439209, 1.5187499523162842]
#camera matrix, distortion coefficients
R, T = stereo_calibrate(mtx1, dist1, mtx2, dist2, 'run1.bag')
print("R:\n", R) 
print("T:\n", T)



