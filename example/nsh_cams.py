import cv2 as cv
import glob
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import argparse
import rosbag
import rospy
def stereo_calibrate(mtx1, dist1, topic1, mtx2, dist2, topic2, frames_folder):
    c1_images = []
    c2_images = []
    bridge = CvBridge()
    bagIn = rosbag.Bag(frames_folder, "r")
    count = 0
    for topic, msg, t in bagIn.read_messages(topics=[topic1]):
        cv_img = np.array(bridge.imgmsg_to_cv2(msg))
        cv_img = cv_img[:,:,0:3]
        count += 1
        if(count%5!=0):
            continue
        c1_images.append(cv_img)
    count = 0
    for topic, msg, t in bagIn.read_messages(topics=[topic2]):
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
    print("error: ", ret/( len(imgpoints_left)*2) )
    return R, T
#   // For "rational_polynomial", the 8 parameters are: (k1, k2, p1, p2, k3, k4, k5, k6).
#   camera_info.D = {parameters->param.k1, parameters->param.k2, parameters->param.p1, parameters->param.p2,
#                    parameters->param.k3, parameters->param.k4, parameters->param.k5, parameters->param.k6};
# distCoeffs1 output vector of distortion coefficients [k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,taux,tauy] of 4, 5, 8, 12 or 14 elements. The output vector length depends on the options.
# D: [0.5463702082633972, -2.601414203643799, 0.0008451102185063064, -0.0003721700340975076, 1.4684650897979736, 0.42450839281082153, -2.430366039276123, 1.4001946449279785]
# k1 k2 p1 p2 k3
#cam1
mtx1 = np.array([
    [ 628.49639982, 0.0, 639.43835794],
    [ 0.0, 629.29462499, 361.07535495],
    [ 0.0, 0.0, 1.0]
])
dist1 = np.array( [9.84535386e-02, -3.80486334e-02, -4.49539881e-03,  2.85065003e-05, -4.77705039e-02] )
mtx2 = np.array([
    [621.86283081, 0.0, 638.01012662],
    [0.0, 621.48430252, 357.60141644],
    [ 0.0, 0.0, 1.0]
])
dist2 = np.array([0.1133566,  -0.10474354, -0.00360539, -0.00078946,  0.04916464] )
mtx3 = np.array([[616.27784461,   0.,         640.92196493],
 [  0.,         616.78715286, 370.50921222],
 [  0.,           0.,           1.        ]])
dist3 = np.array( [[ 0.09338165, -0.07603509,  0.00072092,  0.00256457,  0.03110497]])
mtx4 = np.array( [[610.70215529,   0.,         639.95135753],
 [  0.,         610.45072585, 362.20373558],
 [  0.,          0.,           1.        ]] )
dist4 = np.array( [[ 0.08772232, -0.03140344, -0.00100429,  0.00120651, -0.0021742 ]])
topic1 = "/cam1/rgb/image_raw"
topic2 = "/cam2/rgb/image_raw"
topic3 = "/cam3/rgb/image_raw"
topic4 = "/cam4/rgb/image_raw"
#camera matrix, distortion coefficients
R, T = stereo_calibrate(mtx1, dist1, topic1, mtx4, dist4, topic4, '1and4_03.bag')
print("R:\n", R)
print("T:\n", T)
# NSH
# cam1 cam2
# error 1:  4.4140831883974065e-07
# error 2:  7.317789517099724e-07
# mtx1:  [[628.49639982   0.         639.43835794]
#  [  0.         629.29462499 361.07535495]
#  [  0.           0.           1.        ]]
# dist1:  [[ 9.84535386e-02 -3.80486334e-02 -4.49539881e-03  2.85065003e-05 -4.77705039e-02]]
# mtx2:  [[621.86283081   0.         638.01012662]
#  [  0.         621.48430252 357.60141644]
#  [  0.           0.           1.        ]]
# dist2:  [[ 0.1133566  -0.10474354 -0.00360539 -0.00078946  0.04916464]]
# run1
# error:  0.0003127770240530697
# R:
#  [[ 0.01899019  0.71493924 -0.69892865]
#  [-0.73963041  0.48041605  0.47132503]
#  [ 0.6727453   0.50799833  0.53791399]]
# T:
#  [[ 0.47955101]
#  [-0.34617139]
#  [ 0.25750319]]
# run2
# error:  0.00041444984055560705
# R:
#  [[ 0.01889514  0.7149549  -0.6989152 ]
#  [-0.73965182  0.48034091  0.47136801]
#  [ 0.67272443  0.50804734  0.5378938 ]]
# T:
#  [[ 0.47951527]
#  [-0.34618131]
#  [ 0.25760447]]
# run3
# error:  0.00034824636409752066
# R:
#  [[ 0.01778008  0.71463816 -0.69926831]
#  [-0.73884965  0.4805921   0.47236896]
#  [ 0.67363571  0.50825539  0.53655512]]
# T:
#  [[ 0.47973419]
#  [-0.34683412]
#  [ 0.25843358]]
# cam2 & cam3
# error:  0.0006186194266184225
# R:
#  [[-0.02778379  0.68829588 -0.72489781]
#  [-0.73088782  0.48072489  0.48446525]
#  [ 0.68193186  0.54327926  0.48971072]]
# T:
#  [[ 0.27993001]
#  [-0.25097295]
#  [ 0.35349544]]
# error:  0.0008102634380934751
# R:
#  [[-0.02800272  0.68851104 -0.72468503]
#  [-0.73049219  0.48077942  0.48500754]
#  [ 0.68234669  0.54295829  0.48948881]]
# T:
#  [[ 0.27969777]
#  [-0.25117819]
#  [ 0.35359901]]
# error:  0.0018352510029214625
# R:
#  [[-0.02797701  0.68817934 -0.72500103]
#  [-0.73168141  0.48006673  0.48391967]
#  [ 0.68107239  0.5440084   0.4900972 ]]
# T:
#  [[ 0.27959422]
#  [-0.25061686]
#  [ 0.35392073]]
# cam3 & cam4
# error:  0.000571792689288192
# R:
#  [[ 0.38991166  0.87171213 -0.29679432]
#  [-0.73895082  0.48851201  0.46401261]
#  [ 0.54947301  0.03839248  0.83462891]]
# T:
#  [[ 0.26768416]
#  [-0.28147912]
#  [ 0.18662761]]
# error:  0.0006571008542406484
# R:
#  [[ 0.39016313  0.87138031 -0.29743753]
#  [-0.73835993  0.48909914  0.46433463]
#  [ 0.55008849  0.0384497   0.83422076]]
# T:
#  [[ 0.26790064]
#  [-0.28138235]
#  [ 0.18714135]]
# error:  0.0006632437979887215
# R:
#  [[ 0.39051016  0.87153611 -0.29652423]
#  [-0.73871291  0.48886877  0.46401568]
#  [ 0.54936786  0.03784344  0.83472321]]
# T:
#  [[ 0.26742297]
#  [-0.28130844]
#  [ 0.18687581]]
# cam1 & cam4
# error:  0.002437386345520016
# R:
#  [[-0.41468802 -0.3581968   0.836498  ]
#  [ 0.68912692  0.47670697  0.54576053]
#  [-0.5942541   0.80277364  0.04915839]]
# T:
#  [[-0.53625751]
#  [-0.19502799]
#  [ 0.59168761]]
# error:  0.0013080698382970778
# R:
#  [[-0.40974689 -0.35548093  0.8400838 ]
#  [ 0.68923467  0.48262757  0.54039449]
#  [-0.59754754  0.80043984  0.04725463]]
# T:
#  [[-0.53816467]
#  [-0.19090876]
#  [ 0.59172961]]
# error:  0.0014211170363042313
# R:
#  [[-0.40977408 -0.35553952  0.84004574]
#  [ 0.68881794  0.48312434  0.54048193]
#  [-0.59800923  0.80011407  0.04693017]]
# T:
#  [[-0.53791122]
#  [-0.19086703]
#  [ 0.59195299]]
