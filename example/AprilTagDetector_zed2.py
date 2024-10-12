#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pupil_apriltags import Detector
import tf.transformations as tr
from geometry_msgs.msg import Pose
from geometry_msgs.msg import *
import tf2_ros
import tf
from std_srvs.srv import SetBool
from std_msgs.msg import Bool, Int32, UInt8,UInt32MultiArray
from visualization_msgs.msg import Marker
import numpy as np


from std_msgs.msg import Bool, String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose,Quaternion, Vector3, Point
from tf.transformations import *
import time

import cv2
import numpy as np

# distortion_model: "rational_polynomial"
# D: [0.4385905861854553, -2.6185202598571777, -0.00028256000950932503, -0.00051872682524845, 1.5916898250579834, 0.3232973515987396, -2.449460506439209, 1.5187499523162842]

# K: [607.1500244140625, 0.0, 641.7113647460938,
#     0.0, 607.0665893554688, 365.9603576660156,
#     0.0, 0.0, 1.0]

# R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

# P: [607.1500244140625, 0.0, 641.7113647460938,
#     0.0, 607.0665893554688, 365.9603576660156,
#     0.0, 0.0, 1.0]

class AutoAutoCal:
    def __init__(self):
    
        self.bridge = CvBridge()
        self.relative_pose= Odometry()          
        # Create a detector object    
        self.detector = Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
        )

        #self.radio_pub2 = rospy.Publisher("/%s/AprilTagTransform"%(self.rc_car_name), Odometry,queue_size=10)    
        
        #self.AprilTagDetectedPub=rospy.Publisher("/%s/AprilTagDetected"%(self.rc_car_name),String,queue_size=1)       
        
        self.odom=Odometry()
        self.odom_pub = rospy.Publisher( "AprilTagOdom", Odometry, queue_size=1000)
        #self.radio_topic2 = "/{0}/{1}".format(self.rc_car_name, "radio_command")            
        #self.requestmappub = rospy.Publisher(self.radio_topic2, UInt8, queue_size=10)

        
        
        # Camera Instrinsics used for Undistorting the Fisheye Images
        self.DIM=(621, 1104)
        self.K=np.array([[537.062880969855, 0.0, 545.2841926994336], [0.0, 537.4913086024168, 322.48637729774714], [0.0, 0.0, 1.0]])

        self.D=np.array([-0.0011421544278270694, 0.0001897652124749736, 0.0006611587633698628, -0.002061749881492359])
            
    def subscribeRobotImage(self, robot_name="cmu_rc2"):
        # self.robo_img_subscriber=rospy.Subscriber("/webcam/image_raw", Image, self.image_callback) # /basestation/%s/front/image_raw"%robot_name
        # print("Subscribed to WebCam Image************************************************************************8")  
        self.robo_img_subscriber=rospy.Subscriber("/zed2/zed_node/left/image_rect_color", Image, self.image_callback) # 

    def image_callback(self, msg):
        # Process the received image data here
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        result = self.detector.detect(gray_image, True, camera_params=(607.1500244140625, 607.0665893554688, 641.7113647460938, 365.9603576660156),tag_size = 0.155) 
        # Camera Intrinsics after undistorting fisheye images, Tag Size is the length of the side of an aprilTag
        
        if result: 
            # print("*****************************************************************************************")
            # print(result)
            for tag in result: 
                if(tag.tag_id):  
                
                    original_estimated_rot = tag.pose_R 
                    original_estimated_trans = tag.pose_t

                    print("trans: ", original_estimated_trans)
                    print("rot: ", original_estimated_rot)
                    roll, pitch, yaw = euler_from_matrix(original_estimated_rot)
                    ''' The Coordinate System for the AprilTag and the RVIZ are different. 
                            - AprilTag -> X to the right, Y down, Z into the screen . 
                            - Rviz -> X into screen, Y towards the left and Z upwards.
                                * Orientation : the Pitch in the AprilTag's frame of reference = - Yaw in RVIZ because -Y(AprilTag) is equivalent to Z axis in RVIZ
                                * Position :
                                        ---------------------- 
                                        |  AprilTag  |  RVIZ  |
                                        ---------------------- 
                                        |     X      |    -Y  |
                                        |     Y      |    -Z  |
                                        |     Z      |     X  |
                                        ---------------------- 
                    '''
                    odom_quat = quaternion_from_euler(0, 0, -pitch)  
                    self.odom.pose.pose.position = Point(original_estimated_trans[2], -original_estimated_trans[0], -original_estimated_trans[1])# Adding 0.15m becasue of offsetCoordinates of RVIZ and AprilTag are interchanged 
                    
                    self.odom.pose.pose.orientation.x=odom_quat[0]
                    self.odom.pose.pose.orientation.y=odom_quat[1]
                    self.odom.pose.pose.orientation.z=odom_quat[2]
                    self.odom.pose.pose.orientation.w=odom_quat[3]
                    
                    self.odom.header.stamp=rospy.Time.now()
                    self.odom.header.frame_id="map"
                    self.odom_pub.publish(self.odom)


if __name__ == '__main__':
    rospy.init_node('RobotAutoCalibrationNode', anonymous=True)   
    autocal = AutoAutoCal() 
    autocal.subscribeRobotImage()
    print(" Running Auto Auto Calib Node ")  
    rospy.spin()
    
# trans:  [[0.06565139]
#  [0.3071609 ]
#  [0.94774627]]
# rot:  [[-0.96811529 -0.16438874 -0.18902152]
#  [-0.07368375 -0.53431218  0.84206959]
#  [-0.23942326  0.82914826  0.50516301]]

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # # Detect April tags in the image
        # result = self.detector.detect(gray_image,True,camera_params=(744.743972997628, 663.1086924787265,  330.78141749004544, 243.5350614899132),tag_size=0.1) #` half Image camera_params`` (``[fx, fy, cx, cy]``) and ``tag_size``(in meters) should be supplied. # Default AprilTag printed from the apriltags repo is 100 mm.
        # result = self.detector.detect(gray_image,True,camera_params=(1493.0606696757898, 1329.879207062618,  662.1274260105292, 486.2877991096021),tag_size=0.1) #` VINS   camera_params`` (``[fx, fy, cx, cy]``) and ``tag_size``(in meters) should be supplied. # Default AprilTag printed from the apriltags repo is 100 mm.
        # result = self.detector.detect(gray_image, True, camera_params=(680.69638, 681.08129, 280.85723, 257.58842),tag_size = 0.0717) #`camera_params`` (``[fx, fy, cx, cy]``) and ``tag_size``(in meters) should be supplied. # Default AprilTag printed from the apriltags repo is 100 mm.
#     '''def image_callback(self,msg):
#         # Process the received image data here
#         cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        
#         # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! !!!!!!!!!!!!!!!")
#         # Convert the image to grayscale
#         gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
#         # # Detect April tags in the image
#         # result = self.detector.detect(gray_image,True,camera_params=(744.743972997628, 663.1086924787265,  330.78141749004544, 243.5350614899132),tag_size=0.1) #` half Image camera_params`` (``[fx, fy, cx, cy]``) and ``tag_size``(in meters) should be supplied. # Default AprilTag printed from the apriltags repo is 100 mm.
#         # result = self.detector.detect(gray_image,True,camera_params=(1493.0606696757898, 1329.879207062618,  662.1274260105292, 486.2877991096021),tag_size=0.1) #` VINS   camera_params`` (``[fx, fy, cx, cy]``) and ``tag_size``(in meters) should be supplied. # Default AprilTag printed from the apriltags repo is 100 mm.
#         result = self.detector.detect(gray_image, True, camera_params=(680.69638, 681.08129, 280.85723, 257.58842),tag_size = 0.0717) #`camera_params`` (``[fx, fy, cx, cy]``) and ``tag_size``(in meters) should be supplied. # Default AprilTag printed from the apriltags repo is 100 mm.

#         if result: 
#             # print("*****************************************************************************************")
#             # print(result)
#             for tag in result: 
#                 if(tag.tag_id):  
                    
#                     original_estimated_rot = tag.pose_R 
#                     original_estimated_trans = tag.pose_t
                    
#                     roll, pitch, yaw = euler_from_matrix(original_estimated_rot)
#                     odom_quat = quaternion_from_euler(0, 0, -pitch)
                    
                    
#                     self.odom.pose.pose.position = Point(original_estimated_trans[2], -original_estimated_trans[0], -original_estimated_trans[1])# Adding 0.15m becasue of offsetCoordinates of RVIZ and AprilTag are interchanged 
                    
#                     self.odom.pose.pose.orientation.x=odom_quat[0]
#                     self.odom.pose.pose.orientation.y=odom_quat[1]
#                     self.odom.pose.pose.orientation.z=odom_quat[2]
#                     self.odom.pose.pose.orientation.w=odom_quat[3]

#                     self.odom.header.stamp=rospy.Time.now()
#                     self.odom.header.frame_id="rc3_camera"
                    
#                     self.radio_pub2.publish(self.odom)
#                     print("Publishing Robot Relative Transform to Robot Autocalib node ")
                    
#                     # # Unsubscribe fro the Image Topic - This done to avoid Unnecessary Image Callbacks
#                     print("De-Registering Image Callback")
#                     self.robo_img_subscriber.unregister()
#                     self.AprilTagDetectedPub.publish(True)
                    
        
# def EnableDisableAutoCal(msg):
#     print(" Enable disable Auocal message ", msg)
#     if msg.data:
#         autocal.subscribeRobotImage()
#         print(" Subsribing to Image")
#     else :
#         autocal.unsubscribeRobotImage()
#         print(" Deregistering Image")


# if __name__ == '__main__':
#     rospy.init_node('RobotAutoCalibrationNode', anonymous=True)   
#     autocal = AutoAutoCal() 
    
#     print(" Running Auto Auto Calib Node ")
#     #  # Check For Key press
#     rospy.Subscriber("/%s/EnableDisableAutoCal" % ("cmu_rc2"), Bool, EnableDisableAutoCal)
  
#     rospy.spin()
    
