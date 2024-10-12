import pickle
from pathlib import Path

import cv2
import tqdm
import tyro


from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from single_hand_detector import SingleHandDetector
from bi_hands_detector import BiHandsDetector

import message_filters
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from typing import Dict, Tuple
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
bridge = CvBridge()
import time
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray
#from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Point
import nav_msgs.msg as nav 

import time
import math

class retarget():
    
    def __init__(self,input_topic: str, output_topic: str, pos_topic: str):

        self.detector = BiHandsDetector( selfie=False)
        
        self.rgb_sub = rospy.Subscriber(input_topic, Image, self.callback)
        self.joint_pub = rospy.Publisher( pos_topic, nav.Path, queue_size=5)
        
        self.joint_pub_right = rospy.Publisher( pos_topic + "/Right" , nav.Path, queue_size=5)
        self.joint_pub_left = rospy.Publisher( pos_topic + "/Left" , nav.Path, queue_size=5)

        self.annotated_img_pub = rospy.Publisher(output_topic, Image, queue_size=5)

        self.count = 0
        
    def run(self):
        rospy.spin()  

    def callback(self, rgb_msg):   #depth_msg):
        start = time.time()
        rgb = bridge.imgmsg_to_cv2(rgb_msg)
        #end = time.time()
        #print("running time: ", (end - start)*1000, " ms" )

        # keypoint_2d_right, keypoint_2d_left, annotated_img = self.detector.detect(rgb)
        keypoint_2d_right, keypoint_2d_left = self.detector.detect(rgb)

        # print("keypoint_2d: \n", keypoint_2d)
        # print("mediapipe_wrist_rot:\n", mediapipe_wrist_rot)
        if keypoint_2d_right is None:
           print("no hands found")
           return
        
        # img_msg = bridge.cv2_to_imgmsg(annotated_img, encoding="rgb8")
        # img_msg.header = rgb_msg.header
        # self.annotated_img_pub.publish(img_msg)
        self.count += 1

        image_rows, image_cols, _ = rgb.shape

        joint_msg_right = nav.Path()
        joint_msg_right.header = rgb_msg.header
        for idx, landmark in enumerate(keypoint_2d_right.landmark):
            x_px = min(math.floor(landmark.x * image_cols), image_cols - 1)
            y_px = min(math.floor(landmark.y * image_rows), image_rows - 1)
            joint = PoseStamped()
            joint.pose.position.x = x_px
            joint.pose.position.y = y_px          
            joint_msg_right.poses.append(joint)            
        
        self.joint_pub_right.publish(joint_msg_right)


        joint_msg_left = nav.Path()
        joint_msg_left.header = rgb_msg.header
        for idx, landmark in enumerate(keypoint_2d_left.landmark):
            x_px = min(math.floor(landmark.x * image_cols), image_cols - 1)
            y_px = min(math.floor(landmark.y * image_rows), image_rows - 1)
            joint = PoseStamped()
            joint.pose.position.x = x_px
            joint.pose.position.y = y_px          
            joint_msg_left.poses.append(joint)
        self.joint_pub_left.publish(joint_msg_left)

        end = time.time()
        print("running time: ", (end - start)*1000, " ms" )



    def pos_callback(self, joint_msg):
        for i in range(21):
            print(joint_msg.poses[i].pose.position.x, joint_msg.poses[i].pose.position.y)


def main(input_topic: str, output_topic: str, pos_topic: str, node_name: str):
    """
    Detects the human hand pose from a video and translates the human pose trajectory into a robot pose trajectory.

    Args:
        robot_name: The identifier for the robot. This should match one of the default supported robots.
        video_path: The file path for the input video in .mp4 format.
        output_path: The file path for the output data in .pickle format.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
            Please note that retargeting is specific to the same type of hand: a left robot hand can only be retargeted
            to another left robot hand, and the same applies for the right hand.
    """

    rospy.init_node(node_name)
    retarget_node = retarget( input_topic, output_topic, pos_topic)
    retarget_node.run()


if __name__ == "__main__":
    tyro.cli(main)
