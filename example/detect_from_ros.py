import pickle
from pathlib import Path

import cv2
import tqdm
import tyro


from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from single_hand_detector import SingleHandDetector

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

class retarget():
    
    def __init__(self, hand_type: str, input_topic: str, output_topic: str, pos_topic: str):

        self.detector = SingleHandDetector(hand_type=hand_type, selfie=False)
        
        self.rgb_sub = rospy.Subscriber(input_topic, Image, self.callback)
        self.joint_pub = rospy.Publisher( pos_topic, nav.Path, queue_size=1000)
        #self.joint_pub2 = rospy.Publisher( "/cam1/joint_pos", nav.Path, queue_size=1000)
        self.annotated_img_pub = rospy.Publisher(output_topic, Image, queue_size=100)
        #self.joint_sub = rospy.Subscriber(pos_topic, nav.Path, self.pos_callback)
        self.count = 0
        
    def run(self):
        rospy.spin()  

    def callback(self, bgra_msg):   #depth_msg):
        
        bgra = bridge.imgmsg_to_cv2(bgra_msg)
        bgr = np.array(bgra[:,:,0:3])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot, annotated_img , handbox= self.detector.detect(rgb)

        # print("keypoint_2d: \n", keypoint_2d)
        # print("mediapipe_wrist_rot:\n", mediapipe_wrist_rot)
        if joint_pos is None:
           print("no hands found")
           return
        
        self.count += 1
        print("found hand, current count: ", self.count)
        img_msg = bridge.cv2_to_imgmsg(annotated_img, encoding="rgb8")
        img_msg.header = bgra_msg.header
        self.annotated_img_pub.publish(img_msg)

        joint_msg = nav.Path()
        joint_msg.header = bgra_msg.header
        for i in range(21):
            joint = PoseStamped()
            joint.pose.position.x = keypoint_2d[i][0]
            joint.pose.position.y = keypoint_2d[i][1]
            joint_msg.poses.append(joint)

        self.joint_pub.publish(joint_msg)
        #self.joint_pub2.publish(joint_msg)

        img_msg = bridge.cv2_to_imgmsg(annotated_img, encoding="rgb8")
        img_msg.header = bgra_msg.header
        self.annotated_img_pub.publish(img_msg)

    def pos_callback(self, joint_msg):
        for i in range(21):
            print(joint_msg.poses[i].pose.position.x, joint_msg.poses[i].pose.position.y)


def main(hand_type: str, input_topic: str, output_topic: str, pos_topic: str, node_name: str):
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
    retarget_node = retarget(hand_type, input_topic, output_topic, pos_topic)
    retarget_node.run()


if __name__ == "__main__":
    tyro.cli(main)
