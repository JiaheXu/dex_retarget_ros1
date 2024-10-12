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
    
    def __init__(self, retargeting: SeqRetargeting, video_path: str, output_path: str, config_path: str, input_topic: str, output_topic: str, pos_topic: str):
        self.retargeting = retargeting
        self.output_path = output_path
        self.config_path = config_path
        self.detector = SingleHandDetector(hand_type="Right", selfie=False)
        self.joint_sub = rospy.Subscriber("/joint_3d", nav.Path, self.pos_callback)
        self.joint_pub = rospy.Publisher('/qpos', Float32MultiArray, queue_size=1000)
    def run(self):
        rospy.spin()  

    def callback(self, joint_msg):   #depth_msg):
        
        keypoint_3d = np.empty([21, 3])
        for i in range(21):
            keypoint_3d[i][0] = joint_msg.pose.position.x
            keypoint_3d[i][1] = joint_msg.pose.position.y
            keypoint_3d[i][2] = joint_msg.pose.position.z
        keypoint_3d = keypoint_3d - keypoint_3d[0:1, :]
        
        mediapipe_wrist_rot = self.detector.estimate_frame_from_hand_points(keypoint_3d)
        
        joint_pos = keypoint_3d @ mediapipe_wrist_rot @ self.detector.operator2mano        
        
        #print("mediapipe_wrist_rot:\n", mediapipe_wrist_rot)
        retargeting_type = self.retargeting.optimizer.retargeting_type
        indices = self.retargeting.optimizer.target_link_human_indices

        if retargeting_type == "POSITION":
            indices = indices
            ref_value = joint_pos[indices, :]
        else:
            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
        qpos = self.retargeting.retarget(ref_value)
        qpos_msg = Float32MultiArray()
        qpos_msg.data = qpos
        self.joint_pub.publish(qpos_msg)

    def pos_callback(self, joint_msg):
        for i in range(21):
            print(joint_msg.poses[i].pose.position.x, joint_msg.poses[i].pose.position.y)


def main(robot_name: RobotName, video_path: str, output_path: str, retargeting_type: RetargetingType, hand_type: HandType, input_topic: str, output_topic: str, pos_topic: str, node_name: str):
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

    config_path = get_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = Path(__file__).parent.parent / "assets" / "robots"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    rospy.init_node(node_name)
    retarget_node = retarget(retargeting, video_path, output_path, str(config_path), input_topic, output_topic, pos_topic)
    retarget_node.run()


if __name__ == "__main__":
    tyro.cli(main)
