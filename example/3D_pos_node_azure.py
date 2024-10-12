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
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
#from nav_msgs.msg import Path
import nav_msgs.msg as nav
from scipy import linalg

from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
import std_msgs
class constructor():
    
    def __init__(self,  retargeting: SeqRetargeting, output_path: str, config_path: str):

        self.retargeting = retargeting
        self.output_path = output_path
        self.config_path = config_path
        self.detector = SingleHandDetector(hand_type="Right", selfie=False)
        self.initial_pose = None
        self.cam1_sub = message_filters.Subscriber("/cam1/joint_pos", nav.Path)
        self.cam2_sub = message_filters.Subscriber("/cam2/joint_pos", nav.Path)
        
        self.ts = message_filters.ApproximateTimeSynchronizer([self.cam1_sub, self.cam2_sub], 10, 1, allow_headerless=True)
        self.ts.registerCallback(self.callback)

        self.joint_pub = rospy.Publisher('/joint_3d', PointCloud, queue_size=1000)
        
        self.joint_pc_publisher = rospy.Publisher("/joint_pc", PointCloud, queue_size=10)

        self.ref_joint_pc_publisher = rospy.Publisher("/ref_joint_pc", PointCloud, queue_size=10)

        self.target_joint_pc_publisher = rospy.Publisher("/target_joint_pc", PointCloud, queue_size=10)
        self.robot_joint_pc_publisher = rospy.Publisher("/robot_joint_pc", PointCloud, queue_size=10)

        self.root_publisher = rospy.Publisher("/root_odometry", Odometry, queue_size=10)
        self.action_root_publisher = rospy.Publisher("/action_root_odometry", Odometry, queue_size=10)
        self.cam1_odom=Odometry()
        self.cam2_odom=Odometry()

        self.mtx1 = np.array([ 
            [ 611.9021606445312, 0.0, 637.0317993164062],
            [ 0.0, 611.7799682617188, 369.0512390136719],
            [ 0.0, 0.0, 1.0]
            ])
        self.dist1 = np.array( [0.5463702082633972, -2.601414203643799, 0.0008451102185063064, -0.0003721700340975076, 1.4684650897979736] )

        self.mtx2 = np.array([
            [607.1500244140625, 0.0, 641.7113647460938], 
            [0.0, 607.0665893554688, 365.9603576660156],
            [ 0.0, 0.0, 1.0]
            ])
        self.dist2 = np.array([0.4385905861854553, -2.6185202598571777, -0.00028256000950932503, -0.00051872682524845, 1.5916898250579834] )

        #RT matrix for C1 is identity.
        self.RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
        self.P1 = self.mtx1 @ self.RT1 #projection matrix for C1
 # 10.16308765418464
# R:
#  [[ 0.92984772  0.17969404 -0.3210814 ]
#  [-0.17804781  0.98340832  0.03474281]
#  [ 0.3219972   0.02486232  0.94641411]]
# T:
#  [[ 0.17697034]
#  [-0.00810655]
#  [-0.02119834]]

        self.R = np.array( 
            [[ 0.92984772,  0.17969404, -0.3210814 ],
            [-0.17804781,  0.98340832,  0.03474281],
            [ 0.3219972,   0.02486232,  0.94641411]] 
            )
        self.T = np.array([
            [ 0.17697034],
            [-0.00810655],
            [-0.02119834]] 
            )
        self.RT2 = np.concatenate([self.R, self.T], axis = -1)
        self.P2 = self.mtx2 @ self.RT2 
        self.hand_action_pub = rospy.Publisher('/qpos', Float32MultiArray, queue_size=1000)


    def DLT(self, P1, P2, point1, point2):
 
        A = [
            point1[1]*P1[2,:] - P1[1,:],
            P1[0,:] - point1[0]*P1[2,:],
            point2[1]*P2[2,:] - P2[1,:],
            P2[0,:] - point2[0]*P2[2,:]
        ]

        A = np.array(A).reshape((4,4))
        B = A.transpose() @ A
        U, s, Vh = linalg.svd(B, full_matrices = False)
 
        #print('Triangulated point: ')
        #print(Vh[3,0:3]/Vh[3,3])
        return Vh[3,0:3]/Vh[3,3]

    def rot2eul(self, R):
        #print("R:", R)
        beta = -np.arcsin(R[2,0])
        alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
        gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
        return np.array((alpha, beta, gamma))

    def eul2quat(self, roll, pitch, yaw):

        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
        return np.array([qx, qy, qz, qw])


    def run(self):
        rospy.spin()  

    def callback(self, cam1_joint_msg, cam2_joint_msg):
        
        #print(cam2_joint_msg.header.stamp)
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'map'

        joint_3d = PointCloud()
        joint_3d.header = header
        keypoint_3d = []

        for i in range(21):
            uv1 = [ cam1_joint_msg.poses[i].pose.position.x, cam1_joint_msg.poses[i].pose.position.y]
            uv2 = [ cam2_joint_msg.poses[i].pose.position.x, cam2_joint_msg.poses[i].pose.position.y]
            p3d = self.DLT(self.P1, self.P2, uv1, uv2)
            keypoint_3d.append(p3d)
            joint_3d.points.append(Point32(p3d[0], p3d[1], p3d[2])) 
        
        self.joint_pub.publish(joint_3d)
        keypoint_3d = np.array(keypoint_3d)
        keypoint_3d = keypoint_3d.reshape(21,-1)

        root_3d = keypoint_3d[0:1, :].copy()
        #print("root_3d: ", root_3d)
        keypoint_3d = keypoint_3d - keypoint_3d[0:1, :]


        joint_pointcloud = PointCloud()
        joint_pointcloud.header = header
        for i in range(21):
            joint_pointcloud.points.append(Point32(keypoint_3d[i][0], keypoint_3d[i][1], keypoint_3d[i][2])) 
        self.joint_pc_publisher.publish(joint_pointcloud)
        # print("published!!!!!!")

        mediapipe_wrist_rot = self.detector.estimate_frame_from_hand_points(keypoint_3d)
        # print("mediapipe_wrist_rot: ", mediapipe_wrist_rot)
        eul = self.rot2eul(mediapipe_wrist_rot)

        joint_pos = keypoint_3d @ mediapipe_wrist_rot @ self.detector.operator2mano        

        # print("mediapipe_wrist_rot:\n", mediapipe_wrist_rot)
        
        retargeting_type = self.retargeting.optimizer.retargeting_type
        indices = self.retargeting.optimizer.target_link_human_indices
        
        # print("retargeting_type: ", retargeting_type)
        
        if retargeting_type == "POSITION":
            indices = indices
            ref_value = joint_pos[indices, :]
        else:
            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]


        ref_joint_pointcloud = PointCloud()
        ref_joint_pointcloud.header = header
        for i in range( ref_value.shape[0] ):
            ref_joint_pointcloud.points.append( Point32( ref_value[i][0], ref_value[i][1], ref_value[i][2])) 
        self.ref_joint_pc_publisher.publish(ref_joint_pointcloud)
        #print("ref_value:\n", ref_value)

        qpos = self.retargeting.retarget(ref_value)
        
        target_vec = ref_value * self.retargeting.optimizer.scaling
        robot_pos = self.retargeting.optimizer.get_position(qpos, ref_value)

        target_joint_pointcloud = PointCloud()
        target_joint_pointcloud.header = header
        for i in range( target_vec.shape[0] ):
            target_joint_pointcloud.points.append( Point32( target_vec[i][0], target_vec[i][1], target_vec[i][2])) 
        self.target_joint_pc_publisher.publish(target_joint_pointcloud)

        robot_joint_pointcloud = PointCloud()
        robot_joint_pointcloud.header = header
        for i in range( len(robot_pos) ):
            robot_joint_pointcloud.points.append( Point32( robot_pos[i][0], robot_pos[i][1], robot_pos[i][2])) 
        self.robot_joint_pc_publisher.publish(robot_joint_pointcloud)

        root_3d = root_3d.reshape(3,)

        root_odometry =  Odometry()
        root_odometry.header = header
        root_pos = Point(root_3d[0], root_3d[1], root_3d[2])
        root_odometry.pose.pose.position = root_pos
        quat = self.eul2quat(eul[0], eul[1], eul[2])
        root_orient = Quaternion(quat[0], quat[1], quat[2], quat[3])
        root_odometry.pose.pose.orientation = root_orient
        self.root_publisher.publish(root_odometry)
        #print("odom:\n",root_odometry.pose.pose.position)
        
        # print("root_3d: ", root_3d.shape)
        # print("eul: ", eul.shape)
        # print("qpos: ", qpos.shape)

        if self.initial_pose is None:
            self.initial_pose = root_3d.copy()
        #root_3d = root_3d - self.initial_pose

        action_root_odometry =  Odometry()
        action_root_odometry.header = header
        action_root_pos = Point(root_3d[0], root_3d[1], root_3d[2])
        action_root_odometry.pose.pose.position = action_root_pos
        #quat = self.eul2quat(eul[0], eul[1], eul[2])
        #root_orient = Quaternion(quat[0], quat[1], quat[2], quat[3])
        action_root_odometry.pose.pose.orientation = root_orient
        self.action_root_publisher.publish(action_root_odometry)
        #print("action odom: ",action_root_odometry.pose.pose.position)
        
        action = np.concatenate( (root_3d, eul, qpos) )
        # print("action: ", action.shape)
        # print("qpos: ", qpos)
        # print("type: ", type(qpos) )
        # print("action: ", action)
        # print("action shape: ", action.shape)
        qpos_msg = Float32MultiArray()
        qpos_msg.data = action
        self.hand_action_pub.publish(qpos_msg)

        return

def main(robot_name: RobotName, output_path: str, retargeting_type: RetargetingType, hand_type: HandType):
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
    print("retargeting_type: ", retargeting_type)
    rospy.init_node("triangulation_node")
    constructor_node = constructor(retargeting, output_path, str(config_path))
    constructor_node.run()


if __name__ == "__main__":
    tyro.cli(main)