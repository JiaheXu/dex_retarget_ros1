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

from sensor_msgs.msg import JointState

z_rot = np.array([
    [0.0, -1.0, 0.0],
    [1.0,  0.0, 0.0],
    [0.0,  0.0, 1.0]
])

x_rot = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0]
]) 

y_rot = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0]
]) 
class constructor():
    
    def __init__(self,  retargeting: SeqRetargeting, output_path: str, config_path: str, hand_type: str):

        self.retargeting = retargeting
        self.output_path = output_path
        self.config_path = config_path
        self.hand_type = hand_type
        print("hand_type: ", hand_type)

        self.detector = SingleHandDetector(hand_type=hand_type, selfie=False)
        self.initial_pose = None
        self.cam1_sub = message_filters.Subscriber("/cam1/joint_pos/%s"%hand_type, nav.Path)
        self.cam2_sub = message_filters.Subscriber("/cam2/joint_pos/%s"%hand_type, nav.Path)
        
        self.ts = message_filters.ApproximateTimeSynchronizer([self.cam1_sub, self.cam2_sub], 10, 1, allow_headerless=True)
        self.ts.registerCallback(self.callback)

        self.joint_pub = rospy.Publisher("/joint_3d/%s"%hand_type, PointCloud, queue_size=1000)
        
        self.joint_pc_publisher = rospy.Publisher("/joint_pc/%s"%hand_type, PointCloud, queue_size=10)

        self.ref_joint_pc_publisher = rospy.Publisher("/ref_joint_pc/%s"%hand_type, PointCloud, queue_size=10)

        self.target_joint_pc_publisher = rospy.Publisher("/target_joint_pc/%s"%hand_type, PointCloud, queue_size=10)
        self.robot_joint_pc_publisher = rospy.Publisher("/robot_joint_pc/%s"%hand_type, PointCloud, queue_size=10)

        self.global_publisher = rospy.Publisher("/global_odometry/%s"%hand_type, Odometry, queue_size=10)
        self.root_publisher = rospy.Publisher("/root_odometry/%s"%hand_type, Odometry, queue_size=10)
        self.action_root_publisher = rospy.Publisher("/action_root_odometry/%s"%hand_type, Odometry, queue_size=10)
        self.cam1_odom=Odometry()
        self.cam2_odom=Odometry()

        self.mtx1 = np.array([ 
            [ 537.062880969855, 0.0, 545.2841926994336],
            [ 0.0, 537.4913086024168, 322.48637729774714],
            [ 0.0, 0.0, 1.0]
            ])
        self.dist1 = np.array( [-0.0011421544278270694, 0.0001897652124749736, 0.0006611587633698628, -0.002061749881492359] )

        self.mtx2 = np.array([
            [536.9785094834107, 0.0, 545.2494322809677], 
            [0.0, 537.4881426327885, 321.8942875071399],
            [ 0.0, 0.0, 1.0]
            ])
        self.dist2 = np.array( [0.0005093854804305595, -0.002645308988378109, 0.0006469607047076213, -0.002117898296552639])

        #RT matrix for C1 is identity.
        self.RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
        self.P1 = self.mtx1 @ self.RT1 #projection matrix for C1

        self.R = np.array( 
            [[ 1.0,  0.0, 0.0 ],
            [0.0,  1.0, 0.0],
            [ 0.0,  0.0,  1.0]] 
            )
        self.T = np.array([
            [-0.12],
            [0.0],
            [0.0]]
            )



        self.RT2 = np.concatenate([self.R, self.T], axis = -1)
        self.P2 = self.mtx2 @ self.RT2 
        self.hand_action_pub = rospy.Publisher("/qpos/%s"%hand_type, JointState, queue_size=1000)

        self.cam1_to_root_trans = np.array([
            [0.06565139],
            [0.3071609],
            [0.94774627]
        ])

        self.cam1_global_rot = np.array([
            [-0.96811529, -0.16438874, -0.18902152],
            [-0.07368375, -0.53431218,  0.84206959],
            [-0.23942326,  0.82914826,  0.50516301],
        ])

        self.cam1_to_root_rot = self.cam1_global_rot @ x_rot.T @ x_rot.T @ z_rot

        self.root_to_cam1_rot = self.cam1_to_root_rot.T
        self.root_to_cam1_trans = -1 * self.cam1_to_root_rot.T @ self.cam1_to_root_trans

        self.count = 0

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

        self.count = self.count + 1
        print("count: ", self.count)

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

            p3d = p3d.reshape(3,1)
            p3d = self.root_to_cam1_rot @ p3d + self.root_to_cam1_trans

            keypoint_3d.append(p3d)
            joint_3d.points.append(Point32(p3d[0], p3d[1], p3d[2])) 
        
        self.joint_pub.publish(joint_3d)
        keypoint_3d = np.array(keypoint_3d)
        keypoint_3d = keypoint_3d.reshape(21,-1)

        root_3d = keypoint_3d[0:1, :].copy()
        
        keypoint_3d = keypoint_3d - keypoint_3d[0:1, :]

        joint_pointcloud = PointCloud()
        joint_pointcloud.header = header
        for i in range(21):
            joint_pointcloud.points.append(Point32(keypoint_3d[i][0], keypoint_3d[i][1], keypoint_3d[i][2])) 
        self.joint_pc_publisher.publish(joint_pointcloud)

        mediapipe_wrist_rot = self.detector.estimate_frame_from_hand_points(keypoint_3d)
        # print("mediapipe_wrist_rot: ", mediapipe_wrist_rot)

        hand_eul = None
        if(self.hand_type == "Right"):
            hand_eul = self.rot2eul(mediapipe_wrist_rot @ z_rot.T @ x_rot)
        else:
            hand_eul = self.rot2eul(mediapipe_wrist_rot @ z_rot @ x_rot.T)
        joint_pos = keypoint_3d @ mediapipe_wrist_rot @ self.detector.operator2mano        

        # print("mediapipe_wrist_rot:\n", mediapipe_wrist_rot)
        
        retargeting_type = self.retargeting.optimizer.retargeting_type
        indices = self.retargeting.optimizer.target_link_human_indices
        
        #print("retargeting_type: ", retargeting_type)
        
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
        
        target_vec = self.retargeting.optimizer.get_root_base_pos(ref_value, self.hand_type)
        
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
        
        quat = self.eul2quat(hand_eul[0], hand_eul[1], hand_eul[2])
        root_orient = Quaternion(quat[0], quat[1], quat[2], quat[3])
        root_odometry.pose.pose.orientation = root_orient
        self.root_publisher.publish(root_odometry)


        global_odometry =  Odometry()
        global_odometry.header = header
        global_pos = Point(self.root_to_cam1_trans[0], self.root_to_cam1_trans[1], self.root_to_cam1_trans[2])
        global_odometry.pose.pose.position = global_pos
        cam_eul = self.rot2eul(self.root_to_cam1_rot)
        quat = self.eul2quat(cam_eul[0], cam_eul[1], cam_eul[2])
        global_orient = Quaternion(quat[0], quat[1], quat[2], quat[3])
        global_odometry.pose.pose.orientation = global_orient
        self.global_publisher.publish(global_odometry)

        if self.initial_pose is None:
            self.initial_pose = root_3d.copy()
        #root_3d = root_3d - self.initial_pose

        action_root_odometry =  Odometry()
        action_root_odometry.header = header
        action_root_pos = Point(root_3d[0], root_3d[1], root_3d[2])
        action_root_odometry.pose.pose.position = action_root_pos

        action_root_odometry.pose.pose.orientation = root_orient
        self.action_root_publisher.publish(action_root_odometry)
        #print("action odom: ",action_root_odometry.pose.pose.position)
        
        action = np.concatenate( (root_3d, hand_eul, qpos) )

        joint_msg = JointState()
        joint_msg.header = header
        #print("published")
        joint_msg.position = action
        self.hand_action_pub.publish(joint_msg)

        return

def main(robot_name: RobotName, output_path: str, retargeting_type: RetargetingType, hand_type: HandType, node_name: str):
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

    print("robot_dir: ", robot_dir)

    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    print("retargeting_type: ", retargeting_type)
    #print("hand_type: ", hand_type)
    hand_type_str = "Right"
    if ( hand_type != HandType.right ):
        hand_type_str= "Left"
    rospy.init_node(node_name)
    constructor_node = constructor(retargeting, output_path, str(config_path), hand_type_str )
    constructor_node.run()


if __name__ == "__main__":
    tyro.cli(main)