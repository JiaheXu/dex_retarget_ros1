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

class converter():
    
    def __init__(self):
    
        self.img_sub1 = rospy.Subscriber("/cam1/rgb/image_raw", Image, self.callback1)
        self.img_sub2 = rospy.Subscriber("/cam2/rgb/image_raw", Image, self.callback2)
                
        # self.joint_pub = rospy.Publisher( pos_topic, nav.Path, queue_size=5)
        
        self.img_pub1 = rospy.Publisher("/cam1/rgb/image", Image, queue_size=5)
        self.img_pub2 = rospy.Publisher("/cam2/rgb/image", Image, queue_size=5)
        
    def run(self):
        rospy.spin()  

    def callback1(self, bgra_msg):
        bgra = bridge.imgmsg_to_cv2(bgra_msg)
        bgr = np.array(bgra[:,:,0:3])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        image_message = bridge.cv2_to_imgmsg(rgb, encoding="passthrough")
        image_message.header = bgra_msg.header
        
        self.img_pub1.publish(image_message)
    
    def callback2(self, bgra_msg):
        bgra = bridge.imgmsg_to_cv2(bgra_msg)
        bgr = np.array(bgra[:,:,0:3])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        image_message = bridge.cv2_to_imgmsg(rgb, encoding="passthrough")
        image_message.header = bgra_msg.header
        
        self.img_pub2.publish(image_message)
                
def main():

    rospy.init_node("img_converter")
    retarget_node = converter()
    retarget_node.run()


if __name__ == "__main__":
    tyro.cli(main)
