
import argparse
import rosbag
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np



def main():

# run1 start from 10
    bagIn = rosbag.Bag("./run3.bag", "r")
    bagOut = rosbag.Bag("./run3_rgb.bag", "w")

    bridge = CvBridge()

    count = 0
    for topic, msg, t in bagIn.read_messages(topics=["/cam1/rgb/image_raw"]):
        cv_img = np.array(bridge.imgmsg_to_cv2(msg))
        cv_img = cv_img[:,:,0:3]
        count += 1
        if(count % 5 != 0):
            continue

        img_msg = bridge.cv2_to_imgmsg(cv_img, encoding="rgb8")
        img_msg.header = msg.header
        bagOut.write("/cam1/rgb/image_raw", img_msg, msg.header.stamp )
        print("count: ",count)

    count = 0
    for topic, msg, t in bagIn.read_messages(topics=["/cam2/rgb/image_raw"]):
        cv_img = np.array(bridge.imgmsg_to_cv2(msg))
        cv_img = cv_img[:,:,0:3]
        count += 1
        if(count % 5 != 0):
            continue

        img_msg = bridge.cv2_to_imgmsg(cv_img, encoding="rgb8")
        img_msg.header = msg.header
        bagOut.write("/cam2/rgb/image_raw", img_msg, msg.header.stamp )
        print("count: ",count)

    print("Done converting " + str(count) + " images")

    bagIn.close()
    bagOut.close()

if __name__ == "__main__":
    main()
