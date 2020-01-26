#! /usr/bin/env python

import rospy
import numpy as np
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from PIL import ImageFont, ImageDraw
from PIL import Image as IMG

rospy.init_node('Inferece_Bridge')

image_pub = rospy.Publisher('/Pose/image_raw',Image,queue_size=5)
bridge = CvBridge()

Prediction = "None"

def update_pred(msg):
  global Prediction
  Prediction = msg.data.split(',')[0].split(':')[1]


def cb(msg):
  global Prediction
  img = np.array(list(bytearray(msg.data)),dtype='uint8')
  img = img.reshape(msg.width,msg.height,3)
  cv2_im_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  pil_img = IMG.fromarray(cv2_im_rgb)
  draw = ImageDraw.Draw(pil_img)
  font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf",25,encoding="unic")
  draw.text((0,0),Prediction,font=font)
  img = cv2.cvtColor(np.array(pil_img),cv2.COLOR_RGB2BGR)
  try:
    image_pub.publish(bridge.cv2_to_imgmsg(img,"bgr8"))
    #rospy.loginfo("Inference image published")
  except CvBridgeError as e:
    print(e)

rospy.Subscriber('/Inference/pre_image',Image,cb)
rospy.Subscriber("/Human_Pose/Prediction",String,update_pred)


while not rospy.is_shutdown():
  rospy.loginfo("Started Bridge Node")
  rospy.spin()
