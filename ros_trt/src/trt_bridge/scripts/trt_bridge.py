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

image_pub = rospy.Publisher('/pose/image_raw',Image,queue_size=1)

bridge = CvBridge()

Prediction = "None"

def update_pred(msg):
  global Prediction
  Prediction = msg.data.split(',')[0].split(':')[1]


def img2np(msg):
  try:
    image=bridge.imgmsg_to_cv2(msg,msg.encoding)
    image = cv2.resize(image,(224,224))
    img = Image()
    (img.width ,img.height,n)=image.shape
    img.encoding = "rgb8"
    img.is_bigendian = 0
    img.data = image.ravel().tolist()
    img.step=3
    np_pub.publish(img)
    rospy.loginfo("Input Image Published")
    
  except CvBridgeError as e:
    print(e)

def np2img(msg):
  global Prediction
  img = np.array(list(bytearray(msg.data)),dtype='uint8')
  img = img.reshape(msg.width,msg.height,3)
  cv2_im_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  font = cv2.FONT_HERSHEY_TRIPLEX
  fontScale = 0.45
  thickness = 1
  org = (5,10)
  color=(255,255,255)
  cv2.putText(img,Prediction,org,font,fontScale,color,thickness,cv2.LINE_AA)
#  img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
  try:
    image_pub.publish(bridge.cv2_to_imgmsg(img,"bgr8"))
    rospy.loginfo("Inference image published")
  except CvBridgeError as e:
    print(e)

rospy.Subscriber('/inference/np_out',Image,np2img)
rospy.Subscriber('/human_Pose/prediction',String,update_pred)



while not rospy.is_shutdown():
  rospy.loginfo("Started Bridge Node")
  rospy.spin()
