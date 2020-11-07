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

np_pub = rospy.Publisher('/inference/np_in',Image,queue_size=5)
image_pub = rospy.Publisher('/pose/image_raw',Image,queue_size=5)

bridge = CvBridge()

Prediction = "None"

def update_pred(msg):
  global Prediction
  Prediction = msg.data.split(',')[0].split(':')[1]


def img2np(msg):
  try:
    image=bridge.imgmsg_to_cv2(msg,msg.encoding)
    image = cv2.resize(224,224,3)
    img = Image()
    (img.width ,img.height,n)=image.shape
    img.encoding = "rgb8"
    img.is_bigendian = 0
    img.data = image.ravel().tolist()
    img.step=3
    np_pub.publish(img)
    
  except CvBridgeError as e:
    print(e)

def np2img(msg):
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

rospy.Subscriber('/inference/np_out',Image,np2img)
rospy.Subscriber('/human_Pose/prediction',String,update_pred)
rospy.Subscriber('/camera/color/image_raw',Image,img2np)


while not rospy.is_shutdown():
  rospy.loginfo("Started Bridge Node")
  rospy.spin()
