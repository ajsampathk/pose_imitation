print("Loading Torch Modules..")
import torch
from torch.autograd import Variable
import json
from ModelClass import LinearModel
print("Done")
print("Loading ROS Modules..")
import rospy
from trt_bridge.msg import Human
from trt_bridge.msg import keypoint
from std_msgs.msg import String
import math

print("Done")
device = torch.device("cuda")

labels = json.load(open('LabelIndex.json','r'))
human_pose = json.load(open('human_pose.json','r'))


print("Loading Classifier Model..")
IN_SIZE = 36
OUT_SIZE = len(labels)
model = LinearModel(IN_SIZE,OUT_SIZE)
print(model.load_state_dict(torch.load("models/classifier_net_Labels_36x5_h1xh2_36x21_size_494.pth")))
print("Done")

print("Starting ROS Node..")
rospy.init_node("Human_Pose_Classifier")
pred_pub = rospy.Publisher("/Human_Pose/Prediction",String,queue_size=5)


def get_angles(person):
    skeleton = human_pose['skeleton']
    keypoints = human_pose['keypoints']
    angles = []
    #rospy.loginfo(person)
    for link in skeleton:
        try:
        #rospy.loginfo(keypoints[link[0]])
          if person.get(keypoints[link[0]-1])==(-1,-1) or person.get(keypoints[link[1]-1])==(-1,-1):
              angles.append(-10)
          else:
              y1 = person.get(keypoints[link[1]-1])[1]
              y0 = person.get(keypoints[link[0]-1])[1]
              x1 = person.get(keypoints[link[1]-1])[0]
              x0 = person.get(keypoints[link[0]-1])[0]
              numerator = (y1-y0)
              denominator = (x1-x0)
              m = numerator/denominator
              angles.append(math.atan(m))
        except ZeroDivisionError as e:
            rospy.logwarn(e)
            angles.append(-10)
    return angles

def keypoints_proc(msg):
    person = {}
    num_data = []
    for key in human_pose['keypoints']:
        person[key] = (getattr(msg,key).x,getattr(msg,key).y)
    rospy.loginfo("Recieved input tensor length:{}".format(len(person)))
    angles = get_angles(person)
    rospy.loginfo(angles)
    x_in = Variable(torch.tensor([angles]))
    pred = model(x_in.float())
    scores = pred.data.tolist()[0]
    pred_conf = max(scores)
    pred_label = labels[scores.index(pred_conf)]
    pubstr = "Prdection: {}, confidence:{}, scores:{}".format(pred_label,pred_conf,scores)
    rospy.loginfo(pubstr)
    pred_pub.publish(pubstr)
    
    

def keypoints_raw(msg):
    num_data = []
    for key in human_pose['keypoints']:
        num_data.append(getattr(msg,key).x)
        num_data.append(getattr(msg,key).y)
    rospy.loginfo("Recieved input tensor length:{}".format(len(num_data)))
    x_in = Variable(torch.tensor([num_data]))
    pred = model(x_in.float())
    scores = pred.data.tolist()
    pred_conf = max(scores[0])
    pred_label = labels[scores[0].index(pred_conf)]
    pubstr = "Prdection: {}, confidence:{}, scores:{}".format(pred_label,pred_conf,scores)
    rospy.loginfo(pubstr)
    pred_pub.publish(pubstr)

if __name__ == '__main__':
    rospy.Subscriber("/Inference/Humans",Human,keypoints_raw)
    while not rospy.is_shutdown():
        rospy.spin()
    
    

