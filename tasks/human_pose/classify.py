import sys
args = sys.argv
if not len(args)>1:
    print("Usage: \n \t python {} [INDEX_FILE]".format(args[0]))
    exit()
INDEX = args[1]
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


labels = json.load(open(INDEX,'r'))
human_pose = json.load(open('human_pose.json','r'))


print("Loading Classifier Model..")
IN_SIZE = 6
OUT_SIZE = len(labels)
model = LinearModel(IN_SIZE,OUT_SIZE)
print(model.load_state_dict(torch.load("models/classifier_net_Labels_processed_PoseDataset_6x5_h1xh2_6x5_size_21.pth"))) #
print("Done")

print("Starting ROS Node..")
rospy.init_node("Human_Pose_Classifier")
pred_pub = rospy.Publisher("/Human_Pose/Prediction",String,queue_size=5)


def get_pairs(person):
  links = {'N_RS':('neck','right_shoulder'),'N_LS':('neck','left_shoulder'),'RS_RE':('right_shoulder','right_elbow'),'LS_LE':('left_shoulder','left_elbow'),'RE_RW':('right_elbow','right_wrist'),'LE_LW':('left_elbow','left_wrist'),'N_RH':('neck','right_hip'),'N_LH':('neck','left_hip'),'RH_RK':('right_hip','right_knee'),'RK_RA':('right_knee','right_ankle'),'LH_LK':('left_hip','left_knee'),'LK_LA':('left_knee','left_ankle')}
  
  upper_links = ['N_RS','N_LS','RS_RE','LS_LE','RE_RW','LE_LW']

  pairs = {}
  for link in upper_links:
    pairs[link] = (person.get(links[link][0]),person.get(links[link][1]))
  return pairs


def get_angles(person):

    keypoints = human_pose['keypoints']
    angles = []
    #rospy.loginfo(person)
    skeleton = get_pairs(person)
    for link in skeleton:
        try:
          
          if skeleton[link][0]==(-1,-1) or skeleton[link][1]==(-1,-1):
              angles.append(0)
          else:
              y1 = skeleton[link][1][1]#person.get(keypoints[link[1]-1])[1]
              y0 = skeleton[link][0][1]#person.get(keypoints[link[0]-1])[1]
              x1 = skeleton[link][1][0]#person.get(keypoints[link[1]-1])[0]
              x0 = skeleton[link][0][0]#person.get(keypoints[link[0]-1])[0]
              numerator = (y1-y0)
              denominator = (x1-x0)
              #rospy.loginfo(skeleton)
              m = numerator/denominator
              angles.append(math.atan(m))
        except ZeroDivisionError as e:
            #rospy.logwarn(e)
            angles.append(0)
    return angles

def keypoints_proc(msg):
    person = {}
    num_data = []
    for key in human_pose['keypoints']:
        person[key] = (getattr(msg,key).x,getattr(msg,key).y)
    #rospy.loginfo("Recieved input tensor length:{}".format(len(person)))
    angles = get_angles(person)
    rospy.loginfo(angles)
    x_in = Variable(torch.tensor([angles]))
    pred = model(x_in.float())
    scores = pred.data.tolist()[0]
    pred_conf = max(scores)
    pred_label = labels[scores.index(pred_conf)]
    pubstr = "Prdection: {}, confidence:{}".format(pred_label,pred_conf)
    rospy.loginfo(pubstr)
    pred_pub.publish(pubstr)
    
    

def keypoints_raw(msg):
    num_data = []
    for key in human_pose['keypoints']:
        num_data.append(getattr(msg,key).x)
        num_data.append(getattr(msg,key).y)
    #rospy.loginfo("Recieved input tensor length:{}".format(len(num_data)))
    x_in = Variable(torch.tensor([num_data]))
    pred = model(x_in.float())
    scores = pred.data.tolist()
    pred_conf = max(scores[0])
    pred_label = labels[scores[0].index(pred_conf)]
    pubstr = "Prdection: {}, confidence:{}".format(pred_label,pred_conf)
    rospy.loginfo(pubstr)
    pred_pub.publish(pubstr)

if __name__ == '__main__':
    rospy.Subscriber("/Inference/Humans",Human,keypoints_proc)
    while not rospy.is_shutdown():
        rospy.spin()
    
    

