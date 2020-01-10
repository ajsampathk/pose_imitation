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

print("Done")
device = torch.device("cuda")

labels = json.load(open('LabelIndex.json','r'))
human_pose = json.load(open('human_pose.json','r'))


print("Loading Classifier Model..")
IN_SIZE = 21
OUT_SIZE = len(labels)
model = LinearModel(IN_SIZE,OUT_SIZE)
print(model.load_state_dict(torch.load("models/classifier_net_Labels_processed_21x5_101.pth")))
print("Done")

print("Starting ROS Node..")
rospy.init_node("Human_Pose_Classifier")
pred_pub = rospy.Publisher("/Human_Pose/Prediction",String,queue_size=5)


def get_angles(person):
    skeleton = human_pose['skeleton']
    angles = []
    for link in skeleton:
        try:
            if person[link[0]]==(-1,-1) or person[link[1]]==(-1,-1):
                angles.append(-10)
            else:
                m = (person[link[1]][1]-person[link[0]][1])/(person[link[1]][0]-person[link[0]][0])
                angles.append(math.atan(m))
        except:
            angles.append(-10)
    return angles

def keypoints_proc(msg):
    person = {}
    num_data = []
    for key in human_pose['keypoints']:
        person[key] = (getattr(msg,key).x,getattr(msg,key).y)
    rospy.loginfo("Recieved input tensor length:{}".format(len(person)))
    x_in = Variable(torch.tensor([get_angles(person)]))
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
    rospy.Subscriber("/Inference/Humans",Human,keypoints_proc)
    while not rospy.is_shutdown():
        rospy.spin()
    
    

