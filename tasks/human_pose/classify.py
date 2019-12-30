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
human_pose = json.load(open('human_pose.json','r'))['keypoints']


print("Loading Classifier Model..")
IN_SIZE = 36
OUT_SIZE = len(labels)
model = LinearModel(IN_SIZE,OUT_SIZE)
print(model.load_state_dict(torch.load("models/classifier_net_18x18_5.pth")))
print("Done")

print("Starting ROS Node..")
rospy.init_node("Human_Pose_Classifier")
pred_pub = rospy.Publisher("/Human_Pose/Prediction",String,queue_size=5)

def keypoints(msg):
    num_data = []
    for key in human_pose:
        num_data.append(getattr(msg,key).x)
        num_data.append(getattr(msg,key).y)
    rospy.loginfo("Recieved input tensor length:{}".format(len(num_data)))
    x_in = Variable(torch.tensor([num_data]))
    pred = model(x_in.float())
    scores = pred.data.tolist()
    pred_conf = max(scores)
    pred_label = labels[scores.index(pred_conf)]
    pubstr = "Prdection: {}, confidence:{}, scores:{}".format(pred_label,pred_conf,scores)
    rospy.loginfo(pubstr)
    pred_pub.publish(pubstr)

if __name__ == '__main__':
    rospy.Subscriber("/Inference/Humans",Human,keypoints)
    while not rospy.is_shutdown():
        rospy.spin()
    
    

