print("Loading TRT Modules....")
import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
from torch2trt import TRTModule
import cv2
import torchvision.transforms as transforms
import PIL.Image
from jetcam.usb_camera import USBCamera


from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects



OPTIMIZED_MODEL = 'models/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'


with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

print("Done")

print("Loading ROS Modules....")
import rospy
from sensor_msgs.msg import Image
from trt_bridge.msg import *
human_msg = Human()

rospy.init_node('Inference')
image_pub = rospy.Publisher("/Inference/pre_image",Image,queue_size=5)
human_pub = rospy.Publisher("/Inference/Humans", Human,queue_size=5)
print("Done")

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])
print("Loading Model...")
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
print("Done")
mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

parse = ParseObjects(topology)
draw = DrawObjects(topology)
camera = USBCamera(width=224, height=224, capture_fps=30)
camera.running = True

def publish_image(image):
    img = Image()
    (img.width,img.height,n) = image.shape
    img.encoding = "rgb8"
    img.is_bigendian = 0
    img.data = image.ravel().tolist()
    img.step = 3
    image_pub.publish(img)

def human_publish(people):
    for i in range(len(people)):
        human_msg.id = i
        for key in people[i]:
            part = keypoint()
            part.x,part.y = people[i][key]
            setattr(human_msg,key,part)
        human_pub.publish(human_msg)        
    

print("Starting Inference..")
def execute(change):
    try:
        image = change['new']
        data = preprocess(image)
        cmap, paf = model_trt(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = parse(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
        draw(image, counts, objects, peaks)
        publish_image(image)
        human_publish(get_points(counts,objects,peaks))
        rospy.loginfo("Inference complete {} person(s) detected".format(counts[0]))
    except KeyboardInterrupt:
        print("Shutting Down...")
        camera.unobserve_all()
        exit()

camera.observe(execute,names='value')

    
def get_points(counts,objects,peaks):
    height = 224
    width = 224
    k = topology.shape[0]
    count = counts[0]
    people = []
    for human in range(count):
        obj = objects[0][human]
        person = {}
        for key in range(obj.shape[0]):
            value = int(obj[key])
            if value >=0:
                peak = peaks[0][key][value]
                x,y = (round(float(peak[1])*width),round(float(peak[0])*height))
            else:
                x,y = -1,-1
            person[human_pose['keypoints'][key]]=(x,y)
        people.append(person)
    return people        
    
    
    

