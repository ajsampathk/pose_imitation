print("Loading Modules....",end=' ')
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
from jetcam.utils import bgr8_to_jpeg

from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

OPTIMIZED_MODEL = 'models/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'


with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

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

print("Starting Inference..")
def execute(change):
    image = change['new']
    data = preprocess(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    print(get_points(counts,objects,peaks))
    draw(image, counts, objects, peaks)
    cv2.imshow('image',image)
    cv2.waitKey(1)


camera.observe(execute,names='value')

try:
    pass
except KeyboardInterrupt:
    print("shutting down..")
    camera.unobserve_all()
    
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
                person[human_pose['keypoints'][key]]=(x,y)
        people.append(person)
    return people        
    
    
    

