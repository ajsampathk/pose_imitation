import sys
import os
args = sys.argv
if not len(args)>1:
    print("Usage: \n \t python {} [PATH_TO_DATA_FOLDER]".format(args[0]))
    exit()
DATA_PATH = args[1]
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


from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects



OPTIMIZED_MODEL = 'models/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
dataset ={}


with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

print("Done")


print("Loading Model...")
num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
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

def rename_sample(sample,labpath,label,n):
    name,ext = sample.split('.')
    name = label+'_'+str(n).zfill(8)
    sample_new = name+'.'+ext
    os.rename(os.path.join(labpath,sample),os.path.join(labpath,sample_new))
    return sample_new
    
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

parse = ParseObjects(topology)

print("Done")

print("Dataset source dir:{}".format(DATA_PATH))
labels = os.listdir(DATA_PATH)
print("Labels found: {}".format(labels))
json.dump(labels,open('LabelIndex.json','w'))

for label in labels:
    labpath=os.path.join(DATA_PATH,label)
    samples = os.listdir(labpath)
    print("Found {} samples for label {}".format(len(os.listdir(labpath)),label))
    for i in range(len(os.listdir(labpath))):
        keypoints = {}
        sample = rename_sample(samples[i],labpath,label,i)
        print("Processing sample [{}]".format(sample))
        image = cv2.imread(os.path.join(labpath,sample))
        image = cv2.resize(image,(224,224))
        data  = preprocess(image)
        cmap, paf = model_trt(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu() 
        counts,objects,peaks = parse(cmap,paf)
        keypoints = get_points(counts,objects,peaks)[0]
        num_data = []
        for key in keypoints:
            #print(keypoints[key])
            for point in keypoints[key]:
                num_data.append(point)
        dataset[sample]={"input":num_data,"output":[label==l for l in labels]}


print("Writing to \"Lables.json\"..")
json.dump(dataset,open('Labels.json','w'))


        
        
        








