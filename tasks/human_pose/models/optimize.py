import sys
import json

if not len(sys.argv) == 2:
  print("Usage: \n python3 {} [MODEL_WEIGHTS_NAME].pth".format(sys.argv[0]))
  sys.exit()

MODEL_WEIGHTS = sys.argv[1]
import torch
import torch2trt
import trt_pose.models

with open('human_pose.json','r') as f:
  huamn_pose = json.load(f)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

print("Loading Resnet Model")


model = trt_pose.models.resnet18_baseline_att(num_parts,2*num_links).cuda().eval()
model.load_state_dict(torch.load(MODEL_WEIGHTS))

WIDTH = 224
HEIGHT = 224

data = torch.zeros((1,3,HEIGHT,WIDTH)).cuda()

print("Optimizing Model")

model_trt = torch2trt.torch2trt(model,[data],fp16_mode=True, max_workspace_size=1<<25)

OPTIMIZED_MODEL = 'trt'+MODEL_WEIGHTS

print("Saving Optimized model as {}".format(OPTIMIZED_MODEL))

torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)
