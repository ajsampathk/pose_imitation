import torch
from torch.autograd import Variable
import json
import sys
from ModelClass import LinearModel
import random


args = sys.argv
if not len(args)>1:
    print("Usage: \n \t python {} [PATH_TO_DATASET_JSON]".format(args[0]))
    exit()
DATASET = args[1]

device = torch.device("cuda")

dataset = json.load(open(DATASET,'r'))
labels = json.load(open('LabelIndex.json','r'))


INPUT = [dataset[dat]['input'] for dat in dataset]
OUTPUT = [dataset[dat]['output'] for dat in dataset]
print("Total Sample Size:{}\n".format(len(INPUT)))
print("Partitioning Samples...") 

#partition
VAL_INPUT = []
VAL_OUTPUT = []
VAL_SIZE = int(0.1*len(INPUT))
while len(VAL_INPUT) < VAL_SIZE:
  item_idx = random.randrange(0,len(INPUT))
  VAL_INPUT.append(INPUT.pop(item_idx))
  VAL_OUTPUT.append(OUTPUT.pop(item_idx))

BATCH_INPUT = []
BATCH_OUTPUT = []
BATCH_SIZE = 100

while len(INPUT)>=BATCH_SIZE:
  BATCH_INPUT.append(INPUT[0:BATCH_SIZE])
  BATCH_OUTPUT.append(OUTPUT[0:BATCH_SIZE])
  INPUT = INPUT[BATCH_SIZE:]
  OUTPUT = OUTPUT[BATCH_SIZE:]
if len(INPUT)>=1:
  BATCH_INPUT.append(INPUT)
  BATCH_OUTPUT.append(OUTPUT)



print("Validation Sample Size:{}\n".format(len(VAL_INPUT)))


BATCH_INPUT = [Variable(torch.tensor(i)) for i in BATCH_INPUT]
BATCH_OUTPUT =[Variable(torch.tensor(i)) for i in BATCH_OUTPUT]
VAL_INPUT = Variable(torch.tensor(VAL_INPUT))
VAL_OUTPUT = Variable(torch.tensor(VAL_OUTPUT))

model = LinearModel(len(INPUT[0]),len(OUTPUT[0]))
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(),lr=0.7)

for epoch in range(500):
    for i in range(len(BATCH_INPUT)):
      pred_y = model(BATCH_INPUT[i].float())
      loss = criterion(pred_y,BATCH_OUTPUT[i].float())
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      print("Epoch {}, Loss {}".format(epoch,loss.data),end='\r')
      sys.stdout.flush()	
    val_pred = model(VAL_INPUT.float())
    val_loss = criterion(val_pred,VAL_OUTPUT.float())
    print("Validation Loss at Epoch {}: {}".format(epoch,val_loss))


saved_model = "models/classifier_net_"+DATASET.split('.')[0]+"_"+str(len(INPUT[0]))+"x"+str(len(OUTPUT[0]))+"_h1xh2_"+str(model.hidden_1.in_features)+"x"+str(model.hidden_2.out_features)+"_size_"+str(len(INPUT))+".pth"
torch.save(model.state_dict(),saved_model)

try:
  while True:
    x = int(input("\nEnter index of element:"))
    new_x = VAL_INPUT[x]
    pred = model(new_x.float())
    pred_label = labels[pred.data.tolist().index(max(pred.data.tolist()))]
    actual_label = [labels[i] for i,p in enumerate(VAL_OUTPUT[x]) if p]
    print("Prediction: {}\n Actual: {}\n Scores:{}".format(pred_label,actual_label,pred.data.tolist()))
except KeyboardInterrupt:
  print("\nModel Written to: {}".format(saved_model))
