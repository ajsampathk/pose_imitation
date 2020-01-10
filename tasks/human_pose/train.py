import torch
from torch.autograd import Variable
import json
import sys
from ModelClass import LinearModel



args = sys.argv
if not len(args)>1:
    print("Usage: \n \t python {} [PATH_TO_DATASET_JSON]".format(args[0]))
    exit()
DATASET = args[1]

device = torch.device("cuda")

dataset = json.load(open(DATASET,'r'))
labels = json.load(open('LabelIndex.json','r'))


INPUT = Variable(torch.tensor([dataset[dat]['input'] for dat in dataset]))
OUTPUT = Variable(torch.tensor([dataset[dat]['output'] for dat in dataset]))


model = LinearModel(len(INPUT[0]),len(OUTPUT[0]))
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(),lr=0.05)

for epoch in range(70000):
    pred_y = model(INPUT.float())
    loss = criterion(pred_y,OUTPUT.float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Epoch {}, Loss {}".format(epoch,loss.data),end='\r')
    sys.stdout.flush()

x = int(input("\nEnter index of element:"))
new_x = INPUT[x]
pred = model(new_x.float())
pred_label = labels[pred.data.tolist().index(max(pred.data.tolist()))]
actual_label = [labels[i] for i,p in enumerate(OUTPUT[x]) if p]
print("Prediction: {}\n Actual: {}\n Scores:{}".format(pred_label,actual_label,pred.data.tolist()))
torch.save(model.state_dict(),"models/classifier_net_"+DATASET.split('.')[0]+"_"+str(len(INPUT[0]))+"x"+str(len(OUTPUT[0]))+"_"+str(len(INPUT))+".pth")
