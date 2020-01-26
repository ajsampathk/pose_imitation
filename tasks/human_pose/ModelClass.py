import torch

class LinearModel(torch.nn.Module):

  def __init__(self,_in,out):
    super(LinearModel,self).__init__()
    self.input = torch.nn.Linear(_in,_in)

    self.hidden_1 = torch.nn.Linear(_in,out)
    self.hidden_2 = torch.nn.Linear(out,out)

    self.output = torch.nn.Linear(out,out)
    
    self.activation = torch.nn.Sigmoid()



  def forward(self,x):
    x = self.input(x)


    x = self.hidden_1(x)
    x = self.activation(x)


    x = self.hidden_2(x)
    x = self.activation(x)
    

    x = self.output(x)
    y_pred = self.activation(x)
   
    return y_pred

