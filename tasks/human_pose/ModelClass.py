import torch

class LinearModel(torch.nn.Module):

  def __init__(self,_in,out):
    super(LinearModel,self).__init__()
    self.input = torch.nn.Linear(_in,round(_in*0.6))
    self.hidden_1 = torch.nn.Linear(round(_in*0.6),18)
    self.hidden_2 = torch.nn.Linear(18,18)
    self.hidden_3 = torch.nn.Linear(18,8)
    self.hidden_4 = torch.nn.Linear(8,8)
    self.output = torch.nn.Linear(8,out)
    
    self.activation = torch.nn.Sigmoid()


  def forward(self,x):
    x = self.input(x)

    x = self.hidden_1(x)



    x = self.hidden_2(x)

    x = self.hidden_3(x)

    x = self.activation(x)

    x = self.hidden_4(x)

    x = self.activation(x)

    x = self.output(x)
    y_pred = self.activation(x)
   
    return y_pred

