"""
Contains PyTorch model code to instantiate a TinVGG model.
"""
import torch
from torch import nn
class TinyVGG(nn.Module):
  """
  Creates the TinyVGG architecture.
  Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
  Args:
    input_shape:An integer indicating number of input channels.
    hidden_units:An integer indicating number of hidden units between layers.
    output_shape:An integer indicating number of output channels.
  """
  def __init__(self, input_shape:int, hidden_units:int, output_shape:int)->None:
    super().__init__()
    self.conv_block1=nn.Sequential(
        nn.Conv2d(in_channels=input_shape,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2)
    )
    self.conv_block2=nn.Sequential(
        nn.Conv2d(hidden_units,hidden_units,3,padding=0),
        nn.ReLU(),
        nn.Conv2d(hidden_units,hidden_units,kernel_size=3,padding=0),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    self.classifier=nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*53*53,
                  out_features=output_shape)
    )
  def forward(self,x:torch.Tensor):
    x=self.conv_block1(x)
    # print(x.shape)
    x=self.conv_block2(x)
    # print(x.shape)
    x=self.classifier(x)
    return x
    # return self.classifier(self.conv_block2(self.conv_block1(x))) # <- leverage the  benefits of operator fusion
