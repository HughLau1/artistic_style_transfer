import torch.nn as nn
from torch.nn.functional import mse_loss
from utils import *
# layers.py: Implementations for layers that output loss, normalization.
# Written by Hugh Lau and Spencer Jenkins

class Norm_Layer(nn.Module):
    """Network layer to be introduced at top of model, normalizes the (noisy) input."""

    def __init__(self,mean,std):
        super(Norm_Layer,self).__init__()
        self.mean=mean.clone()
        self.mean=self.mean.view(-1,1,1)
        self.std=std.clone()
        self.std=self.std.view(-1,1,1)

    def forward(self,x):
        """Pass on normalized x given mean and std. """
        return (x-self.mean)/self.std


class StyleLoss_Layer(nn.Module):
    """Network layer that computes Style Loss (MSE Loss) of Gram Matrices at given layers
    in network, to be introduced. This can be conceptualized as 2nd derivatives."""

    def __init__(self,target_style):
        super(StyleLoss_Layer,self).__init__()
        self.target_style=calculate_gram(target_style)

    def forward(self,x):
        """Take loss, store at self.loss, then pass through original input to rest of
        network. Detach because we don't need gradient of target here."""
        self.loss=mse_loss(calculate_gram(x),
                           self.target_style.detach())
        return x


class ContentLoss_Layer(nn.Module):
    """Network layer that computes Content Loss (MSE Loss) at given layers
    in network, to be introduced. """

    def __init__(self,target_content):
        super(ContentLoss_Layer,self).__init__()
        self.target_content=target_content

    def forward(self,x):
        """Take loss, store at self.loss, then pass through original input to rest of
        network. Detach because we don't need gradient of target here."""
        self.loss=mse_loss(x,self.target_content.detach())
        return x
