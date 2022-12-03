import torch
import torchvision.transforms as T

def calculate_gram(x):
    """Takes gram matrix of x with shape batch_size, channels,
    h, w. """
    batch_size,channels,h,w=x.size()
    new_h,new_w=batch_size*channels,h*w
    x=x.view(new_h,new_w)  # reshape to 2d
    return torch.mm(x,x.t()).div(new_h*new_w)  # multiply reshaped x by its transpose,
    # then divide by the size to normalize

preprocess = lambda x: T.Compose([T.Resize(x),T.CenterCrop(x),T.ToTensor()])
# preprocess by resizing, taking center, then making to tensor

im_dir = './img/'
in_dir = 'in/'
out_dir = 'out/'