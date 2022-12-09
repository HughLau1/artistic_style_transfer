import torch
import torchvision.transforms as T
# Contains a few other utilities.

def calculate_gram(x):
    """Takes gram matrix of x with shape batch_size, channels,
    h, w. """
    batch_size,channels,d1,d2=x.size()
    new_d1,new_d2=batch_size*channels,d1*d2
    x=x.view(new_d1,new_d2)  # reshape to 2d
    return torch.mm(x,x.t()).div(new_d1*new_d2)  # multiply reshaped x by its transpose,
    # then divide by the size to normalize


preprocess=lambda x:T.Compose([T.Resize(x),T.CenterCrop(x),T.ToTensor()])
# preprocess by resizing, taking center, then making to tensor

im_dir='./img/'
in_dir='in/'
out_dir='out/'
