import copy
from layers import *


def build_cnn(device,vgg,mean,std,style_img,content_img,style_layers,content_layers):
    """Deconstructs the inputted CNN (which should be VGG19), and inserts LossLayers
    as defined above at the specified positions."""

    # Start network using nn.Sequential, with first (bottom) layer being Norm_Layer
    net = nn.Sequential(Norm_Layer(mean, std).to(device))

    # Initialize lists which will point to style layers and content layers so we
    # can retrieve losses later
    style_loss_list, content_loss_list = [], []

    # Start off with a copied version of VGG19 with copied pre-trained weights
    vgg = copy.deepcopy(vgg)
    conv_index = 1

    # Go through layers of inputted CNN, inserting Loss Layer when needed
    for name, layer in vgg.named_children():
        # Add layers in order, if ReLU convert to inplace=False for proper backprop functionality
        net.add_module(name, nn.ReLU(inplace=False) if isinstance(layer, nn.ReLU) else layer)

        # Insert Loss Layers (computing loss and passing output through) at certain points.
        if isinstance(layer, nn.Conv2d):
            if conv_index in content_layers:
                target_content = net(content_img).detach()
                loss_module = ContentLoss_Layer(target_content)
                net.add_module('loss_layer_content_' + str(conv_index), loss_module)
                content_loss_list.append(loss_module)
            if conv_index in style_layers:
                target_style = net(style_img).detach()
                loss_module = ContentLoss_Layer(target_style)
                net.add_module('loss_layer_style_' + str(conv_index), loss_module)
                style_loss_list.append(loss_module)
            conv_index += 1

    # Gets index of last layer
    for i in range(len(net) - 1, -1, -1):
        if isinstance(net[i], StyleLoss_Layer) or isinstance(net[i], ContentLoss_Layer):
            break
    # Removes last layers based on index 
    net = net[:(i + 1)]
    return net, style_loss_list, content_loss_list