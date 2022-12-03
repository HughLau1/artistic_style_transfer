import copy
from layers import *


def build_cnn(device,vgg,mean,std,im_style,im_content,style_layers,content_layers):
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
    # Markers showing whether style and content loss layers have all been added
    style_finished = False
    content_finished = False

    # Go through layers of inputted CNN, inserting Loss Layer when needed
    for label, layer in vgg.named_children():
        if content_finished and style_finished:
            break
        # Add layers in order, if ReLU convert to inplace=False for proper backprop functionality
        net.add_module(label, nn.ReLU(inplace=False) if isinstance(layer, nn.ReLU) else layer)

        # Insert Loss Layers (computing loss and passing output through) at certain points.
        if isinstance(layer, nn.Conv2d):
            if conv_index in content_layers:
                content = net(im_content).detach()
                loss_module = ContentLoss_Layer(content)
                net.add_module('loss_layer_content_' + str(conv_index), loss_module)
                content_loss_list.append(loss_module)
                if conv_index == max(content_layers):
                    content_finished = True
            if conv_index in style_layers:
                style = net(im_style).detach()
                loss_module = ContentLoss_Layer(style)
                net.add_module('loss_layer_style_' + str(conv_index), loss_module)
                style_loss_list.append(loss_module)
                if conv_index == max(style_layers):
                    style_finished = True
            conv_index += 1

    return net, style_loss_list, content_loss_list