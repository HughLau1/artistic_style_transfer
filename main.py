from network_builder import *
import skimage as sk
import skimage.io as skio
from torch.optim import LBFGS
import torch
from torchvision.models import vgg19

def main():
    im_size = 512
    # Which device is available?
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # Weights should come from the VGG19 model. No further training necessary
    vgg_model = vgg19(pretrained=True).features
    vgg_model.to(device)
    vgg_model.eval()

    #print(vgg_model)
    #print([i for i in vgg_model.named_modules()][0])
    #exit()

    # Normalization mean and standard deviation, used in first layer of model
    mean = torch.tensor([0,0,0])
    std = torch.tensor([0,0,0])
    mean.to(device)
    std.to(device)

    # Weights for style and content
    w_style = 1e6
    w_content = 1e1

    # Import photos using SK Image into tensors
    im_content = load_image(im_dir+in_dir+'sj.jpg')
    im_style = load_image(im_dir+in_dir+'winter.jpg')
    #im_content = sk.img_as_float(skio.imread(im_dir+in_dir+'sj.jpg'))
    #im_style = sk.img_as_float(skio.imread(im_dir+in_dir+'winter.jpg'))
    im_content = preprocess(im_size)(im_content).unsqueeze(0).float().to(device)
    im_style = preprocess(im_size)(im_style).unsqueeze(0).float().to(device)
    im_target = torch.clone(im_content)

    # Using L-BFGS optimizer, aka gradient descent in first and second derivative
    optimizer = LBFGS([im_target.requires_grad_()])

    # At which layers do we want to calculate style loss and content loss?
    style_layers = [1,2,3,4,5]
    content_layers = [4]


    content_loss_over_time = []
    style_loss_over_time = []

    model, style_layers, content_layers = build_cnn(device, vgg_model,
        mean, std,
        im_style, im_content,
        style_layers, content_layers
    )
    
    n_epochs = 5
    for epoch in range(n_epochs):

        def closure():

            optimizer.zero_grad()
            im_target.data.clamp_(0,1)
            model(im_target)
            
            style_loss = 0
            content_loss = 0

            for layer in style_layers:
                style_loss += layer.loss
            
            for layer in content_layers:
                content_loss += layer.loss

            content_loss_over_time.append(content_loss.item())
            style_loss_over_time.append(style_loss.item())

            loss = style_loss * w_style + content_loss * w_content
            loss.backward()

            if epoch % 1 == 0:
                print("Epoch {} Complete.")
                print("Content Loss: {:4f} Style Loss: {:4f}".format(
                    content_loss.item(),
                    style_loss.item()
                ))

            return loss

        optimizer.step(closure)

    im_target.data.clamp_(0,1)

    skio.imsave(im_dir+out_dir+'bruhmom.jpg',sk.img_as_uint(im_target.cpu().squeeze(0)))

if __name__ == "__main__":
    main()