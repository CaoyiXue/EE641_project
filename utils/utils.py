import torch.nn.functional as F
import torch
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from skimage.util import random_noise
import numpy as np

def accuracy(pred, real):
    pred = pred.view(-1)
    real = real.view(-1)
    acc = (pred==real).sum()
    return acc/len(real)

def IoU(pred, real, smooth=1):
    pred = pred.view(-1)
    real = real.view(-1)
    intersection = (pred * real).sum()
    total = (pred + real).sum()
    union = total - intersection
    return (intersection+smooth)/(union+smooth)

def DSC(pred, real, smooth=1):
    pred = pred.view(-1)
    real = real.view(-1)
    intersection = (pred * real).sum()
    return (2.*intersection+smooth)/(pred.sum()+real.sum()+smooth)

def infection_rate(lung_mask, infection_mask):
    lung_mask = lung_mask.view(-1)
    infection_mask = infection_mask.view(-1)
    infect = (lung_mask * infection_mask).sum()
    return infect / lung_mask.sum()

class DiceBCELoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
        
class IoUDiceBCELoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        total = (inputs + targets).sum()
        union = total - intersection
        
        IoU_loss = 1 - (intersection + smooth)/(union + smooth)
        dice_loss = 1 - (2.*intersection + smooth)/(total + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        res = BCE + dice_loss + IoU_loss
        
        return res

class IoULoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU
        
class AdjustContrast:
    def __init__(self, contrast_factor):
        self.contrast_factor = contrast_factor

    def __call__(self, x):
        return TF.adjust_contrast(x, self.contrast_factor)
        
class gaussian_noise():
    def __init__(self, mean, stddev):
       self.mean = mean
       self.stddev = stddev
    def __call__(self, img):
       gauss_img = random_noise(img, mode='gaussian', mean=self.mean, var=self.stddev, clip=True)
       return torch.tensor(gauss_img, dtype=torch.float32)
       
class salt_pepper_noise():
    def __init__(self, amount):
      self.amount = amount
    def __call__(self, img):
       sp_img = random_noise(img, mode='s&p', amount=self.amount)
       return torch.tensor(sp_img, dtype=torch.float32)
       
def mark_boundaries(image, infect_label_img, lung_label_img, mode='outer', background_label=0):

    marked = image.copy().astype('float64') / 255
    marked = np.stack(3 * (marked,), axis=-1)
    boundaries = find_boundaries(lung_label_img, mode=mode, background=background_label)
    marked[boundaries] = (0,1,0)
    boundaries = find_boundaries(infect_label_img, mode=mode, background=background_label)
    marked[boundaries] = (1,0,0)
    return marked
    
def infection_rate(infect_mask, lung_mask):
    infect = (lung_mask * infect_mask).sum()
    return np.round((infect / lung_mask.sum())*100, 2)

def plot_one_model_one_img(img, ax, encoder_name, device='cpu'):
    input = torch.from_numpy(img.copy()).unsqueeze(0) / 255.0
    test_img_transforms = transforms.Compose([transforms.Normalize((0.5302),(0.2452))])
    input = test_img_transforms(input).unsqueeze(0)
    infect_model = torch.load(f'trained_models/infect_Unet_{encoder_name}.pt', map_location=device)
    lung_model = torch.load(f'trained_models/lung_Unet_{encoder_name}.pt', map_location=device)
    with torch.no_grad():
        infect_model.eval(), lung_model.eval()
        output = infect_model(input.to(device))
        infect_predict = torch.zeros(output.shape)
        infect_predict[output >= 0.5] = 1
        infect_predict = infect_predict.long().squeeze().numpy()

        output = lung_model(input.to(device))
        lung_predict = torch.zeros(output.shape)
        lung_predict[output >= 0.5] = 1
        lung_predict = lung_predict.long().squeeze().numpy()
    marked = mark_boundaries(img, infect_predict, lung_predict)
    ax.imshow(marked)
    ax.axis('off')
    dict_name = {'resnet18':'ResNet18','mobilenet_v2':'MobileNet V2','micronet_m0':'MicroNet M0',
             'micronet_m1':'MicroNet M1','micronet_m2':'MicroNet M2','micronet_m3':'MicroNet M3'}
    ax.set_title(f"{dict_name[encoder_name]} Infection Rate: {infection_rate(infect_predict, lung_predict)}%")

def plot_one_image(data, device='cpu'):
    fig = plt.figure(figsize=(15,15))
    img, infect_mask, lung_mask = data[0], data[1], data[2]
    name_list = ['resnet18', 'mobilenet_v2']
    name_list.extend([f'micronet_m{i}' for i in range(4)])
    for i in range(9):
        if i == 1:
            ax = fig.add_subplot(3, 3, i+1)
            marked = mark_boundaries(img, infect_mask, lung_mask)
            ax.imshow(marked)
            ax.axis('off')
            ax.set_title(f"True Infection Rate: {infection_rate(infect_mask, lung_mask)}%")
        elif i > 2:
            ax = fig.add_subplot(3, 3, i+1)
            plot_one_model_one_img(img, ax, name_list[i-3], device)
    plt.show()
