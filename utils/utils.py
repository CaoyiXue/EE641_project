import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

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

class AdjustContrast:
    def __init__(self, contrast_factor):
        self.contrast_factor = contrast_factor

    def __call__(self, x):
        return TF.adjust_contrast(x, self.contrast_factor)

def plot_one_res(data, model, mask_name="infect", device='cpu'):
    assert(mask_name in ["infect", "lung"])
    num2label = {0:'COVID-19', 1:'Non-COVID', 2:'Normal'}
    img, mask, label = data[0], data[1], num2label[data[2]]
    with torch.no_grad():
        model.eval()
        output = model(img.unsqueeze(0).to(device))
        predict = torch.zeros(output.shape)
        predict[output >= 0.5] = 1
        predict = predict.long()
    mask = mask.long()

    fig = plt.figure(figsize=(10,10))
    rows, cols = 1, 3
    ax = fig.add_subplot(rows, cols, 1)
    ax.imshow(img.squeeze_().numpy())
    ax.set_title(f"{mask_name} {label} image")
    ax = fig.add_subplot(rows, cols, 2)
    ax.imshow(mask.squeeze_().numpy())
    ax.set_title(f"{mask_name} {label} true mask")
    ax = fig.add_subplot(rows, cols, 3)
    ax.imshow(predict.squeeze_().numpy())
    ax.set_title(f"{mask_name} {label} predict mask")

    plt.show()