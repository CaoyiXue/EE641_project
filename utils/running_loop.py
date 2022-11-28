import time
import torch
import torchio as tio
import torch.nn.functional as F
#label2num = {'COVID-19': 0, 'Non-COVID': 1, 'Normal': 2}
def training_loop(model, optimizer, loss_fcn, 
                  train_loader, val_loader, mask_name='infection mask',
                  n_epochs=1, device="cpu",
                  save_path="model.pt", stop=2):
    
    n_stop = 0
    val_loss_min = torch.inf
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        model.train()
        print(f"Epoch {epoch} Training Start:")
        start_time = time.time()
        for batch_i, (imgs, masks, labels) in enumerate(train_loader):

            if batch_i % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_i}")
            imgs = imgs.to(device)
            masks = masks.to(device)
            output = model(imgs)

            optimizer.zero_grad()
            loss = loss_fcn(output, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.shape[0]
        
        train_loss /= train_size
        end_time = time.time()
        print("Epoch {}, train loss: {:2.3f}".format(
            epoch, 
            train_loss))
        print(f"Duration: {(end_time - start_time)/60:2.3f} minutes")

        # validation
        val_loss = 0.0
        print(f"Epoch {epoch} Validation Start:")
        with torch.no_grad():
            model.eval()
            for batch_i, (imgs, masks, labels) in enumerate(val_loader):

                if batch_i % 100 == 0:
                    print(f"Epoch {epoch}, Batch {batch_i}")

                imgs = imgs.to(device)
                masks = masks.to(device)
                output = model(imgs)

                loss = loss_fcn(output, masks)
                val_loss += loss * imgs.shape[0]
        
            val_loss /= val_size
            print("Epoch {}, val loss: {:2.3f}".format(
                epoch, 
                val_loss))

            if val_loss < val_loss_min:
                    val_loss_min = val_loss
                    torch.save(model, save_path)
                    print('Detect Improvement, Save Model')
            else:
                n_stop += 1
        # early stopping
        if(n_stop == stop):
            break

def accuracy(pred, real):
    pred = pred.view(-1)
    real = real.view(-1)
    acc = (pred==real).sum()
    return acc/len(real)

def IoU(pred, real):
    pred = pred.view(-1)
    real = real.view(-1)
    intersection = (pred * real).sum()
    total = (pred + real).sum()
    union = total - intersection
    if union == 0:
        return 0.0
    return intersection/union

def IoUloss(pred, real):
    pred = pred.view(-1)
    real = real.view(-1)
    intersection = (pred * real).sum()
    total = (pred + real).sum()
    union = total - intersection
    if union == 0:
        return 0.0
    return 1-intersection/union

def DSC(pred, real):
    pred = pred.view(-1)
    real = real.view(-1)
    intersection = (pred * real).sum()
    if (pred.sum()+real.sum()) == 0:
        return 0.0
    return (2.*intersection)/(pred.sum()+real.sum())

class DiceBCELoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

def infection_rate(lung_mask, infection_mask):
    lung_mask = lung_mask.view(-1)
    infection_mask = infection_mask.view(-1)
    infect = (lung_mask * infection_mask).sum()
    return infect / lung_mask.sum()

def test_loop(model, test_loader, mask_name='infection mask', device="cpu"):

    assert(test_loader.batch_size==1)
    assert(mask_name in ['lung mask', 'infection mask'])
    acc_tensor = []
    IoU_tensor = []
    DSC_tensor = []

    with torch.no_grad():
        model.eval()
        for batch_i, (img, mask, label) in enumerate(test_loader):

            if mask_name == 'infection mask' and label != 0:
                continue
            
            img = img.to(device)
            mask = mask.to(device)
            outputs = model(img)
            outputs = torch.sigmoid(outputs)            
            predicted = torch.zeros(outputs.shape)
            predicted[outputs > 0.5] = 1

            acc_tensor.append(accuracy(predicted, mask))
            IoU_tensor.append(IoU(predicted, mask))
            DSC_tensor.append(DSC(predicted, mask))

    acc_tensor = torch.tensor(acc_tensor).to(device)
    IoU_tensor = torch.tensor(IoU_tensor).to(device)
    DSC_tensor = torch.tensor(DSC_tensor).to(device)
    acc_mean, acc_std = acc_tensor.mean().item(), acc_tensor.std().item()
    IoU_mean, IoU_std = IoU_tensor.mean().item(), IoU_tensor.std().item()
    DSC_mean, DSC_std = DSC_tensor.mean().item(), DSC_tensor.std().item()
    print(f"accuracy mean: {acc_mean:.3f}, accuracy standard deviation: {acc_std:.3f}")
    print(f"IoU mean: {IoU_mean:.3f}, IoU standard deviation: {IoU_std:.3f}")
    print(f"DSC mean: {DSC_mean:.3f}, DSC standard deviation: {DSC_std:.3f}")
    res = (acc_mean, acc_std, IoU_mean, IoU_std, DSC_mean, DSC_std)
    
    return res

def COVID_detection(lung_model, infect_model, test_loader, device='cpu'):
    assert(test_loader.batch_size==1)
    correct = 0
    with torch.no_grad():
        lung_model.eval(), infect_model.eval()
        for batch_i, batch in enumerate(test_loader):
            imgs = batch['image'][tio.DATA].squeeze_(-1).to(device)
            label = batch['category'] == 'COVID-19'
            predict = infection_rate(lung_model(imgs), infect_model(imgs)) > 0
            if label == predict:
                correct += 1
    acc = correct / len(test_loader.dataset)
    print(f"COVID_19 detection accuracy: {acc}")
    # TO-DO: precision, recall ....
    return acc