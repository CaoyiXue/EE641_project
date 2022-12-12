import torch
from .utils import *
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from sklearn.metrics import confusion_matrix

#label2num = {'COVID-19': 0, 'Non-COVID': 1, 'Normal': 2}
def training_loop(model, optimizer, loss_fcn, 
                  train_loader, val_loader, scheduler=None,
                  mask_name='infection mask',
                  n_epochs=1, device="cpu",
                  save_path="model.pt", stop=None):
    
    n_stop = 0
    train_loss_list = []
    val_loss_list = []
    val_loss_min = torch.inf

    iters = len(train_loader)
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    for epoch in range(n_epochs):
        train_loss = 0.0
        model.train()
        for batch_i, (imgs, masks, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            masks = masks.to(device)
            output = model(imgs)

            optimizer.zero_grad()
            loss = loss_fcn(output, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.shape[0]

            if isinstance(scheduler, lr_scheduler.CosineAnnealingWarmRestarts):
                scheduler.step(epoch + batch_i / iters)
                
        if isinstance(scheduler, lr_scheduler.CosineAnnealingWarmRestarts):
            pass        
        elif scheduler is not None:
            scheduler.step()

        
        train_loss /= train_size
        train_loss_list.append(train_loss)
        print("Epoch {}, train loss: {:2.3f}".format(
            epoch, 
            train_loss))

        # validation
        val_loss = 0.0
        with torch.no_grad():
            model.eval()
            for batch_i, (imgs, masks, labels) in enumerate(val_loader):
                imgs = imgs.to(device)
                masks = masks.to(device)
                output = model(imgs)

                loss = loss_fcn(output, masks)
                val_loss += loss.item() * imgs.shape[0]
        
            val_loss /= val_size
            val_loss_list.append(val_loss)
            print("Epoch {}, val loss: {:2.3f}".format(
                epoch, 
                val_loss))

            if val_loss < val_loss_min:
                    n_stop = 0
                    val_loss_min = val_loss
                    torch.save(model, save_path)
                    print('Detect Improvement, Save Model')
            else:
                n_stop += 1
        # early stopping
        if(stop is not None and n_stop == stop):
            break
            
    return train_loss_list, val_loss_list
        
def test_loop(model, encoder_name, test_loader, mask_name='infection mask', device="cpu"):

    assert(test_loader.batch_size==1)
    assert(mask_name in ['lung mask', 'infection mask'])
    acc_tensor = []
    IoU_tensor = []
    DSC_tensor = []

    with torch.no_grad():
        model.eval()
        for batch_i, (img, mask, label) in enumerate(test_loader):
            
            img = img.to(device)
            mask = mask.to(device)
            outputs = model(img)
            # outputs = torch.sigmoid(outputs)            
            predict = torch.zeros(outputs.shape).to(device)
            predict[outputs >= 0.5] = 1
            predict = predict.long()
            mask = mask.long()

            acc_tensor.append(accuracy(predict, mask)*100)
            IoU_tensor.append(IoU(predict, mask)*100)
            DSC_tensor.append(DSC(predict, mask)*100)

    acc_tensor = torch.tensor(acc_tensor).to(device)
    IoU_tensor = torch.tensor(IoU_tensor).to(device)
    DSC_tensor = torch.tensor(DSC_tensor).to(device)
    acc_mean, acc_std = round(acc_tensor.mean().item(),2), round(acc_tensor.std().item(),2)
    IoU_mean, IoU_std = round(IoU_tensor.mean().item(),2), round(IoU_tensor.std().item(),2)
    DSC_mean, DSC_std = round(DSC_tensor.mean().item(),2), round(DSC_tensor.std().item(),2)
    print(f"{encoder_name} & {acc_mean} ± {acc_std} & {IoU_mean} ± {IoU_std} & {DSC_mean} ± {DSC_std}")
    res = (acc_mean, acc_std, IoU_mean, IoU_std, DSC_mean, DSC_std)
    
    return res


def COVID_detection(infect_model, encoder_name, test_loader, device='cpu'):
    assert(test_loader.batch_size == 1)
    correct = 0
    true_label = np.zeros(len(test_loader.dataset))
    predict_label = np.zeros(len(test_loader.dataset))
    with torch.no_grad():
        infect_model.eval()
        for batch_i, (img, mask, label) in enumerate(test_loader):
            if label != 0:
                true_label[batch_i] = 1
            img = img.to(device)
            output = infect_model(img)
            if (output >= 0.5).sum() == 0:
                predict_label[batch_i] = 1

    cm = confusion_matrix(y_true=true_label, y_pred=predict_label, labels=[0, 1])
    TP = np.diag(cm) # True Positive
    FP = cm.sum(axis=0) - TP  # False Positive
    FN = cm.sum(axis=1) - TP # False Negative
    TN = cm.sum() - (FP + FN + TP) # True Negative

    accuracy = (TP+TN)/(TP+FP+FN+TN)
    # True positive rate, sensitivity, hit rate or recall
    sensitivity = TP/(TP+FN)
    # True negative rate or specificity
    specificity = TN/(TN+FP)
    # Precision or positive predictive value
    precision = TP/(TP+FP)
    # F1 score
    F1 = 2*precision*sensitivity/(precision+sensitivity)
    accuracy = np.round(accuracy[0]*100,2)
    sensitivity = np.round(sensitivity[0]*100,2)
    specificity = np.round(specificity[0]*100,2)
    precision = np.round(precision[0]*100,2)
    F1 = np.round(F1[0]*100,2)

    print(f"{encoder_name} & {accuracy} & {sensitivity} & {specificity} & {precision} & {F1}")

    return 


