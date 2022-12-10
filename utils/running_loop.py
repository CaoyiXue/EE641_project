import torch
from .utils import *
import torch.optim.lr_scheduler as lr_scheduler

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
        
def test_loop(model, test_loader, mask_name='infection mask', device="cpu"):

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

            acc_tensor.append(accuracy(predict, mask))
            IoU_tensor.append(IoU(predict, mask))
            DSC_tensor.append(DSC(predict, mask))

    acc_tensor = torch.tensor(acc_tensor).to(device)
    IoU_tensor = torch.tensor(IoU_tensor).to(device)
    DSC_tensor = torch.tensor(DSC_tensor).to(device)
    acc_mean, acc_std = round(acc_tensor.mean().item()*100,2), round(acc_tensor.std().item()*100,2)
    IoU_mean, IoU_std = round(IoU_tensor.mean().item()*100,2), round(IoU_tensor.std().item()*100,2)
    DSC_mean, DSC_std = round(DSC_tensor.mean().item()*100,2), round(DSC_tensor.std().item()*100,2)
    print(f"accuracy: {acc_mean}% ± {acc_std}%")
    print(f"IoU: {IoU_mean}% ± {IoU_std}%")
    print(f"DSC: {DSC_mean}% ± {DSC_std}%")
    res = (acc_mean, acc_std, IoU_mean, IoU_std, DSC_mean, DSC_std)
    
    return res

# def COVID_detection(lung_model, infect_model, test_loader, device='cpu'):
#     assert(test_loader.batch_size==1)
#     correct = 0
#     with torch.no_grad():
#         lung_model.eval(), infect_model.eval()
#         for batch_i, batch in enumerate(test_loader):
#             imgs = batch['image'][tio.DATA].squeeze_(-1).to(device)
#             label = batch['category'] == 'COVID-19'
#             predict = infection_rate(lung_model(imgs), infect_model(imgs)) > 0
#             if label == predict:
#                 correct += 1
#     acc = correct / len(test_loader.dataset)
#     print(f"COVID_19 detection accuracy: {acc}")
#     # TO-DO: precision, recall ....
#     return acc


