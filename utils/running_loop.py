import time
import torch

def training_loop(model, optimizer, loss_fcn, 
                  train_loader, val_loader, mask_name='lung mask', 
                  n_epochs=1, device="cpu",
                  save_path="model.pt", stop=2):
    n_stop = 0
    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        correct = 0
        model.train()
        print(f"Epoch {epoch} Start:")
        start_time = time.time()
        for batch in train_loader:
            imgs = batch['image']['data'].squeeze_(-1).to(device)
            masks = batch[mask_name]['data'].squeeze_(-1).to(device)
            output = model(imgs)

            optimizer.zero_grad()
            loss = loss_fcn(output, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.shape[0]
            _, predict_indices = torch.max(output, dim=1)
            correct += torch.sum(predict_indices == masks).item()

        train_size = len(train_loader.dataset)
        train_loss /= train_size
        train_acc = correct/train_size
        end_time = time.time()
        print("Epoch {}, train loss: {:2.3f}, train accuracy: {:2.3f} %".format(
            epoch, 
            train_loss, 
            train_acc*100))
        print(f"Duration: {(end_time - start_time)/60:2.3f} minutes")

        # validation
        val_loss = 0.0
        correct = 0
        val_acc_max = 0.0
        with torch.no_grad():
            model.eval()
            for batch in val_loader:
                imgs, masks = batch['image'].to(device), batch[mask_name].to(device)
                output = model(imgs)

                loss = loss_fcn(output, masks)
                val_loss += loss * imgs.shape[0]
                _, predict_indices = torch.max(output, dim=1)
                correct += torch.sum(predict_indices == masks).item()
            
            val_size = len(val_loader.dataset)
            val_loss /= val_size
            val_acc = correct / val_size
            print("Epoch {}, val loss: {:2.3f}, val accuracy: {:2.3f} %".format(
                epoch, 
                val_loss, 
                val_acc*100))

            if val_acc > val_acc_max:
                    val_acc_max = val_acc
                    model_scripted = torch.jit.script(model)
                    model_scripted.save(save_path)
                    print('Detect Improvement, Save Model')
            else:
                n_stop += 1
        # early stopping
        if(n_stop == stop):
            break

def test_loop(test_loader, model, loss_fcn, device="cpu"):
    loss_total = 0.0
    correct = 0
    with torch.no_grad():
        model.eval()
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, dim=1)

            correct += int((predicted == labels).sum())
            loss = loss_fcn(outputs, labels)
            loss_total += loss.item() * imgs.shape[0]

    test_size = len(test_loader.dataset)
    test_acc = 100*correct/test_size
    print('Total loss {:.4f}, Total accuracy {:2.3f}%'
            .format(loss_total/test_size, test_acc))
    return test_acc, correct, predicted, test_size