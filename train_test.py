import time 
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# performs both training and validation.  Taken from pytorch tutorial: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def train_model(model, dataloaders, dataset_sizes, device, criterion, optimizer, scheduler, writer, num_epochs=25):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # train_set = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'train':
                writer.add_scalar("Loss/train_batch", epoch_loss, epoch)
                writer.add_scalar("Accuracy/train_batch", epoch_acc, epoch)        
                print()
            
            else:
                writer.add_scalar("Loss/val_batch", epoch_loss, epoch)
                writer.add_scalar("Accuracy/val_batch", epoch_acc, epoch)

            print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model


# tests the model on the test set
def test(model, device, test_loader):
    torch.cuda.empty_cache()
    model.eval()

    correct, total = 0, 0
    total_loss = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            data, target = batch[0].to(device), batch[-1].to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            total += len(target)
            running_acc = 100 * correct / total
            total_loss += loss.item()
            running_loss = total_loss / (batch_idx + 1)
            if batch_idx % 10 == 0 or batch_idx == len(test_loader) - 1:
                print("Test [{}/{}], Loss: {:.6f}, Acc: {:.2f}".format(
                    total, len(test_loader.dataset), running_loss, running_acc))
                                



def per_class_accuracy(model, device, test_loader):
    torch.cuda.empty_cache()
    model.eval()
    n_classes = 2
    correct = 0
    confusion_matrix = torch.zeros(n_classes, n_classes)

    with torch.no_grad():
        for batch in test_loader:
            data, target = batch[0].to(device), batch[-1].to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            for t, p in zip(target.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            
        per_class_acc = 100 * (confusion_matrix.diag()/confusion_matrix.sum(1)).cpu().numpy()

        print(f'Accuracy for Normal Class: {per_class_acc[0]:.2f}%')
        print(f'Accuracy for Pneumonia Class: {per_class_acc[1]:.2f}%')
    


'''Loads a saved model from a checkpoint'''
def load_model_from_checkpoint(model: nn.Module, load_path):
    model.load_state_dict(torch.load(load_path))

