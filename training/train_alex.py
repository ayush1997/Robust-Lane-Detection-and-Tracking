from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import models, transforms
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
import time, copy, os
from skimage import io
from PIL import Image
import pandas as pd
# from tensorboard_logger import configure, log_value

torch.manual_seed(5)
torch.cuda.manual_seed(5)

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
       # transforms.RandomSizedCrop(224),
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        # transforms.Resize(224),
        
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# You should build custom dataset as below.
class CustomDataset(Dataset):
    def __init__(self,csv_file,transform=None):
        self.tmp_paths = pd.read_csv(csv_file)
        self.transform = transform

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        img_name = self.tmp_paths.ix[index,0]
        image = Image.open("/home/ayush/lane/ext_image_aug/"+img_name).convert('RGB')
        label = self.tmp_paths.ix[index,1].astype(int)

        if self.transform:
            image = self.transform(image)

        
        return image, label

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.tmp_paths)

# Then, you can just use prebuilt torch's data loader.
wii_trdata = CustomDataset(csv_file='train_aug.txt',transform=data_transforms['train'])
wii_vadata = CustomDataset(csv_file='val_aug.txt',transform=data_transforms['val'])
# wii_trdata = CustomDataset(csv_file='train_dual_aug_new.txt',transform=data_transforms['train'])
# wii_vadata = CustomDataset(csv_file='val_dual_aug_new.txt',transform=data_transforms['val'])


# 20
train_loader = torch.utils.data.DataLoader(dataset=wii_trdata,
                                           batch_size=15,
                                           shuffle=True,
                                           num_workers=2)

val_loader = torch.utils.data.DataLoader(dataset=wii_vadata,
    batch_size=20,
    shuffle=True,
    num_workers=2)


dset_loaders = {'train': train_loader, 'val': val_loader}


use_gpu = torch.cuda.is_available()

def exp_lr_scheduler(optimizer, epoch, init_lr, lr_decay_epoch):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""

    lr = init_lr * (0.1**((epoch-1) // lr_decay_epoch))

    if (epoch-1) % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))
        # log_value('lr',lr,epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def train_model(model,criterion,optimizer, lr_scheduler, num_epochs, init_lr, delay_epoch):

    # configure('logs/'+log_dir)
    since = time.time()

    best_model = model
    best_acc = 0.0

    # log_value('val_loss', 1.0, 0)
    # log_value('val_acc',0.0, 0)
    # log_value('tr_loss', 1.0, 0)
    # log_value('tr_acc', 0.0, 0)
    # log_value('lr',0.001,1)
    train_acc = []
    val_acc = []
    for epoch in range(1,num_epochs+1):

        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        since2 = time.time()
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch,init_lr, delay_epoch)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            tot = 0.0
            cnt = 0
            # Iterate over data.
            print (len(dset_loaders[phase]))
            for data in dset_loaders[phase]:
                # get the inputs
                # print (data)
                inputs, labels = data

                # print (labels)
                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                                     Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                # print (outputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                tot += len(labels)

                if cnt % 400 == 0:
                    print('[%d, %5d] loss: %.5f, Acc: %.4f' %
                        (epoch, cnt + 1, loss.data[0], running_corrects/tot))

                cnt = cnt + 1

            epoch_loss = running_loss / len(dset_loaders[phase].dataset)
            epoch_acc = running_corrects / len(dset_loaders[phase].dataset)

            if phase == 'val':
                val_acc.append([epoch_acc,epoch_loss])
            else:
                train_acc.append([epoch_acc,epoch_loss])
        

            print('{} Loss: {:.6f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)
        time_elapsed2 = time.time() - since2
        print('Epoch complete in {:.0f}m {:.0f}s'.format(
            time_elapsed2 // 60, time_elapsed2 % 60))

        # torch.save(best_model,'models/alexnet_merged.pkl')
        # torch.save(best_model,'models/alexnet_3_08.pkl')
        torch.save(best_model,'models/alexnet_2_08.pkl')
        # torch.save(best_model,'models/wii_alexnet_icv.pkl')


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    print (train_acc)
    print (val_acc)
    return best_model


#alexnet
model_ft = models.alexnet(pretrained=True)
num_ftrs = model_ft.classifier[6].in_features
feature_model = list(model_ft.classifier.children())
feature_model.pop()
# feature_model.append(nn.Linear(num_ftrs, 3))
feature_model.append(nn.Linear(num_ftrs, 2))

model_ft.classifier = nn.Sequential(*feature_model)

print (model_ft)

if use_gpu:
    model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()

init_lr = 0.001
# optimizer_ft = optim.SGD(model_ft.parameters(), init_lr, momentum=0.9, weight_decay=0.0001)
optimizer_ft = optim.SGD(model_ft.parameters(), init_lr, momentum=0.9, weight_decay=0.0001)

# 30
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=40, init_lr=init_lr, delay_epoch=12)
# model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=45, init_lr=init_lr, delay_epoch=40)
