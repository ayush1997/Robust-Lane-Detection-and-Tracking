# 32 32 100

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
from sklearn.metrics import confusion_matrix
# from tensorboard_logger import configure, log_value
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt

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

class cnn_lstm(torch.nn.Module):
    def __init__(self,feature,hidden_unit, D_in, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(cnn_lstm, self).__init__()
        self.model_ft = models.alexnet(pretrained=True)
        # print (model_ft)

        self.num_ftrs = self.model_ft.classifier[6].in_features
        self.feature_model = list(self.model_ft.classifier.children())
        self.feature_model.pop()
        self.feature_model.pop()
        # feature_model.append(nn.Linear(num_ftrs, 3))
        self.feature_model.append(nn.Linear(self.num_ftrs, 1046))
        # self.feature_model.append(nn.Linear(self.num_ftrs, 524))
        self.feature_model.append(nn.Linear(1046, 100))
        # self.feature_model.append(nn.Linear(524, 100))

        self.model_ft.classifier = nn.Sequential(*self.feature_model)

        self.rnn = nn.LSTM(feature,hidden_unit,batch_first=True).cuda()
        self.linear = torch.nn.Linear(D_in, D_out).cuda()


    def forward(self,x):
        
        fc1 = self.model_ft(x)
        fc1 = torch.unsqueeze(fc1,2)
        # print (fc1.size())

        rnn,(_,_) = self.rnn(fc1)
        # print (rnn)
        # print (rnn[:,-1])
        y_pred = self.linear(rnn[:,-1])
        # print (y_pred)
        return y_pred

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
        # image = Image.open("/home/ayush/lane/ext_image_dual_aug/"+img_name).convert('RGB')
        image = Image.open("/home/ayush/lane/ext_image_aug/"+img_name).convert('RGB')
        label = self.tmp_paths.ix[index,1].astype(int)

        if self.transform:
            image = self.transform(image)

        
        return image, label

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.tmp_paths)

# Then, you can just use prebuilt torch's data loader.
wii_trdata = CustomDataset(csv_file='train_aug_not_aug.txt',transform=data_transforms['train'])
wii_vadata = CustomDataset(csv_file='val_aug_not_aug.txt',transform=data_transforms['val'])
# wii_trdata = CustomDataset(csv_file='train_aug.txt',transform=data_transforms['train'])
# wii_vadata = CustomDataset(csv_file='val_aug.txt',transform=data_transforms['val'])

train_loader = torch.utils.data.DataLoader(dataset=wii_trdata,
                                           batch_size=1,
                                           shuffle=True,
                                           num_workers=2)

val_loader = torch.utils.data.DataLoader(dataset=wii_vadata,
    batch_size=1,
    shuffle=True,
    num_workers=2)


dset_loaders = {'train': train_loader, 'val': val_loader}

model = torch.load("models/exp_aug.pkl")
model.train(False)


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


def eval_fn():
    eval = {"train":[],"val":[]}

    for phase in ['val', 'train']:
        # if phase == 'train':
        #     optimizer = lr_scheduler(optimizer, epoch,init_lr, delay_epoch)
        #     model.train(True)  # Set model to training mode
        # else:
        #     model.train(False)  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        tot = 0.0
        cnt = 0
        # Iterate over data.
        print (len(dset_loaders[phase]))

        pred = []
        true_pred = []
        for count,data in enumerate(dset_loaders[phase]):
            # get the inputs
            # print (data)
            print (count)
            inputs, labels = data

            # print (labels)
            # wrap them in Variable
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), \
                                 Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

       

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)

            # print (preds.cpu().numpy(),labels.cpu().data.numpy())

            pred.append(preds.cpu().numpy()[0])
            true_pred.append(labels.cpu().data.numpy()[0])

        eval[phase].append(pred)
        eval[phase].append(true_pred)

        print (phase+"done")

        file_Name = "eval"
        # open the file for writing
        fileObject = open(file_Name,'wb') 

        # this writes the object a to the
        # file named 'testfile'
        pickle.dump(eval,fileObject)  
            


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



# eval_fn()



fileObject = open("eval",'r')  
# load the object from the file into var b
b = pickle.load(fileObject)  

y_test = b["val"][0]
y_pred = b["val"][1]
class_names = ["true","false"]
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')

plt.show()
