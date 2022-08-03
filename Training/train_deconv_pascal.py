# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 13:45:09 2021

@author: Dell
"""

import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import vocset
from torch.utils.data import Dataset
from torchvision import transforms
from modeldn import Conv_Deconv
from deco import conv_deconv
import torch.nn as nn
#torch.set_printoptions(profile="full")

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

parser = argparse.ArgumentParser()
LookupChoices = type('', (argparse.Action, ), dict(__call__ = lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))
parser.add_argument('--dataset_year', choices = dict(pascal_2011="2011", pascal_2012="2012"), default = "2012", action = LookupChoices)
parser.add_argument('--data', default = './data')
parser.add_argument('--epochs', default = 20, type = int )
parser.add_argument('--batch_size', default = 4, type = int)
parser.add_argument('--input_size', default = 224, type = int)
parser.add_argument('--cuda', default = False, type = str2bool)
parser.add_argument('--check_every', default = 10, type = int)
parser.add_argument('--save_every', default = 10, type = int)

opts    = parser.parse_args()
device  = torch.device("cuda:0" if opts.cuda else "cpu")
kwargs  = {'num_workers': 1, 'pin_memory': True} if opts.cuda else {}

# normalize filter values to 0-1 so we can visualize them
def NormalizeImg(img):
    nimg = (img - img.min()) / (img.max() - img.min())
    return nimg

def show_MNIST(img):
    grid = torchvision.utils.make_grid(img)
    trimg = grid.numpy().transpose(1, 2, 0)
    plt.imshow(trimg)
    plt.title('Batch from dataloader')
    plt.axis('off')
    plt.show()

def train(model, num_epoch, dataset_train, train_loader, train_size, val_loader, val_size, optimizer):
    count       = 0
    save_path   = "C:/Users/Dell/Desktop/results"

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        os.mkdir(save_path + "/checkpoints")
        os.mkdir(save_path + "/saved_images")

    for epoch in range(num_epoch):
        print("\nEPOCH " + str(epoch) + " of " + str(num_epoch))
        epoch_loss = 0
        model.train()
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            #inputs, labels = [torch.autograd.Variable(tensor.to(device)) for tensor in batch]
            inputs = inputs.to(device)
            labels = labels.to(device)
            #labels=(labels,dtype=torch.long)
           # print(inputs)
            # Feed the network with the datas
            #show_MNIST(inputs[0])
            #show_MNIST(labels[0])
            #print(type(labels))
            labels = torch.tensor(labels).to(dtype=torch.long)
            output, score = model(inputs)
           
            print(score.shape)
            print(labels.shape)
            #score = torch.tensor(score).to(dtype=torch.long)
            loss = model.lossfunction(score, labels)
            #loss = torch.nn.CrossEntropyLoss(score, labels)
            if loss > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            print('Loss: {:.6f}'.format(loss.item()/len(inputs)))

            # Saving images to have visual results
            if batch_idx % 200 == 0:
                img_label   = transforms.ToPILImage()(dataset_train.decode_segmap(np.array(labels[0].detach().cpu())).astype(np.uint8))
                img_label.save(save_path + "/saved_images/" + str(count) + "_label_" + str(epoch) + "_" + str(batch_idx) + ".png")

                nimg        = NormalizeImg(output[0])
                img_output  = transforms.ToPILImage()(nimg)
                img_output.save(save_path + "/saved_images/" + str(count) + "_output_" + str(epoch) + "_" + str(batch_idx) + ".png")

                '''img_dec     = torch.argmax(score[0], dim=0, keepdim=True).squeeze().detach().cpu()
                img_dec     = dataset_train.decode_segmap(np.array(img_dec)).astype(np.uint8)
                img_score   = transforms.ToPILImage()(img_dec)
                img_score.save(save_path + "/saved_images/" + str(count) + "_score_" + str(epoch) + "_" + str(batch_idx) + ".png")'''

                count += 1

        scheduler.step()
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, epoch_loss / train_size))

################################test#################################
        if epoch % check_every == 0:
            model.eval()
            loss_validation = 0
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = torch.tensor(labels).to(dtype=torch.long)
                with torch.no_grad():
                    output, score = model(inputs)

                loss = model.lossfunction(score, labels)
                loss_validation += loss.item()
            print('====> Epoch: {} Validation Loss: {:.4f}'.format(epoch, loss_validation / val_size))

################################save#################################
        if epoch % save_every == 0:
            torch.save(model.state_dict(), save_path + '/checkpoints/model_epoch_{}.pt'.format(epoch))
            print("checkpoint: saved model")


data_path   = opts.data
year        = opts.dataset_year
check_every = opts.check_every
save_every  = opts.save_every
num_epoch   = opts.epochs

	## Transformations for datasets
size            = opts.input_size
transform       = transforms.Compose([transforms.Resize((size, size)),transforms.ToTensor()])
transform_label = transforms.Compose([transforms.Resize((size, size)),transforms.ToTensor()]) #NO MORE NEEDED CAUSE ITS DONE IN VOC.PY

dataset_train   = vocset.VOCSegmentation(root=data_path, year=year, image_set="train", transform = transform, target_transform=transform_label)
dataset_val     = vocset.VOCSegmentation(root=data_path, year=year, image_set="val", transform = transform, target_transform=transform_label)
train_size      = len(dataset_train)
val_size        = len(dataset_val)

train_loader    = torch.utils.data.DataLoader(dataset_train,    batch_size=opts.batch_size, shuffle=True, drop_last=True, **kwargs)
val_loader      = torch.utils.data.DataLoader(dataset_val,      batch_size=opts.batch_size, shuffle=True, drop_last=True, **kwargs)

# Create the model before the transformation to get the size
model = Conv_Deconv()
model.to(device)

print("Number of training images:", len(dataset_train))
print("Number of validation images:", len(dataset_val))
print("Model name:",    model.name)

#dataset_train.show_color_map() #show voc color classes
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
train(model, num_epoch, dataset_train, train_loader, train_size, val_loader, val_size, optimizer)
print('all done')