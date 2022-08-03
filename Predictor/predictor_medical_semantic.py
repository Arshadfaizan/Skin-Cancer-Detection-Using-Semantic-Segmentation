
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 14:20:49 2021

@author: Dell
"""
import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import medical
from torchvision import transforms
from modeldn import Conv_Deconv
from unet_arc import UNet
from pytorch_segnet_2nd import SegNet
from dpl import ASPP
def show_MNIST(img):
    grid    = torchvision.utils.make_grid(img)
    trimg   = grid.detach().numpy().transpose(1, 2, 0)
    plt.imshow(trimg)
    plt.title('Batch from dataloader')
    plt.axis('off')
    plt.show()
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

parser = argparse.ArgumentParser()
LookupChoices = type('', (argparse.Action, ), dict(__call__ = lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))
parser.add_argument('--dataset_year', choices = dict(pascal_2011="2011", pascal_2012="2012"), default = "2012", action = LookupChoices)
parser.add_argument('--data', default = './data')
parser.add_argument('--epochs', default = 2, type = int )
parser.add_argument('--batch_size', default = 4, type = int)
parser.add_argument('--input_size', default = 224, type = int)
parser.add_argument('--cuda', default = False, type = str2bool)
parser.add_argument('--check_every', default = 0, type = int)
parser.add_argument('--save_every', default = 1, type = int)
opts    = parser.parse_args()
device  = torch.device("cuda:0" if opts.cuda else "cpu")
kwargs  = {'num_workers': 1, 'pin_memory': True} if opts.cuda else {}

data_path   = opts.data
year        = opts.dataset_year
check_every = opts.check_every
save_every  = opts.save_every
num_epoch   = opts.epochs
save_path   = "C:/Users/Dell/Desktop/ars_deeplab"

if not os.path.isdir(save_path):
    os.mkdir(save_path)
    os.mkdir(save_path + "/pred_images")
size            = opts.input_size
transform       = transforms.Compose([transforms.Resize((size, size)),transforms.ToTensor()])
transform_label = transforms.Compose([transforms.Resize((size, size)),transforms.ToTensor()]) #NO MORE NEEDED CAUSE ITS DONE IN VOC.PY
dataset_train   = medical.VOCSegmentation(root=data_path, image_set="train", transform = transform, target_transform=transform_label)
dataset_val     = medical.VOCSegmentation(root=data_path, image_set="validation", transform = transform, target_transform=transform_label)
val_size        = len(dataset_val)
val_loader      = torch.utils.data.DataLoader(dataset_val,      batch_size=opts.batch_size, shuffle=True, drop_last=True, **kwargs)
PATH='C:/Users/Dell/Desktop/prediction-deeplabv3/checkpoints/model_epoch_5.pt'
model=ASPP()
model.load_state_dict(torch.load(PATH))
model.eval()
loss_validation=0
count=0
accuracy1=0
total=0
correct=0
eval_losses=[]
eval_accu=[]
for batch_idx, (inputs, labels) in enumerate(val_loader):
    count=count+1
    inputs = inputs.to(device)
    labels = labels.to(device)
    labels = torch.tensor(labels).to(dtype=torch.long)
    #show_MNIST(inputs[0])
    with torch.no_grad():
        score = model(inputs)
    '''
    ps = torch.exp(score)
    equality = (labels.data == ps.max(dim=1)[1])
    accuracy1 += equality.type(torch.FloatTensor).mean()
    #print("validation Accuracy: {:.3f}".format(accuracy1))
    '''
    show_MNIST(inputs[0])
    torchvision.utils.save_image(inputs[0],save_path + "/pred_images/" + "input_" + str(count) + ".jpg")
    img_dec     = torch.argmax(score[0], dim=0, keepdim=True).squeeze().detach().cpu()
    img_dec     = dataset_train.decode_segmap(np.array(img_dec)).astype(np.uint8)
    img_score   = transforms.ToPILImage()(img_dec)

    img_score.save(save_path + "/pred_images/"  + "predict_" +str(count) + ".png")  
    loss = model.lossfunction(score, labels)
    loss_validation += loss.item()
    _, predicted = score.max(1)
    total += labels.size(0)
    correct += predicted.eq(labels).sum().item()
test_loss=loss_validation/len(val_loader)
accu=100.*correct/total
eval_losses.append(test_loss)
eval_accu.append(accu)
print('Test Loss: %.3f | Accuracy: %.3f'%(test_loss,accu))
print('====>  Validation Loss: {:.4f}'.format( loss_validation / val_size))
