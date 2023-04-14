import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import numpy as np
import math
import json
import os
import time
import random
import torch
import torch.utils.data as data
from torchvision import transforms
from pathlib import Path
from torch import nn, einsum
from torch.nn import functional as F
import torchvision as tv
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from PIL import Image
from tqdm.notebook import tqdm
from icecream import ic
import matplotlib.pyplot as plt
import pickle
import zipfile
import opt

from model_2_pretrained_2_layers import ViViT_2 as model_2_pretrained_2_layers
from model_pretrained_all_layers import ViViT_2 as model_2_pretrained_all_layers
from model_2_scratch import ViViT as model_2_scratch
from checkpoint_saver import CheckpointSaver
from confusion_matrix import plot_confuse_matrix,add_cm_to_tb,plot_confusion_matrix_diagonal,ConfusionMatrix
from contrastive_loss import ContrastiveLoss



# set device
seed = 400
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


############################# Driver functions for driving the loop
class Driver:
    def train_step(loader,epoch):

        model.train()
        num_videos = 0
        total_epoch_loss=0
        corrects=torch.zeros(1, num_classes).to(device)

        for batch_id, (video_data,labels) in enumerate(loader):
            video_data,labels = video_data.to(device), labels.to(device)

            optimizer.zero_grad()
            prediction = model(video_data)

            loss = loss_criterion(prediction,labels)
            total_epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            prediction = torch.argmax(prediction,dim=1)
            for c in range(num_classes):
                corrects[0,c] += (prediction[labels==c]==c).sum()
            num_videos += video_data.size(0)

            tb_writer.add_scalar("Train Loss/Minibatch",loss.item(),((len(loader))*(epoch-1))+ batch_id)

        tb_writer.add_scalar("Train Loss/Epochs", total_epoch_loss/len(loader),epoch)
        accuracy = corrects.sum()/num_videos*100.0
        #tb_writer.add_scalar("Train Accuracy/Epochs",accuracy)
        print("Train Loss/Epochs: {:06.5f}/epoch {:d} ".format(total_epoch_loss/len(loader),epoch))
        print("Train Accuracy: {:05.5f}, over {:f}/{:d} vsamples and {:d} classes".format(accuracy, corrects.sum(), num_videos, (corrects>0).sum()), flush=True)

        return total_epoch_loss/len(loader)  
    
    # validation step for every epoch
    def val_step(loader,epoch):
        model.eval()

        num_videos = 0
        total_loss=0
        corrects=0
        corrects=torch.zeros(1, num_classes).to(device)

        #Intialized the confusion matrix
        confusion_matrix = np.zeros((num_classes,num_classes))

        with torch.no_grad():
            for batch_id, (video_data,labels) in enumerate(loader):
                video_data,labels = video_data.to(device), labels.to(device)
                prediction = model(video_data)
                loss = loss_criterion(prediction,labels)

                total_loss += loss.item()

                # calculating the accuracy per class
                prediction_1 = torch.argmax(prediction,dim=1)
                for c in range(num_classes):
                    corrects[0,c] += (prediction_1[labels==c]==c).sum()
                num_videos += video_data.size(0)

                #update the confusion matrix
                _,max_preds= torch.max(prediction, dim=1)
                for t, p in zip(labels.view(-1), max_preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

        accuracy = corrects.sum()/num_videos*100.0
        epoch_loss = total_loss/(len(loader))
        #accuracy = corrects/(len(loader)*test_batch_size)
        tb_writer.add_scalar("Validation/Loss",total_loss/len(loader),epoch)
        tb_writer.add_scalar("Validation/Accuracy",accuracy,epoch)
        print("Validation Accuracy: {:05.5f}, over {:f}/{:d} vsamples and {:d} classes".format(accuracy, corrects.sum(), num_videos, (corrects>0).sum()), flush=True)
        if opt.checkpoint_flag:
            checkpoint_saver(model,epoch,total_loss/len(loader))

        return accuracy, epoch_loss, confusion_matrix
    
    # Test step for each epoch
    def test_model(loader,epoch):

        num_videos = 0
        model.eval()
        #corrects=0
        corrects=torch.zeros(1, num_classes).to(device)

        #Intialized the confusion matrix
        confusion_matrix = np.zeros((num_classes,num_classes))
        with torch.no_grad():
            for batch_id, (video_data,labels) in enumerate(loader):

                video_data,labels = (video_data).to(device), labels.to(device)

                prediction = model(video_data)
                #loss = loss_criterion(prediction,labels)
                #total_loss += loss.item()
                #corrects+= (torch.argmax(prediction,dim=1)==labels).sum()

                # calculating the accuracy per class
                prediction_1 = torch.argmax(prediction,dim=1)
                for c in range(num_classes):
                    corrects[0,c] += (prediction_1[labels==c]==c).sum()
                num_videos += video_data.size(0)

                #update the confusion matrix
                _,max_preds= torch.max(prediction, dim=1)
                for t, p in zip(labels.view(-1), max_preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

        accuracy = corrects.sum()/num_videos*100.0
        #accuracy = corrects/(len(loader)*test_batch_size)
        print("Test Accuracy: {:05.5f}, over {:f}/{:d} vsamples and {:d} classes".format(accuracy, corrects.sum(), num_videos, (corrects>0).sum()), flush=True)
        tb_writer.add_scalar("Test/Accuracy",accuracy,epoch)
        return accuracy,confusion_matrix 

# ##################################################################

############################# model initialisation
tb_writer = SummaryWriter()
device ='cuda:0' if torch.cuda.is_available() else 'cpu'
opt = opt.parse_opt()

############################# All parameters
batch_size = opt.batch_size # set to 64
test_batch_size = opt.test_batch_size
num_classes = opt.num_classes
patch_size = opt.patch_size # lower value causing memory issue
frames_per_clip = opt.frames_per_clip # set to 32
stride = 2
lr = opt.lr # set to 0.001
epochs = opt.epochs

loss_criterion = nn.CrossEntropyLoss()
dataset_dir = opt.dataset_dir
annotation_path = opt.annot_dir
train_perc = opt.tt_split # 80% as training , 20% as validation



#Dataset Intialization
if opt.dataset == 'UCF101':
    from ucf_dataset import UCFDataset,get_ucf101_class_length
    train_val_data = UCFDataset( dataset_dir = dataset_dir, subset="train", video_list_file="trainlist01.txt",frames_per_clip=frames_per_clip)
    train_len=int(opt.tt_split*len(train_val_data))
    train_val_split = [ train_len, len(train_val_data) - train_len ]

    train_data , val_data = random_split(train_val_data,train_val_split)
    test_data = UCFDataset(dataset_dir = dataset_dir, subset="test", video_list_file="testlist01.txt" ,frames_per_clip=frames_per_clip)

    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

else:
    from anet_dataset_modified import ActivityNet
    train_data = ActivityNet(root_path = dataset_dir, annotation_path = annotation_path, subset = 'training', num_frames = frames_per_clip)

    class_names = train_data.class_names # saving class names
    class_length = json.load(open(annotation_path))['class-lengths'] if 'class-lengths' in json.load(open(annotation_path)).keys() else {} # saving length of all classes
    train_data, val_data = random_split(train_data, [ round(len(train_data)*train_perc), len(train_data) - round(len(train_data)*train_perc) ])

    test_data = ActivityNet(root_path = dataset_dir, annotation_path = annotation_path, subset = 'validation', num_frames = frames_per_clip)

    print("Train data length : ", len(train_data), "shape : " , next(iter(train_data))[0].shape)
    print("Val data length :", len(val_data), "shape : " , next(iter(val_data))[0].shape)
    print("Test data length :", len(test_data), "shape : " , next(iter(test_data))[0].shape)


#Initialising Data-loaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=test_batch_size)
test_loader = DataLoader(test_data, batch_size=test_batch_size)

# initialize model and plot on tensorboard
if opt.pr == 1:
    model = model_2_pretrained_2_layers(image_size= opt.image_size, patch_size=patch_size, num_classes=num_classes, frames_per_clip=frames_per_clip,tube = True)
elif opt.pr == 0:
    model = model_2_scratch(image_size= opt.image_size, patch_size=patch_size, num_classes=num_classes, frames_per_clip=frames_per_clip)
elif opt.pr == 2:
    model = model_2_pretrained_all_layers(image_size= opt.image_size, patch_size=patch_size, num_classes=num_classes, frames_per_clip=frames_per_clip, tube = False, dropout = 0 ,emb_dropout = 0 )

# if using the all layered pretriained model

checkpoint = torch.load("../pretrain_weights/pre_32f.pt",map_location=torch.device('cuda:0'))
unmatched = model.load_state_dict(checkpoint,strict = False)
for i in unmatched.missing_keys:
    print(i)

frames, _ = next(iter(train_loader))
#tb_writer.add_graph(model, frames)
model.to(device)

# define the loss and optimizers
loss_criterion = nn.CrossEntropyLoss()
#loss_criterion = ContrastiveLoss()

optimizer = torch.optim.Adam(model.parameters(),lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

# Intialising the checkpoint saver
if opt.checkpoint_flag: 
    checkpoint_saver = CheckpointSaver(dirpath='./model_weights7', decreasing=True, top_n=5)

#intializing the class names and class length
if opt.dataset == 'UCF101':
    class_names = train_val_data.class_names
    class_len = get_ucf101_class_length()
    print(class_len[0])


# ########################### Driving train test loop  

# Training Loop
print("Starting Training")

best_accuracy = -1
for epoch in tqdm(range(1,epochs+1)):
    train_loss = Driver.train_step(train_loader, epoch)
    
    val_accuracy , e,confusion_matrix = Driver.val_step(val_loader,epoch)
    accuracy_test, confusion_matrix_test = Driver.test_model(test_loader,epoch)
    scheduler.step()
    
    #if epoch%5 == 0:
        #plot =  plot_confusion_matrix_diagonal(confusion_matrix,class_names, name ="Validation CM diagonal.png")
        #plot_test = plot_confusion_matrix_diagonal(confusion_matrix_test,class_names,name ="Test Diagonal matrix.png")
        #plot_confuse_matrix(confusion_matrix, class_names, normalize=True, title='Normalized confusion matrix')
        #cm_to_tb = add_cm_to_tb("Test Diagonal matrix.png")
        #tb_writer.add_image("Confusion Matrix",cm_to_tb,epoch)    

    if val_accuracy > best_accuracy: 
        torch.save(model, '../model_save/vivit-best-model.pt')
        torch.save(model.state_dict(), '../model_save/vivit-best-model-parameters.pt')

    print("validation cm diagonal: ",np.diag(np.array(confusion_matrix)))
    print("test cm diagonal: ",np.diag(np.array(confusion_matrix_test)))

print("validation cm diagonal: ",np.diag(np.array(confusion_matrix)))
print("test cm diagonal: ",np.diag(np.array(confusion_matrix_test)))
print("test cm sum: ",np.sum(np.array(confusion_matrix_test), axis = 0)) 

torch.save(model,"../model_save/vivit-last-model.pt")
torch.save(model.state_dict(), '../model_save/vivit-last-model-parameters.pt')

# ############################ Driver functions for driving the loop





