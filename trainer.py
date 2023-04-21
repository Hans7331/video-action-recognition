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
from torch.cuda.amp import autocast, GradScaler
from model_2_scratch import ViViT as model_2_scratch
from model_pretrained_all_layers import ViViT_2 as model_2_pretrained_all_layers
from checkpoint_saver import CheckpointSaver
from confusion_matrix import plot_confuse_matrix,add_cm_to_tb,plot_confusion_matrix_diagonal,ConfusionMatrix
from contrastive_loss.nt_xent_original import *
from contrastive_loss.global_local_temporal_contrastive import global_local_temporal_contrastive
from ucf_dataloader_cl import ss_dataset_gen1, collate_fn2
from cl_model import *  # original constrative loss model from TCLR paper (for testing)
from r3d import r3d_18



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


############################################### Original Constrastive loss model

def build_r3d_backbone(): #Official PyTorch R3D-18 model taken from https://github.com/pytorch/vision/blob/master/torchvision/models/video/resnet.py
    model = r3d_18(pretrained = False, progress = False)
    model.layer4[0].conv1[0] = nn.Conv3d(256, 512, kernel_size=(3, 3, 3),\
                                stride=(1, 2, 2), padding=(2, 1, 1),dilation = (2,1,1), bias=False)
    model.layer4[0].downsample[0] = nn.Conv3d(256, 512,\
                          kernel_size = (1, 1, 1), stride = (1, 2, 2), bias=False)
    return model


############################# Driver functions for driving the loop
class Driver:

    def train_epoch(scaler, learning_rate2, epoch, criterion, data_loader, model, optimizer, criterion2):
        print('train at epoch {}'.format(epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate2

            print("Learning rate is: {}".format(param_group['lr']))
    
        losses = []
        labels = []
        losses_gsr_gdr, losses_ic2, losses_ic1, losses_local_local = [], [], [], []
        losses_global_local = []

        model.train()

        #for i, j in enumerate(data_loader):
        #    print(j[-1])

        for i, (sparse_clip, dense_clip0, dense_clip1, dense_clip2, dense_clip3, a_sparse_clip, \
                a_dense_clip0, a_dense_clip1, a_dense_clip2, a_dense_clip3,_ ,_,_,label) in enumerate(data_loader):
            
            labels.append(label)
            optimizer.zero_grad()
            a_sparse_clip = a_sparse_clip.permute(0,1,2,3,4).to(device) #aug_DL output is [120, 16, 3, 112, 112]], model expects [8, 3, 16, 112, 112]
            a_dense_clip0 = a_dense_clip0.permute(0,1,2,3,4).to(device)        
            a_dense_clip1 = a_dense_clip1.permute(0,1,2,3,4).to(device)
            a_dense_clip2 = a_dense_clip2.permute(0,1,2,3,4).to(device)
            a_dense_clip3 = a_dense_clip3.permute(0,1,2,3,4).to(device)

            sparse_clip = sparse_clip.permute(0,1,2,3,4).to(device) #aug_DL output is [120, 16, 3, 112, 112]], model expects [8, 3, 16, 112, 112]
            dense_clip0 = dense_clip0.permute(0,1,2,3,4).to(device)        
            dense_clip1 = dense_clip1.permute(0,1,2,3,4).to(device)
            dense_clip2 = dense_clip2.permute(0,1,2,3,4).to(device)
            dense_clip3 = dense_clip3.permute(0,1,2,3,4).to(device)
            
            # out_sparse will have output in this order: [sparse_clip[5], augmented_sparse_clip]
            # one element from the each of the list has 5 elements: see MLP file for details
            out_sparse = []
            # out_dense will have output in this order : [d0,d1,d2,d3,a_d0,...]
            out_dense = [[],[]]

            with autocast():
                
                out_sparse.append(model((sparse_clip.cuda(),'s')))
                out_sparse.append(model((a_sparse_clip.cuda(),'s')))

                out_dense[0].append(model((dense_clip0.cuda(),'d')))
                out_dense[0].append(model((dense_clip1.cuda(),'d')))
                out_dense[0].append(model((dense_clip2.cuda(),'d')))
                out_dense[0].append(model((dense_clip3.cuda(),'d')))

                out_dense[1].append(model((a_dense_clip0.cuda(),'d')))
                out_dense[1].append(model((a_dense_clip1.cuda(),'d')))
                out_dense[1].append(model((a_dense_clip2.cuda(),'d')))
                out_dense[1].append(model((a_dense_clip3.cuda(),'d')))


                criterion = NTXentLoss(device = 'cuda', batch_size = out_sparse[0][0].shape[0], temperature=opt.temperature, use_cosine_similarity = False).to(device)
                criterion_local_local = NTXentLoss(device = 'cuda', batch_size = 4, temperature=opt.temperature, use_cosine_similarity = False).to(device)
                # Instance contrastive losses with the global clips (sparse clips)
                
                
                # there 4,128
                # ours 101
                loss_ic2 = criterion(out_sparse[0][0], out_sparse[1][0])
                

                loss_ic1 = 0
                
                # Instance contrastive losses with the local clips (dense clips)
                for ii in range(2):
                    for jj in range(2):
                        for chunk in range(1,5):
                            for chunk1 in range(1,5):
                                if (ii == jj and chunk == chunk1):
                                    continue
                                loss_ic1 += criterion(out_dense[ii][chunk-1],out_dense[jj][chunk1-1])
                
                loss_ic1 /= 4 #scaling over ii and jj

                loss_local_local = 0
                # print(out_dense[0][0].shape) # this prints shape of [4,128]
                # print(torch.stack(out_dense[0],dim=1).shape) # this prints shape of [BS, 4, 128]
                # exit()
                for ii in range(out_dense[0][0].shape[0]): #for loop in the batch size
                    loss_local_local += criterion_local_local(torch.stack(out_dense[0],dim=1)[ii], torch.stack(out_dense[1],dim=1)[ii])
                
                loss_global_local=0
                for ii in range(2):
                    for jj in range(2):
                        loss_global_local += criterion2(torch.stack(out_sparse[ii][1:],dim=1), torch.stack(out_dense[jj],dim=1), opt.temperature)

                loss = loss_ic2 + loss_ic1 + loss_local_local + loss_global_local
                

            loss_unw = loss_ic2.item()+ loss_ic1.item() + loss_local_local.item() + loss_global_local.item()
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.append(loss_unw)
            losses_local_local.append(loss_local_local.item())
            losses_global_local.append(loss_global_local.item())

            losses_ic1.append(loss_ic1.item())
            losses_ic2.append(loss_ic2.item())

            
            if (i+1) % 2 == 0: 
                print(f'Training Epoch {epoch}, Batch {i}, Loss: {np.mean(losses) :.5f}')
                print(f'Training Epoch {epoch}, Batch {i}, losses_local_local: {np.mean(losses_local_local) :.5f}')
                print(f'Training Epoch {epoch}, Batch {i}, losses_global_local: {np.mean(losses_global_local) :.5f}')
                print(f'Training Epoch {epoch}, Batch {i}, losses_ic2: {np.mean(losses_ic2) :.5f}')
                print(f'Training Epoch {epoch}, Batch {i}, losses_ic1: {np.mean(losses_ic1) :.5f}')

            # exit()
        print('Training Epoch: %d, Loss: %.4f' % (epoch,  np.mean(losses)))

        
        del out_sparse, out_dense, loss, loss_ic2, loss_ic1, losses_local_local, loss_global_local

        return model, np.mean(losses), scaler, labels

    # Test step for each epoch
    def test_model(model, test_dataloader):

        num_videos = 0
        model.eval()
        corrects=0

        with torch.no_grad():
            for batch_id, (video_data,labels) in enumerate(test_dataloader):
                
                video_data, labels = video_data.to(device), labels.to(device)
                prediction = model( (video_data.permute(0,1,2,3,4), "d"))

                prediction = prediction[:,0:101]
                prediction_1 = torch.argmax(prediction,dim=1)

                #prediction_1 [0,0,0,0]
                #labels [0,0,0,0]
                for label, pred in zip(labels, prediction_1):
                    corrects += (label == pred)
                num_videos += video_data.size(0)

        accuracy = corrects.item()/num_videos

        return accuracy*100


############################# model initialisation
tb_writer = SummaryWriter()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    train_dataset = ss_dataset_gen1(shuffle = True, data_percentage = 1, video_list_file = 'trainlist01.txt')
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn2)    

  

    test_dataset = UCFDataset(dataset_dir = dataset_dir, subset="test", video_list_file="testlist01.txt" ,frames_per_clip=frames_per_clip)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size)

    print(f"Train samples: {len(train_dataloader)}")
    print(f"Test samples: {len(test_dataloader)}")

else:
    from anet_dataset import ActivityNet
    train_data = ActivityNet(root_path = dataset_dir, annotation_path = annotation_path, subset = 'training', num_frames = frames_per_clip)

    class_names = train_data.class_names # saving class names
    class_length = json.load(open(annotation_path))['class-lengths'] if 'class-lengths' in json.load(open(annotation_path)).keys() else {} # saving length of all classes
    train_data, val_data = random_split(train_data, [ round(len(train_data)*train_perc), len(train_data) - round(len(train_data)*train_perc) ])

    test_data = ActivityNet(root_path = dataset_dir, annotation_path = annotation_path, subset = 'validation', num_frames = frames_per_clip)

    print("Train data length : ", len(train_data), "shape : " , next(iter(train_data))[0].shape)
    print("Val data length :", len(val_data), "shape : " , next(iter(val_data))[0].shape)
    print("Test data length :", len(test_data), "shape : " , next(iter(test_data))[0].shape)



# initialize model and plot on tensorboard
if opt.pr == 0:
    model = model_2_scratch(image_size= opt.image_size, patch_size=patch_size, num_classes=num_classes, frames_per_clip=frames_per_clip)
elif opt.pr == 2:
    model = model_2_pretrained_all_layers(image_size= opt.image_size, patch_size=patch_size, num_classes=num_classes, frames_per_clip=frames_per_clip)


f = build_r3d_backbone()
model = nn.Sequential(f,model)
model.to(device)

# define the loss and optimizers
loss_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

# Intialising the checkpoint saver
if opt.checkpoint_flag: 
    checkpoint_saver = CheckpointSaver(dirpath='./model_weights7', decreasing=True, top_n=5)


# ########################### Driving train test loop  

# Training Loop
print("Starting Training")

criterion = NTXentLoss(device = 'cuda', batch_size=opt.batch_size, temperature=opt.temperature, use_cosine_similarity = False)
scaler = GradScaler()

best_accuracy = -1
for epoch in range(1,epochs+1):

    
    model, loss, scaler, labels = Driver.train_epoch(scaler, opt.lr, epoch, criterion, train_dataloader, model, optimizer, criterion2 = global_local_temporal_contrastive)
    print(loss)


    scheduler.step()

####################### saving the model

torch.save(model,"../model_save/vivit-last-model.pt")
torch.save(model.state_dict(), '../model_save/vivit-last-model-parameters.pt')


####################### testset accuracy calculation
test_accuracy = Driver.test_model(model, test_dataloader)
print("Test Accuracy", test_accuracy)

