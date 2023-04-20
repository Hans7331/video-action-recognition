import torchvision as tv
import torch
from torch.nn import functional as F
import os
import decord
import numpy as np
import torchvision.transforms as T
import time
from icecream import ic
import math
import json
import time
import random
import torch.utils.data as data
from torchvision import transforms
from pathlib import Path
from torch import nn, einsum
from torch.utils.data import random_split, DataLoader,Dataset 
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from PIL import Image
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import pickle
import zipfile


class UCFDataset(torch.utils.data.Dataset):
    """
    Dataset Class for reading UCF101 dataset  
    
    Args:
        dataset_dir: (str) - root directory of dataset
        subset: (str) - train or test subset
        video_list_file: (str) - file name containing list of video names 
        frames_per_clip: (int) - number of frames to be read in every video clip [default:16]
    """

    class_names = [x.strip().split()[1] for x in open('../UCF101/classInd.txt').readlines()]
    def __init__(self, dataset_dir, subset, video_list_file, frames_per_clip=16):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.subset=subset
        self.video_list_file = video_list_file

        with open(dataset_dir+'/'+ video_list_file) as video_names_file:
            if self.subset=="train":
                self.video_list,self.labels = zip(*(files[:-1].split() for files in video_names_file.readlines()))
                self.video_list,self.labels = list(self.video_list)[:10],list(self.labels)[:10]
            else:
                self.video_list = [files[:-1] for files in video_names_file.readlines()]
                with open(f'{dataset_dir}/classInd.txt') as self.classIndices:
                    values,keys=zip(*(files[:-1].split() for files in self.classIndices.readlines()))
                    self.indices = dict( (k,v) for k,v in zip(keys,values))

        self.frames_per_clip = frames_per_clip

        self.transform = tv.transforms.Compose([
          tv.transforms.Resize(256),
          tv.transforms.CenterCrop(112), # (224x224)
          tv.transforms.ToTensor(),
          tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # (R,G,B) (mean, std)
        ])

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        videoname = f'UCF-101/{self.video_list[idx]}'
        vid = decord.VideoReader(f'{self.dataset_dir}/{videoname}', ctx=decord.cpu(0)) # for reading frames in videos
        nframes = len(vid)

        # if number of frames of video is less than frames_per_clip, repeat the frames
        if nframes <= self.frames_per_clip:
            idxs = np.arange(0, self.frames_per_clip).astype(np.int32)
            idxs[nframes:] %= nframes

        # else if frames_per_clip is greater, sample uniformly seperated frames
        else:
            idxs = np.linspace(0, nframes-1, self.frames_per_clip)
            idxs = np.round(idxs).astype(np.int32)

        imgs = []
        for k in idxs:
            frame = Image.fromarray(vid[k].asnumpy())
            frame = self.transform(frame)
            imgs.append(frame)
        imgs = torch.stack(imgs)

        # if its train subset, return both the frames and the label 
        if self.subset=="train":
            label = int(self.labels[idx]) - 1    
        # else, for test subset, read the label index
        else:
            label=int(self.indices[videoname.split('/')[1]])-1
        return imgs,label

def get_ucf101_class_length():
    """
    Extracts the class length for UCF101 dataset.
    Returns:
        A dictionary where keys are class names and values are the number of videos in each class.
    """
    class_len = []
    class_length = {}
    with open('../UCF101/classInd.txt', 'r') as f:
        for line in f:
            class_id, class_name = line.strip().split(' ')
            class_length[class_name] = 0
    with open('../UCF101/trainlist01.txt', 'r') as f:
        for line in f:
            class_name = line.strip().split('/')[0]
            class_length[class_name] += 1
    with open('../UCF101/testlist01.txt', 'r') as f:
        for line in f:
            class_name = line.strip().split('/')[0]
            class_length[class_name] += 1
    
    for val in class_length.values():
        class_len.append(val)
    return class_length, class_len



