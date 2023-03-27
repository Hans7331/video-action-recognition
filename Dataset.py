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

    class_names = [x.strip().split()[1] for x in open('UCF101/classInd.txt').readlines()]
    def __init__(self, dataset_dir, subset, video_list_file, frames_per_clip=16):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.subset=subset
        self.video_list_file = video_list_file

        with open(dataset_dir+'/'+ video_list_file) as video_names_file:
            if self.subset=="train":
                self.video_list,self.labels = zip(*(files[:-1].split() for files in video_names_file.readlines()))
                #self.video_list,self.labels = list(self.video_list)[:5],list(self.labels)[:5]
            else:
                self.video_list = [files[:-1] for files in video_names_file.readlines()]
                with open(f'{dataset_dir}/classInd.txt') as self.classIndices:
                    values,keys=zip(*(files[:-1].split() for files in self.classIndices.readlines()))
                    self.indices = dict( (k,v) for k,v in zip(keys,values))

        self.frames_per_clip = frames_per_clip

        self.transform = tv.transforms.Compose([
          tv.transforms.Resize(256),
          tv.transforms.CenterCrop(224), # (224x224)
          tv.transforms.ToTensor(),
          tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # (R,G,B) (mean, std)
        ])

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        videoname = f'/video_data/{self.video_list[idx]}'
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
            label=int(self.indices[videoname.split('/')[2]])-1
        return imgs,label

def get_ucf101_class_length():
    """
    Extracts the class length for UCF101 dataset.
    Returns:
        A dictionary where keys are class names and values are the number of videos in each class.
    """
    class_len = []
    class_length = {}
    with open('UCF101/classInd.txt', 'r') as f:
        for line in f:
            class_id, class_name = line.strip().split(' ')
            class_length[class_name] = 0
    with open('UCF101/trainlist01.txt', 'r') as f:
        for line in f:
            class_name = line.strip().split('/')[0]
            class_length[class_name] += 1
    with open('UCF101/testlist01.txt', 'r') as f:
        for line in f:
            class_name = line.strip().split('/')[0]
            class_length[class_name] += 1
    
    for val in class_length.values():
        class_len.append(val)
    return class_length, class_len



# videoloader and other function for running acitivitynet
class VideoLoader(object):

    def __init__(self, image_loader=None):

        self.transform = tv.transforms.Compose([
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224), # (224x224)
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # (R,G,B) (mean, std)
        ])

    def __call__(self, video_path, frame_indices):
        video = []
        zip_path = str(video_path) +  str("/") +  video_path.split("/")[-1] + str(".zip")
        zip_path = '/'.join(str(video_path).split("/")[:-1]) +  str("/v_") +  video_path.split("/")[-1] + str(".zip")
        pils = []
        z = zipfile.ZipFile(zip_path, "r")

        for file in z.infolist():
            if '.jpg' in str(file) and "MACOSX" not in str(file):
                pil = Image.open(z.open(file))
                pils.append(pil)
        
        # repeating last image 250 times for small fps calculation error fix
        for i in range(250):
            pils.append(pils[-1])
        pils = [pils[x-1] for x in frame_indices] # outside above for loop

            
            
        video = [self.transform(pil) for pil in pils]
        return video

def get_class_labels(data):
    if 'all-classes' in data.keys():
        class_names = data['all-classes']
    else:
        class_names = []
        for node1 in data['taxonomy']:
            is_leaf = True
            for node2 in data['taxonomy']:
                if node2['parentId'] == node1['nodeId']:
                    is_leaf = False
                    break
            if is_leaf:
                class_names.append(node1['nodeName'])

    class_labels_map = {}

    for i, class_name in enumerate(class_names):
        class_labels_map[class_name] = i

    return class_labels_map


def get_video_ids_annotations(data, subset):
    video_ids = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            video_ids.append(key)
            annotations.append(value['annotations'])

    return video_ids, annotations



# acitivitynet dataloader
class ActivityNet():

    def __init__(self, root_path, annotation_path, subset, num_frames):
            
        self.data, self.class_names = self.__make_dataset(root_path, annotation_path, subset, num_frames)
        self.target_type = 'label'
        self.loader = VideoLoader()
        self.num_frames = num_frames

    def __make_dataset(self, root_path, annotation_path, subset, num_frames):
        
        # loading the annotations json
        data = json.load(open(annotation_path))
        
        # extracting video and annotation in list based on subset
        video_ids, annotations = get_video_ids_annotations(data, subset) 
        
        # creating dict for class and indices
        class_to_idx = get_class_labels(data)
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        # creating dataset with sample information
        dataset = []
        for vid in range(len(video_ids)):
            
            if not os.path.exists( str(root_path) +  str("/v_") +  str(video_ids[vid]) + str(".zip")):
                print(video_ids[vid], "doesn't exist")
                continue
                
            video_path = str(root_path) +  str("/") +  f'{video_ids[vid]}'

            fps = 5
            for annotation in annotations[vid]:

                t_begin = math.floor(annotation['segment'][0] * fps) + 1
                t_end = math.floor(annotation['segment'][1] * fps) + 1
                nframes = t_end - t_begin if t_end - t_begin > 0 else 0 #frames in the segment
                
                # if less frames, continue
                if nframes <= num_frames:
                    frame_indices = np.arange(t_begin, t_end + 1).astype(np.int32)
                    frame_indices = np.append(frame_indices, t_begin) if len(frame_indices) == 0 else frame_indices
                # if more than than num_frames, uniformly select
                else:
                    frame_indices = np.linspace(t_begin, t_end - 1, num_frames)
                    frame_indices = np.round(frame_indices).astype(np.int32)                    
                    
                    
                frame_indices = list(frame_indices)

                sample = {
                    'video': video_path,
                    'segment': (frame_indices[0], frame_indices[-1] + 1),
                    'frame_indices': frame_indices,
                    'fps': fps,
                    'video_id': video_ids[vid],
                    'label': class_to_idx[annotation['label']],
                }
                
                # skip categories outside selected categories in json
                if annotation['label'] not in class_to_idx.keys():
                    print('category outside json')
                    pass
                
                dataset.append(sample)
        return dataset, idx_to_class


    def __loading(self, path, frame_indices):
        clip = self.loader(path, frame_indices)
        
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        clip = clip.permute(1,0,2,3)
        clip = clip.permute(1,2,3,0)
        if clip.shape[3] != self.num_frames:
            clip = torch.nn.functional.interpolate(clip, size = [clip.shape[2], self.num_frames], mode='bilinear')
        clip = clip.permute(3,0,1,2)
                
        return clip

    def __getitem__(self, index):
        path = self.data[index]['video']
        if isinstance(self.target_type, list):
            target = [self.data[index][t] for t in self.target_type]
        else:
            target = self.data[index][self.target_type]

        frame_indices = self.data[index]['frame_indices']
        clip = self.__loading(path, frame_indices)

        return clip, target

    def __len__(self):
        return len(self.data)
