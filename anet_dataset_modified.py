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
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable


model = models.resnet18(pretrained=True)
layer = model._modules.get('avgpool')
model.eval()

transforms = tv.transforms.Compose([
    tv.transforms.Resize(256),
    tv.transforms.CenterCrop(224),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_vector(image):
    # Create a PyTorch tensor with the transformed image
    t_img = transforms(image)
    # Create a vector of zeros that will hold our feature vector
    # The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(512)

    # Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.flatten())                 # <-- flatten

    # Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # Run the model on our transformed image
    with torch.no_grad():                               # <-- no_grad context
        model(t_img.unsqueeze(0))                       # <-- unsqueeze
    # Detach our copy function from the layer
    h.remove()
    # Return the feature vector
    return my_embedding

# videoloader and other function for running acitivitynet
class VideoLoader(object):

    def __init__(self, image_loader=None):

        self.transform = tv.transforms.Compose([
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224), # (224x224)
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # (R,G,B) (mean, std)
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

            
            
        #video = [self.transform(pil) for pil in pils]
        video = [get_vector(pil) for pil in pils]
        print(video[0].shape)


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
