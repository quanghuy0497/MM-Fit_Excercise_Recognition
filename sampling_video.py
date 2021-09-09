import os
import tqdm
import pprint
import torch
import torch.nn as nn
import pandas as pd
from utils import utils
import torchvision.transforms as transforms
import utils.video_transforms as video_transforms
from utils.dataset_video import MMFit, Sampling
from torch.utils.data import WeightedRandomSampler, RandomSampler, ConcatDataset, DataLoader, Dataset
from model.vivit2 import ViViT

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR

################
# Configuration
################

args = utils.parse_args()
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(vars(args))
torch.backends.cudnn.benchmark = True

TRAIN_W_IDs = ['01', '02', '03', '04', '06', '07', '08', '16', '17', '18']
VAL_W_IDs = ['14', '15', '19']
if args.unseen_test_set:
    TEST_W_IDs = ['00', '05', '12', '13', '20']
else:
    TEST_W_IDs = ['09', '10', '11']

# We use a subset of all modalities in this demo.
MODALITIES_SUBSET = ['pose_2d']
    
device = torch.device('cuda:0')

skeleton_window_length = int(args.window_length * args.skeleton_sampling_rate)

# Set model training hyperparameters
num_epochs = args.epochs
weight_decay = args.weight_decay
batch_size = 1
learning_rate = args.lr
momentum = args.momentum
dropout = args.dropout
split_sample = args.split_sample
gamma = 0.7

scale = 1
input_size = int(256 * scale)
length= skeleton_window_length
clip_mean = [0.5, 0.5, 0.5] * length
clip_std = [0.5, 0.5, 0.5] * length
scale_ratios = [1.0, 0.875, 0.75, 0.66]
normalize = video_transforms.Normalize(mean=clip_mean, std=clip_std)


################
# Create data loaders
################

train_transform = video_transforms.Compose([
                    video_transforms.MultiScaleCrop((input_size, input_size), scale_ratios),
                    video_transforms.RandomHorizontalFlip(),
                    video_transforms.ToTensor(),
                    normalize,
                ])

val_test_transform = video_transforms.Compose([
        video_transforms.CenterCrop((input_size)),
        video_transforms.ToTensor(),
        normalize,
    ])

################
# Load data
################

train_samples, val_samples, test_samples = [], [], []

for w_id in TRAIN_W_IDs + VAL_W_IDs + TEST_W_IDs:
    modality_filepaths = {}
    workout_path = os.path.join(args.data, 'w' + w_id)
    files = os.listdir(workout_path)
    label_path = None
    for file in files:
        if 'labels' in file:
            label_path = os.path.join(workout_path, file)
            continue
        for modality_type in MODALITIES_SUBSET:
            if modality_type in file:
                modality_filepaths[modality_type] = os.path.join(workout_path, file)
    if label_path is None:
        raise Exception('Error: Label file not found for workout {}.'.format(w_id))

    if w_id in TRAIN_W_IDs:
        train_samples.append(Sampling(modality_filepaths, label_path,
                                skeleton_window_length, split_sample = args.split_sample))
    elif w_id in VAL_W_IDs:
        val_samples.append(Sampling(modality_filepaths, label_path,
                                skeleton_window_length, split_sample = args.split_sample))
    elif w_id in TEST_W_IDs:
        test_samples.append(Sampling(modality_filepaths, label_path,
                                skeleton_window_length, split_sample = args.split_sample))
    else:
        raise Exception('Error: Workout {} not assigned to train, test, or val datasets'.format(w_id))


################
# Weight Sampling
################

def Weight_Sampling(data_samples):
    sample_loader = torch.utils.data.DataLoader(dataset = ConcatDataset(data_samples), batch_size=1, shuffle=False)

    data_label = []
    for i, labels in enumerate(sample_loader):
        data_label.append(labels.to(device, non_blocking=True).item())

    sample_class_count = np.array([len(np.where(data_label==t)[0]) for t in np.unique(data_label)])
    weight = 1. / sample_class_count

    sample_weight = torch.tensor([weight[t] for t in data_label])

    return WeightedRandomSampler(sample_weight, len(sample_weight))

################
# Create data loaders
################

train_weight_sampler = Weight_Sampling(train_samples)
val_weight_sampler = Weight_Sampling(val_samples)
train_dataset = ConcatDataset(train_samples)
val_dataset = ConcatDataset(val_samples)
test_dataset = ConcatDataset(test_samples)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                           sampler=train_weight_sampler, 
                                           pin_memory=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size,
                                         # sampler = val_weight_sampler,
                                         pin_memory=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                          pin_memory=True, num_workers=4)

print()
print("---------------------------")
print("Set".ljust(10),"Image".ljust(10),"Iter")
print("---------------------------")
print("Train".ljust(10),format(len(train_dataset)).ljust(10), len(train_loader))
print("Val".ljust(10),format(len(val_dataset)).ljust(10), len(val_loader))
print("Test".ljust(10),format(len(test_dataset)).ljust(10), len(test_loader))
print("---------------------------")

################
# Sampling data
################

def sampling_data(data_loader, intensive = 0):
    classes = np.zeros((11), dtype=np.int32)

    for i, labels in enumerate(data_loader):
        labels = labels.to(device, non_blocking=True)
        if intensive != 0:
            print("batch index {}: {}/{}/{}/{}/{}/{}/{}/{}/{}/{}/{}".format(i, (labels == 0).sum(), (labels == 1).sum(), (labels == 2).sum(), (labels == 3).sum(), (labels == 4).sum(), (labels == 5).sum(), (labels == 6).sum(), (labels == 7).sum(), (labels == 8).sum(), (labels == 9).sum(), (labels == 10).sum()))
        
        for a in range(labels.size(0)):
            classes[labels[a]] += 1
    
    print(classes)

print("Train:")
sampling_data(train_loader)

print("Val:")
sampling_data(val_loader)

print("Test:")
sampling_data(test_loader)