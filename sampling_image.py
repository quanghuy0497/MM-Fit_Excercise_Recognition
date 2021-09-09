import os
import tqdm
import pprint
import torch
import torch.nn as nn
import pandas as pd
from utils import utils
import torchvision.transforms as transforms
from utils.data_transforms import Unit, Resample
from utils.dataset_image import Sampling
from torch.utils.data import RandomSampler, ConcatDataset, WeightedRandomSampler
import numpy as np

################
# Configuration
################

args = utils.parse_args()
torch.backends.cudnn.benchmark = True

ACTIONS = ['squats', 'lunges', 'bicep_curls', 'situps', 'pushups', 'tricep_extensions', 'dumbbell_rows',
           'jumping_jacks', 'dumbbell_shoulder_press', 'lateral_shoulder_raises', 'non_activity']
TRAIN_W_IDs = ['01', '02', '03', '04', '06', '07', '08', '16', '17', '18']
VAL_W_IDs = ['14', '15', '19']
if args.unseen_test_set:
    TEST_W_IDs = ['00', '05', '12', '13', '20']
else:
    TEST_W_IDs = ['09', '10', '11']
# All modalities available in MM-Fit
MODALITIES = ['sw_l_acc', 'sw_l_gyr', 'sw_l_hr', 'sw_r_acc', 'sw_r_gyr', 'sw_r_hr', 'sp_l_acc', 'sp_l_gyr',
              'sp_l_mag', 'sp_r_acc', 'sp_r_gyr', 'sp_r_mag', 'eb_l_acc', 'eb_l_gyr', 'pose_2d', 'pose_3d']
# We use a subset of all modalities in this demo.
MODALITIES_SUBSET = ['pose_2d']


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

window_stride = int(args.window_stride * args.skeleton_sampling_rate)
skeleton_window_length = int(args.window_length * args.skeleton_sampling_rate)
sensor_window_length = int(args.window_length * args.target_sensor_sampling_rate)

# Set model training hyperparameters
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr

data_transforms = transforms.Compose([Unit(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

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
                                         sampler = val_weight_sampler,
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
sampling_data(train_loader, 1)

print("Val:")
sampling_data(val_loader, 1)

print("Test:")
sampling_data(test_loader)