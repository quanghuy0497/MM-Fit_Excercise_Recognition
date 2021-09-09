import os
import tqdm
import pprint
import torch
import torch.nn as nn
import pandas as pd
from utils import utils
import torchvision.transforms as transforms
import utils.video_transforms as video_transforms
from utils.dataset_vivit import MMFit
from torch.utils.data import RandomSampler, ConcatDataset, DataLoader, Dataset
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

trained_model_path = os.getcwd() + "/output/" + args.trained_model

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

    
device = torch.device('cuda:0')

##################################################

skeleton_window_length = int((args.window_length * args.skeleton_sampling_rate))

# Set model training hyperparameters
num_epochs = args.epochs
weight_decay = args.weight_decay
batch_size = args.batch_size
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

test_datasets = []
for w_id in TEST_W_IDs:
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

    test_datasets.append(MMFit(modality_filepaths, label_path, args.window_length,
                                skeleton_window_length, video_transform = val_test_transform, split_sample = split_sample))

test_dataset = ConcatDataset(test_datasets)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

print()
print("---------------------------")
print("Test".ljust(10),format(len(test_dataset)).ljust(10), len(test_loader))
print("---------------------------")


################
# Instantiate model
################

model = ViViT(
    image_size=256,
    patch_size=32,
    tubelet_temporal_size=2,
    num_classes=11,
    num_frames=skeleton_window_length,
    dim=128,
    layer_spacial=12,
    layer_temporal=4,
    heads=16,
    pool='cls',
    dim_head=64,
    dropout=dropout,
    emb_dropout=0.1,
    in_channels = 3,
    mlp_dim=2048,
    pretrain=False
).to(device)


parameters = filter(lambda p: p.requires_grad, model.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('Trainable Parameters: %.3fM' % parameters)
if args.model_wp != "":
    model_params = torch.load(args.model_wp, map_location=device)
    model.load_state_dict(model_params['model_state_dict'])

################
# Training
################

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)

# evaluation(model, trained_model_path, test_loader, criterion)

model_params = torch.load(trained_model_path, map_location=device)
model.load_state_dict(model_params['model_state_dict'])

with torch.no_grad():
    total, correct, total_loss = 0, 0, 0
    
    with tqdm.tqdm(total=len(test_loader)) as pbar_test:
        for i, (modalities, labels, reps) in enumerate(test_loader):

            for modality, data in modalities.items():
                modalities[modality] = data[0].to(device, non_blocking=True)

            labels = labels.to(device, non_blocking=True)
            reps = reps.to(device, non_blocking=True)
            outputs = model(modalities[modality])

            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_loss_avg = total_loss / ((i + 1) * batch_size)
            total += labels.size(0)

            _, predicted = torch.max(outputs, dim=1)
            batch_correct = (predicted == labels).sum().item()
            correct += batch_correct
            print("=======================")
            print("Predicted:", ACTIONS[predicted.item()])
            print("Groundtruth:", ACTIONS[labels.item()])
            acc = correct / total
            batch_acc = batch_correct / labels.size(0)
            print("Accuracy:", acc)
            print("Loss:", total_loss_avg)

            # pbar_test.update(1)
            # pbar_test.set_description('Test: Accuracy: {:.6f}, Loss: {:.6f}'.format(acc, total_loss_avg))