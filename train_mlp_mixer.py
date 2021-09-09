import os
import tqdm
import pprint
import torch
import torch.nn as nn
from torchsummary import summary
import pandas as pd
from utils import utils
import torchvision.transforms as transforms
from utils.data_transforms import Unit, Resample
from utils.dataset_image import MMFit, Sampling
from torch.utils.data import WeightedRandomSampler, RandomSampler, ConcatDataset, DataLoader, Dataset
from model.mlp_mixer import MLPMixer

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

exp_name = args.name
import datetime
output_path = args.output + str("MLPmixer_{:%Y%m%dT%H%M}/".format(datetime.datetime.now()))
if not os.path.exists(output_path):
    os.makedirs(output_path)

import json
with open(output_path + 'configuration.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

device = torch.device('cuda:0')

window_stride = int(args.window_stride * args.skeleton_sampling_rate)
skeleton_window_length = int(args.window_length * args.skeleton_sampling_rate)
sensor_window_length = int(args.window_length * args.target_sensor_sampling_rate)

# Set model training hyperparameters
num_epochs = args.epochs
batch_size = args.batch_size
weight_decay = args.weight_decay
learning_rate = args.lr
dropout = args.dropout
momentum = args.momentum
gamma = 0.7

data_transforms = transforms.Compose([Unit(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

################
# Load data
################

train_datasets, val_datasets, test_datasets = [], [], []
train_samples, val_samples = [], []

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
        train_datasets.append(MMFit(modality_filepaths, label_path,
                                skeleton_window_length, skeleton_transform = data_transforms, split_sample = args.split_sample))
    elif w_id in VAL_W_IDs:
        val_samples.append(Sampling(modality_filepaths, label_path,
                                skeleton_window_length, split_sample = args.split_sample))
        val_datasets.append(MMFit(modality_filepaths, label_path,
                                skeleton_window_length, skeleton_transform = data_transforms, split_sample = args.split_sample))
    elif w_id in TEST_W_IDs:
        test_datasets.append(MMFit(modality_filepaths, label_path,
                                skeleton_window_length, skeleton_transform = data_transforms, split_sample = args.split_sample))
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
train_dataset = ConcatDataset(train_datasets)
val_dataset = ConcatDataset(val_datasets)
test_dataset = ConcatDataset(test_datasets)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                           # sampler=RandomSampler(train_dataset)
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

############################################################################
def compute_loss(modalities, labels, reps, model, criterion, total, correct, total_loss, step):
    for modality, data in modalities.items():
        modalities[modality] = data.to(device, non_blocking=True)
    # print(modalities[modality].shape)
       
    labels = labels.to(device, non_blocking=True)
    reps = reps.to(device, non_blocking=True)
    
    outputs = model(modalities['pose_2d'])
            
    loss = criterion(outputs, labels)
    total_loss += loss.item()
    total_loss_avg = total_loss / ((step + 1) * batch_size)
    total += labels.size(0)

    predicted = torch.argmax(outputs, dim=1)
    batch_correct = (predicted == labels).sum().item()
    correct += batch_correct
    acc = correct / total
    batch_acc = batch_correct / labels.size(0)
    
    return loss, acc, total_loss_avg, total, correct, total_loss

############################################################################
def train_one_epoch(train_loader, model, criterion, optimizer, epoch):
    model.train()
    total, correct, total_loss = 0, 0, 0
    acc = 0

    with tqdm.tqdm(total=len(train_loader)) as pbar_train:
        for step, (modalities, labels, reps) in enumerate(train_loader):
            loss, acc, total_loss_avg, total, correct, total_loss = compute_loss(modalities, labels, reps, model, criterion, total, correct, total_loss, step)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar_train.update(1)
            pbar_train.set_description('Epoch [{:02d}/{}], Accuracy: {:.6f}, Loss: {:.6f}'.format(epoch + 1, num_epochs, acc, total_loss_avg))
        
    return total_loss_avg, acc

############################################################################
def validation(val_loader, model, criterion):
    model.eval()
    with torch.no_grad():
        total, correct, total_loss = 0, 0, 0
        acc = 0
            
        with tqdm.tqdm(total=len(val_loader)) as pbar_val:
            for step, (modalities, labels, reps) in enumerate(val_loader):

                loss, acc, total_loss_avg, total, correct, total_loss = compute_loss(modalities, labels, reps, model, criterion, total, correct, total_loss, step)

                pbar_val.update(1)
                pbar_val.set_description('Validation:    Accuracy: {:.6f}, Loss: {:.6f}'.format(acc, total_loss_avg))
    
    return total_loss_avg, acc

############################################################################
def evaluation(model, best_model_state_dict, best_epoch, test_loader, criterion):
    model.load_state_dict(best_model_state_dict)
    with torch.no_grad():
        total, correct, total_loss = 0, 0, 0
        acc = 0

        with tqdm.tqdm(total=len(test_loader)) as pbar_test:
            for step, (modalities, labels, reps) in enumerate(test_loader):

                loss, acc, total_loss_avg, total, correct, total_loss = compute_loss(modalities, labels, reps, model, criterion, total, correct, total_loss, step)

                pbar_test.update(1)
                pbar_test.set_description('Test: Accuracy: {:.6f}, Loss: {:.6f}'.format(acc, total_loss_avg))
    return total_loss_avg, acc

############################################################################
def train(train_loader, val_loader, test_loader, model, criterion, optimizer, scheduler, num_epochs):

    best_model_state_dict = model.state_dict()
    best_valid_acc = None
    best_epoch = -1
    df = pd.DataFrame(columns=['Epoch', 'Batch', 'Type', 'Loss', 'Accuracy'])
    cur_index = 0

    for epoch in range(num_epochs):
        # Training
        total_loss_avg, acc = train_one_epoch(train_loader, model, criterion, optimizer, epoch)

        if (epoch + 1) % args.checkpoint == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_valid_acc': best_valid_acc
            }, os.path.join(output_path, exp_name + '_checkpoint_' + str(epoch + 1) + '.pth'))

        df.loc[cur_index] = [epoch + 1, None, 'train', total_loss_avg, acc]
        cur_index += 1

        # Validating
        if (epoch + 1) % args.eval_every == 0:
            total_loss_avg, acc = validation(val_loader, model, criterion)

            if best_valid_acc is None or acc >= best_valid_acc:
                best_valid_acc = acc
                steps_since_improvement = 0
                best_model_state_dict = model.state_dict()
                best_epoch = epoch
            else:
                steps_since_improvement += 1
                if steps_since_improvement == args.early_stop:
                    df.loc[cur_index] = [epoch, None, 'validation', total_loss_avg, acc]
                    cur_index += 1
                    print('No improvement detected in the last %d epochs, exiting.' % args.early_stop)
                    break

            df.loc[cur_index] = [epoch + 1, None, 'validation', total_loss_avg, acc]
            df.to_csv(os.path.join(output_path, exp_name + '.csv'), index=False)
            cur_index += 1
            print()
        
        scheduler.step(total_loss_avg)
        # scheduler.step()

        df.to_csv(os.path.join(output_path, exp_name + '.csv'), index=False)

    # Evaluating
    total_loss_avg, acc = evaluation(model, best_model_state_dict, best_epoch, test_loader, criterion)
    torch.save({'model_state_dict': model.state_dict()},
               os.path.join(output_path, exp_name + '_e' + str(best_epoch + 1) + '_best.pth'))

    df.loc[cur_index] = [best_epoch + 1, None, 'test', total_loss_avg, acc]
    df.to_csv(os.path.join(output_path, exp_name + '.csv'), index=False)

################
# Instantiate model
################

model = MLPMixer(
    image_size=256,
    channels = 1,
    patch_size=32,
    dim = 128,
    depth = 6,
    num_classes=11,
    dropout = dropout,
).to(device)

# summary(model, (1,256,256))

################
# Training
################

if args.model_wp != "":
    model_params = torch.load(args.model_wp, map_location=device)
    model.load_state_dict(model_params['model_state_dict'])

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad= False)

# scheduler = StepLR(optimizer, step_size=2, gamma=gamma)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)

train(train_loader, val_loader, test_loader, model, criterion, optimizer, scheduler, num_epochs)
