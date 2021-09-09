import numpy as np
import torch
import utils.utils as utils
from torch.utils.data import Dataset, Sampler
import io
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import datasets, transforms
size_w = 256
size_h = size_w
action = {'squats': 0, 'lunges': 1, 'bicep_curls': 2, 'situps': 3, 'pushups': 4, 'tricep_extensions': 5, 'dumbbell_rows': 6, 'jumping_jacks': 7, 'dumbbell_shoulder_press': 8, 'lateral_shoulder_raises': 9, 'non_activity': 10}
limit = 250

def visualize_2d_pose(pose, size = (size_w,size_h)):

    fig, ax = plt.subplots()
    for joint in range(pose.shape[1]):
        ax.plot(pose[0, joint], pose[1, joint], 'r.', markersize=10)
    ax.axis('off')
    limbs = [(0, 1), (0, 14), (0, 15), (14, 16), (15, 17), (1, 2), (2, 3), (3, 4),
             (1, 5), (5, 6), (6, 7), (1, 8), (1, 11), (8, 9), (9, 10), (11, 12), (12, 13)]
    for limb in limbs:
        joint1_x, joint1_y = pose[0, limb[0]], pose[1, limb[0]]
        joint2_x, joint2_y = pose[0, limb[1]], pose[1, limb[1]]
        plt.plot([joint1_x, joint2_x], [joint1_y, joint2_y], 'k-')

    radius = 300
    ax.set_xlim((np.mean(pose[0, :]) - radius, np.mean(pose[0, :]) + radius))
    ax.set_ylim((np.mean(pose[1, :]) - radius, np.mean(pose[1, :]) + radius))
    plt.gca().invert_yaxis()

    buf = io.BytesIO()
    fig.savefig(buf)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    img = img.resize(size)
    
    return img

class MMFit(Dataset):
    """
    MM-Fit PyTorch Dataset class.
    """
    def __init__(self, modality_filepaths, label_path,
                    skeleton_window_length, split_sample=1, video_transform=None):
        """
        Initialize MMFit Dataset object.
        :param modality_filepaths: Modality - file path mapping (dict) for a workout.
        :param label_path: File path to MM-Fit CSV label file for a workout.
        :param window_length: Window length in seconds.
        :param skeleton_window_length: Skeleton window length in number of samples.
        :param sensor_window_length: Sensor window length in number of samples.
        :param skeleton_transform: Transformation functions to apply to skeleton data.
        :param sensor_transform: Transformation functions to apply to sensor data.
        """
        self.skeleton_window_length = skeleton_window_length
        self.split_sample = split_sample
        self.modalities = {}
        self.video_transform = video_transform
        for modality, filepath in modality_filepaths.items():
            self.modalities[modality] = utils.load_modality(filepath)
            # if self.split_sample:
            #   self.modalities[modality] = self.modalities[modality][:,0::self.split_sample,:]
        self.ACTIONS = action
        self.labels = utils.load_labels(label_path)
            

    def __len__(self):
        return min(int(self.modalities['pose_2d'].shape[1]/self.split_sample), limit)
        
    def __getitem__(self, i):
        # print(i)
        i = i * self.split_sample
        frame = self.modalities['pose_2d'][0, i, 0]
        sample_modalities = {}
        label = 'non_activity'
        reps = 0
        for row in self.labels:
            if (frame > (row[0] - self.skeleton_window_length/2)) and (frame < (row[1] - self.skeleton_window_length/2)):
                label = row[3]
                reps = row[2]
                break
                
        sample_images = []
        for modality, data in self.modalities.items():
            for j in range(i, i + self.skeleton_window_length):
                image = visualize_2d_pose(data[:, j, 1:])
                sample_images.append(image)
        
        sample_modalities['pose_2d'] = np.concatenate(sample_images, axis =2)
        
        #print(sample_modalities['pose_2d'].shape)

        if self.video_transform is not None:
            sample_modalities['pose_2d'] = self.video_transform(sample_modalities['pose_2d'])
            sample_modalities['pose_2d'] = sample_modalities['pose_2d'].view(-1, 3, self.skeleton_window_length, 256, 256).transpose(1, 2)
        
        # print(i," ", sample_modalities['pose_2d'].shape," ",label)

        return sample_modalities, self.ACTIONS[label], reps

class Sampling(Dataset):
    """
    MM-Fit PyTorch Dataset class.
    """
    def __init__(self, modality_filepaths, label_path,
                    skeleton_window_length, split_sample):

        self.skeleton_window_length = skeleton_window_length
        self.split_sample = split_sample
        self.modalities = {}
        for modality, filepath in modality_filepaths.items():
            self.modalities[modality] = utils.load_modality(filepath)

        self.ACTIONS = action
        self.labels = utils.load_labels(label_path)

    def __len__(self):
        return min(int(self.modalities['pose_2d'].shape[1]/self.split_sample), limit)
        
    def __getitem__(self, i):
        i = i * self.split_sample
        frame = self.modalities['pose_2d'][0, i, 0]
        sample_modalities = {}
        label = 'non_activity'
        reps = 0
        for row in self.labels:
            if (frame > (row[0] - self.skeleton_window_length/2)) and (frame < (row[1] - self.skeleton_window_length/2)):
                label = row[3]
                reps = row[2]
                break

        return self.ACTIONS[label]
