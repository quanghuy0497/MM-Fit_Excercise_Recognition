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
    img = Image.open(buf).convert('L')
    img = img.resize(size)
    
    return img

class MMFit(Dataset):
    """
    MM-Fit PyTorch Dataset class.
    """
    def __init__(self, modality_filepaths, label_path,
                    skeleton_window_length, skeleton_transform, split_sample):
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
        self.skeleton_transform = skeleton_transform
        self.split_sample = split_sample
        self.modalities = {}
        for modality, filepath in modality_filepaths.items():
            self.modalities[modality] = utils.load_modality(filepath)

        self.ACTIONS = action
        self.labels = utils.load_labels(label_path)

    def __len__(self):
        return int(self.modalities['pose_2d'].shape[1] / self.split_sample)
        
    def __getitem__(self, i):
        i = i * self.split_sample
        frame = self.modalities['pose_2d'][0, i, 0]
        sample_modalities = {}
        label = 'non_activity'
        reps = 0
        for row in self.labels:
            if (frame >= row[0]) and (frame <= row[1]):
                label = row[3]
                reps = row[2]
                break

        for modality, data in self.modalities.items():
            if data is None:
                sample_modalities[modality] = torch.zeros(1, size_w, size_h)
            else:
                sample_modalities[modality] = torch.as_tensor(self.skeleton_transform(
                    visualize_2d_pose(data[:, i, 1:])), dtype=torch.float)
        return sample_modalities, self.ACTIONS[label], reps


class Sampling(Dataset):
    """
    MM-Fit PyTorch Dataset class.
    """
    def __init__(self, modality_filepaths, label_path, skeleton_window_length, split_sample):
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
        for modality, filepath in modality_filepaths.items():
            self.modalities[modality] = utils.load_modality(filepath)

        self.ACTIONS = action
        self.labels = utils.load_labels(label_path)

    def __len__(self):
        return int(self.modalities['pose_2d'].shape[1] / self.split_sample)
        
    def __getitem__(self, i):
        i = i * self.split_sample
        frame = self.modalities['pose_2d'][0, i, 0]
        sample_modalities = {}
        label = 'non_activity'
        reps = 0
        for row in self.labels:
            if (frame >= row[0]) and (frame <= row[1]):
                label = row[3]
                reps = row[2]
                break

        return self.ACTIONS[label]


class SequentialStridedSampler(Sampler):
    """
    PyTorch Sampler Class to sample elements sequentially using a specified stride, always in the same order.
    Arguments:
        data_source (Dataset):
        stride (int):
    """

    def __init__(self, data_source, stride):
        """
        Initialize SequentialStridedSampler object.
        :param data_source: Dataset to sample from.
        :param stride: Stride to slide window in seconds.
        """
        self.data_source = data_source
        self.stride = stride
    
    def __len__(self):
        return len(range(0, len(self.data_source), self.stride))

    def __iter__(self):
        return iter(range(0, len(self.data_source), self.stride))