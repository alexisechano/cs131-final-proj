from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image


class ParksAndRecDataset(Dataset):
    """Parks and Recreation character images dataset."""
    
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.read_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        self.labels_map = {
            0: "leslie",
            1: "ben",
            2: "ron",
            3: "donna",
            4: "april",
        }

    def __len__(self):
        return len(self.read_file)

    def __getitem__(self, idx):
        item = self.read_file.iloc[idx]
        img_name = os.path.join(self.root_dir, item[0])
        label = self.getLabelFromName(item[1])
        image = Image.open(img_name).convert('RGB')
        newsize = (128, 128)
        image = image.resize(newsize)
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def getLabelFromName(self, label_name):
        for key in self.labels_map:
            val = self.labels_map[key]
            if val == label_name:
                return key
        return None
    
    def getLabelName(self, label_key):
        return self.labels_map[label_key]