from base import BaseDataset
from skimage import io, transform
from PIL import Image
import pandas as pd
import numpy as np
import os


class OracleDataset(BaseDataset):
    """Oracle bone inscription dataset"""

    def __init__(self, root_dir, train=True, trans=None):
        super().__init__(root_dir)
        if train:
            img_target = pd.read_csv("dataset/train.csv")
        else:
            img_target = pd.read_csv("dataset/test.csv")
        self.images = np.asarray(img_target["image"])
        self.targets = np.asarray(img_target["target"])
        self.transform = trans

    def __len__(self):
        return self.targets.size

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.images[index])
        image = Image.open(img_path)
        target = self.targets[index]

        if self.transform:
            image = self.transform(image)

        return image, target
