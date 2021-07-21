import os
from data.base_dataset import BaseDataset, get_transform, get_params
from data.image_folder import make_dataset
from PIL import Image
import torch
import random
import torchvision.transforms.functional as tvf
import numpy as np


class HoloCDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt, load_to_memory=True):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        _, _, self.A_names = next(os.walk(self.dir_A))
        self.A_names = [name.split(".")[0] for name in self.A_names]
        _, _, self.B_names = next(os.walk(self.dir_B))
        self.B_names = [name.split(".")[0] for name in self.B_names]

        self.idxs_A = [name.split("_")[-1] for name in self.A_names if name.split("_")[0] == "amp"]
        self.idxs_B = [name.split("_")[-1] for name in self.B_names if name.split("_")[0] == "amp"]

        self.idxs = self.common_member(self.idxs_A, self.idxs_B)
        self.size = len(self.idxs)

        # self.transform_B = get_transform(self.opt, nc=1)

        self.database = []
        if load_to_memory:
            for idx in self.idxs:
                amp_path = os.path.join(self.dir_A, f"amp_{idx}.png")
                ang_path = os.path.join(self.dir_A, f"ang_{idx}.png")
                lab = os.path.join(self.dir_B, f"amp_{idx}.png")

                amp = Image.open(amp_path)
                ang = Image.open(ang_path)
                lab = Image.open(lab)

                # apply image transformation (the same transform for amp+ang)
                curr_params = get_params(opt, (opt.load_size, opt.load_size))
                transform = get_transform(self.opt, params=curr_params, nc=1)
                transform2 = get_transform(self.opt, params=curr_params, nc=1, convert=False)
                amp = transform(amp)

                ang = transform2(ang)
                ang = tvf.to_tensor(ang)
                ang = ang*2*np.pi - np.pi
                z = amp*torch.exp(1j*ang)

                lab = transform(lab).type(torch.complex64)

                print(z.shape)
                print(lab.shape)

                self.database.append({'A': z, 'B': lab, 'A_paths': idx, 'B_paths': idx})

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """

        return self.database[index]

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.size

    @staticmethod
    def common_member(a, b):
        a_set = set(a)
        b_set = set(b)

        if not (a_set & b_set):
            print("No common elements")

        return list(a_set & b_set)
