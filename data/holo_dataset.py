import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch
import random


class HoloDataset(BaseDataset):
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
        self.A_names = [name.split(".")[0] for name in self.B_names]

        self.idxs = [name.split("_")[-1] for name in self.A_names if name.split("_")[0] == "amp"]

        self.A_size = len(self.A_names)//2  # get the size of dataset A
        self.B_size = len(self.B_names)  # get the size of dataset B
        assert self.A_size == self.B_size, "A and B input has to be same number of elements."
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, nc=1)
        self.transform_B = get_transform(self.opt, nc=1)

        self.database = []
        if load_to_memory:
            for idx in self.idxs:
                amp_path = os.path.join(self.dir_A, f"amp_{idx}.png")
                ang_path = os.path.join(self.dir_A, f"ang_{idx}.png")
                lab = os.path.join(self.dir_B, f"amp_{idx}.png")

                amp = Image.open(amp_path)
                ang = Image.open(ang_path)
                lab = Image.open(lab)

                self.database.append({"Amp": amp, "Ang": ang, "BF": lab, "id": idx})

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

        sample = self.database[index]

        amp = sample["Amp"]
        ang = sample["Anp"]
        lab = sample["BF"]

        # apply image transformation
        amp = self.transform_A(amp)
        ang = self.transform_A(ang)
        lab = self.transform_B(lab)

        inp = torch.cat((amp, ang), dim=0)

        return {'A': inp, 'B': lab, 'A_paths': sample["id"], 'B_paths': sample["id"]}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
