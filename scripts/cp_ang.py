"""
Copy the corresponding angle image for 2-channel training
"""
import os
from shutil import copy


# dir containing phase
src_dir = "/home/terbed/PROJECTS/DHM/DATA/BF/dataset_v2/dataset/input"

# dir to copy corresponding angle images
trg_dirs = ["/home/terbed/PROJECTS/DHM/DATA/BF/unpaired/dataset_v2_cyclegan_2chan/trainA",
            "/home/terbed/PROJECTS/DHM/DATA/BF/unpaired/dataset_v2_cyclegan_2chan/valA",
            "/home/terbed/PROJECTS/DHM/DATA/BF/unpaired/dataset_v2_cyclegan_2chan/testA"]


for trg_dir in trg_dirs:
    _, _, fnames = next(os.walk(trg_dir))
    fnames = [fname.split(".")[0] for fname in fnames]       # remove extension

    for fname in fnames:
        idx = fname.split("_")[-1]
        trg_fname = f"ang_{idx}.png"
        trg_full_path = os.path.join(src_dir, trg_fname)
        copy(trg_full_path, trg_dir)
        print(f"{trg_full_path} copied to {trg_dir}!")
