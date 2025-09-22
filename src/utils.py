import torch
import os
import shutil
from tqdm import tqdm

def nmse(pred, target):
    return torch.sum((pred - target) ** 2) / torch.sum(target ** 2)

def move_data(files, src_dir, dest_dir):
    
    for filename in tqdm(files, desc=f"Moving files from {src_dir} to {dest_dir}"):
        source = os.path.join(src_dir, filename)
        dest = os.path.join(dest_dir, filename)
        shutil.move(source, dest)