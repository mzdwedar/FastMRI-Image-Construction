import torch
import os
import json
import shutil
from tqdm import tqdm
from typing import Any, Dict, List

def nmse(pred, target):
    return torch.sum((pred - target) ** 2) / torch.sum(target ** 2)

def move_data(files, src_dir, dest_dir):
    
    for filename in tqdm(files, desc=f"Moving files from {src_dir} to {dest_dir}"):
        source = os.path.join(src_dir, filename)
        dest = os.path.join(dest_dir, filename)
        shutil.move(source, dest)

def list_files(directory, extension=".h5", limit=None):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension)]
    if limit:
        files = files[:limit]
    return files

def load_dict(path: str) -> dict:
    """Load a dictionary from a JSON's filepath.

    Args:
        path (str): location of file.

    Returns:
        Dict: loaded JSON data.
    """
    with open(path, encoding='utf-8') as fp:
        d = json.load(fp)
    return d


def save_dict(d: dict, path: str, cls: Any = None, sortkeys: bool = False) -> None:
    """Save a dictionary to a specific location.

    Args:
        d (Dict): data to save.
        path (str): location of where to save the data.
        cls (optional): encoder to use on dict data. Defaults to None.
        sortkeys (bool, optional): whether to sort keys alphabetically. Defaults to False.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):  # pragma: no cover
        os.makedirs(directory)
    with open(path, "w", encoding='utf-8') as fp:
        json.dump(d, indent=2, fp=fp, cls=cls, sort_keys=sortkeys)
        fp.write("\n")