# data classes

import os
from pathlib import Path

from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
from utils import tqdm_wrapper

data_path = {"train": "./data/train", "test": "./data/test"}

# random_crop = transforms.RandomCrop(33)
_crop_size = 33
_crop_stride = 14


def collect_data(path, progress=False):
    # get all files
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    ret = []
    # open all files
    wrapper = tqdm_wrapper
    if progress:
        print("Preparing data...")
        wrapper = tqdm
    for filename in wrapper(files, ascii=True):
        image_pil = Image.open(os.path.join(path, filename)).convert("RGB")
        width = image_pil.width
        height = image_pil.height
        width = width - width % _crop_size
        height = height - height % _crop_size
        for x in range(0, width - _crop_size, _crop_stride):
            for y in range(0, height - _crop_size, _crop_stride):
                image_crop = image_pil.crop((x, y, x + _crop_size, y + _crop_size))
                ret.append(image_crop)
    if progress:
        print("Preparing data... Completed")
    return ret


def collect_image(path, progress=False):
    # get all files
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    ret = []
    # open all files
    wrapper = tqdm_wrapper
    if progress:
        print("Preparing data...")
        wrapper = tqdm
    for filename in wrapper(files, ascii=True):
        image_pil = Image.open(os.path.join(path, filename)).convert("RGB")
        width, height=image_pil.size
        image_crop = image_pil.crop((0, 0, width-(width%3), height-(height%3)))
        ret.append(image_crop)
    if progress:
        print("Preparing data... Completed")
    return ret


class SRDataset(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        hr = self.data[idx]
        # hr = random_crop(hr)
        width, height=hr.size
        lr = TF.resize(hr, size=(height//3,width//3), interpolation=transforms.InterpolationMode.BICUBIC)
        return TF.to_tensor(lr), TF.to_tensor(hr)
