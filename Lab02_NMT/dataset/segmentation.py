import json
import os
import random

import cv2
import numpy as np
from torch.utils.data import Dataset

from inference_utils import order_pts, compute_max_wh

TRAIN_SIZE = 0.8


class DetectionDataset(Dataset):
    def __init__(self, data_path, config_file=None, transforms=None, split="train"):
        super(DetectionDataset, self).__init__()
        self.data_path = data_path
        self.transforms = transforms
        self.split = split

        self.image_filenames, self.mask_filenames, self.nums = self._parse_root_(config_file)
        if self.split is not None:
            train_size = int(len(self.image_filenames) * TRAIN_SIZE)
            if self.split == "train":
                self.image_filenames = self.image_filenames[:train_size]
                self.mask_filenames = self.mask_filenames[:train_size]
            elif split == "val":
                self.image_filenames = self.image_filenames[train_size:]
                self.mask_filenames = self.mask_filenames[train_size:]
            else:
                raise NotImplementedError(split)

    @staticmethod
    def _parse_root_(config_file):
        with open(config_file, "rt") as f:
            config = json.load(f)
        image_filenames, mask_filenames, nums = [], [], []
        for item in config:
            if "mask" in item:  # handling bad files during transfer
                image_filenames.append(item["file"])
                mask_filenames.append(item["mask"])
                nums.append(item['nums'])

        assert len(image_filenames) == len(mask_filenames), "Images and masks lengths mismatch"
        return image_filenames, mask_filenames, nums

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, item):
        image_filename = os.path.join(self.data_path, self.image_filenames[item])
        mask_filename = os.path.join(self.data_path, self.mask_filenames[item])
        nums = self.nums[item]

        image = cv2.imread(image_filename).astype(np.float32) / 255.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.

        if len(nums) == 1:
            image, mask = self.license_augment(image, mask, nums)

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        return image.transpose(2, 0, 1), mask

    def license_augment(self, image, mask, nums):
        box = np.array(nums[0]['box'])
        box = order_pts(box)
        max_w, max_h = compute_max_wh(box)

        img_w = image.shape[1]
        if box[:, 1].min() - max_h <= 0 or img_w - max_w <= 0:
            return image, mask

        tl_y = np.random.randint(0, box[:, 1].min() - max_h)
        tl_x = np.random.randint(0, img_w - max_w)
        dstn = np.array([
            [tl_x, tl_y],
            [tl_x + max_w, tl_y],
            [tl_x + max_w, tl_y + max_h],
            [tl_x, tl_y + max_h]
        ], dtype="float32")

        license_dir = os.path.join(self.data_path, 'rec_additional')
        license_fname = random.sample(os.listdir(license_dir), 1)[0]
        license_src = cv2.imread(os.path.join(license_dir, license_fname))
        license_src = cv2.cvtColor(license_src, cv2.COLOR_BGR2RGB)

        lic_height, lic_width = license_src.shape[:-1]
        lic_box = np.array([
            [0, 0],
            [lic_width - 1, 0],
            [lic_width - 1, lic_height - 1],
            [0, lic_height - 1]],
            dtype="float32"
        )
        M = cv2.getPerspectiveTransform(lic_box, dstn)
        rslt = cv2.warpPerspective(license_src, M, image.shape[1::-1])
        image = (rslt == 0) * image + rslt
        mask += (rslt != 0) * 255
        return image, mask


