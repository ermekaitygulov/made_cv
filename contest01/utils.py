import os

import cv2
import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils import data

from transforms import NUM_PTS, CROP_SIZE

np.random.seed(1234)
torch.manual_seed(1234)

TRAIN_SIZE = 0.8
WIDTH_BINS = [0, 148, 231, 1192]
SUBMISSION_HEADER = "file_name,Point_M0_X,Point_M0_Y,Point_M1_X,Point_M1_Y," \
                    "Point_M2_X,Point_M2_Y,Point_M3_X,Point_M3_Y,Point_M4_X," \
                    "Point_M4_Y,Point_M5_X,Point_M5_Y,Point_M6_X,Point_M6_Y," \
                    "Point_M7_X,Point_M7_Y,Point_M8_X,Point_M8_Y,Point_M9_X," \
                    "Point_M9_Y,Point_M10_X,Point_M10_Y,Point_M11_X,Point_M11_Y," \
                    "Point_M12_X,Point_M12_Y,Point_M13_X,Point_M13_Y,Point_M14_X," \
                    "Point_M14_Y,Point_M15_X,Point_M15_Y,Point_M16_X,Point_M16_Y," \
                    "Point_M17_X,Point_M17_Y,Point_M18_X,Point_M18_Y,Point_M19_X," \
                    "Point_M19_Y,Point_M20_X,Point_M20_Y,Point_M21_X,Point_M21_Y," \
                    "Point_M22_X,Point_M22_Y,Point_M23_X,Point_M23_Y,Point_M24_X," \
                    "Point_M24_Y,Point_M25_X,Point_M25_Y,Point_M26_X,Point_M26_Y," \
                    "Point_M27_X,Point_M27_Y,Point_M28_X,Point_M28_Y,Point_M29_X," \
                    "Point_M29_Y\n"


class ThousandLandmarksDataset(data.Dataset):
    def __init__(self, root, transforms, split="train", bad_img_names=None):
        super(ThousandLandmarksDataset, self).__init__()
        self.root = root
        landmark_file_name = os.path.join(root, 'landmarks.csv') if split != "test" \
            else os.path.join(root, "test_points.csv")
        images_root = os.path.join(root, "images")

        self.image_names = []
        self.landmarks = []
        bad_img_names = bad_img_names or []

        with open(landmark_file_name, "rt") as fp:
            num_lines = sum(1 for _ in fp)
        num_lines -= 1  # header

        with open(landmark_file_name, "rt") as fp:
            for i, line in tqdm.tqdm(enumerate(fp), total=num_lines + 1):
                if i == 0:
                    continue  # skip header
                if split == "train" and i == int(TRAIN_SIZE * num_lines):
                    break  # reached end of train part of data
                elif split == "val" and i < int(TRAIN_SIZE * num_lines):
                    continue  # has not reached start of val part of data
                elements = line.strip().split("\t")
                image_name = os.path.join(images_root, elements[0])
                if image_name in bad_img_names:
                    continue
                self.image_names.append(image_name)

                if split in ("train", "val"):
                    landmarks = list(map(np.int, elements[1:]))
                    landmarks = np.array(landmarks, dtype=np.int).reshape((len(landmarks) // 2, 2))
                    self.landmarks.append(landmarks)

        if split in ("train", "val"):
            self.landmarks = torch.as_tensor(self.landmarks)
        else:
            self.landmarks = None

        self.transforms = transforms

    def __getitem__(self, idx):
        sample = {}
        to_transform = {}
        if self.landmarks is not None:
            landmarks = np.array(self.landmarks[idx], dtype='float32')
            torch_landmarks = torch.from_numpy(landmarks)
            sample["original_landmarks"] = torch_landmarks
            to_transform["keypoints"] = landmarks

        image = cv2.imread(self.image_names[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sample["image_name"] = self.image_names[idx]
        sample['original_shape'] = np.array(image.shape)

        if self.transforms is not None:
            to_transform["image"] = image
            transformed = self.transforms(**to_transform)
            assert transformed['image'].shape == (3, CROP_SIZE, CROP_SIZE), (transformed['image'].shape,
                                                                             self.image_names[idx])
            if 'keypoints' in transformed:
                transformed['landmarks'] = transformed.pop('keypoints')
                assert transformed['landmarks'].shape == (NUM_PTS, 2), (transformed['landmarks'].shape,
                                                                        self.image_names[idx])
            sample.update(transformed)
        else:
            sample['image'] = image

        return sample

    def __len__(self):
        return len(self.image_names)


def restore_landmarks(landmarks, f, margins):
    dx, dy = margins
    landmarks[:, 0] += dx
    landmarks[:, 1] += dy
    landmarks /= f
    return landmarks


def restore_landmarks_batch(landmarks, original_shapes, crop_size=128):
    fs = compute_fs(original_shapes, crop_size)
    margins_x, margins_y = compute_margins(original_shapes * fs[:, None], crop_size)

    landmarks[:, :, 0] += margins_x[:, None]
    landmarks[:, :, 1] += margins_y[:, None]
    landmarks /= fs[:, None, None]
    return landmarks


def compute_margins(original_shape, crop_size=128):
    margins_x = (original_shape[:, 1] - crop_size) // 2
    margins_y = (original_shape[:, 0] - crop_size) // 2
    return margins_x, margins_y


def compute_fs(original_shape, crop_size=128):
    if isinstance(original_shape, torch.Tensor):
        min_side = torch.min(original_shape[:, :2], dim=1)[0]
    else:
        min_side = np.min(original_shape[:, :2], axis=1)
    fs = crop_size / min_side
    return fs


def create_submission(path_to_data, test_predictions, path_to_submission_file):
    test_dir = os.path.join(path_to_data, "test")

    output_file = path_to_submission_file
    wf = open(output_file, 'w')
    wf.write(SUBMISSION_HEADER)

    mapping_path = os.path.join(test_dir, 'test_points.csv')
    mapping = pd.read_csv(mapping_path, delimiter='\t')

    for i, row in mapping.iterrows():
        file_name = row[0]
        point_index_list = np.array(eval(row[1]))
        points_for_image = test_predictions[i]
        needed_points = points_for_image[point_index_list].astype(np.int)
        wf.write(file_name + ',' + ','.join(map(str, needed_points.reshape(2 * len(point_index_list)))) + '\n')


def log_val_loss(val_loss, step, writer):
    writer.add_scalar('val loss', np.mean(val_loss), step)


def get_shape_loss(loss, shapes, left, right):
    idx = (left < shapes[:, 1]) * (shapes[:, 1] <= right)
    shape_loss = loss[idx]
    shape_loss = np.mean(shape_loss)
    return shape_loss


def log_shapes_loss(val_loss, shapes, step, writer):
    val_loss = np.concatenate(val_loss)
    shapes = np.concatenate(shapes)

    little_loss = get_shape_loss(val_loss, shapes, WIDTH_BINS[0], WIDTH_BINS[1])
    middle_loss = get_shape_loss(val_loss, shapes, WIDTH_BINS[1], WIDTH_BINS[2])
    big_loss = get_shape_loss(val_loss, shapes, WIDTH_BINS[2], WIDTH_BINS[3])

    writer.add_scalar('little img loss', little_loss, step)
    writer.add_scalar('middle img loss', middle_loss, step)
    writer.add_scalar('big img loss', big_loss, step)


def print_val_result(val_loss, shapes):
    val_loss = np.concatenate(val_loss)
    shapes = np.concatenate(shapes)

    little_loss = get_shape_loss(val_loss, shapes, WIDTH_BINS[0], WIDTH_BINS[1])
    middle_loss = get_shape_loss(val_loss, shapes, WIDTH_BINS[1], WIDTH_BINS[2])
    big_loss = get_shape_loss(val_loss, shapes, WIDTH_BINS[2], WIDTH_BINS[3])

    print(f'All: {val_loss.mean():5.2}\t little: {little_loss:5.2}\t middle: {middle_loss:5.2}\t big: {big_loss:5.2}')
