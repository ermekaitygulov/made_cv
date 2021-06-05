"""Create recognition crops from train.json (using car plates coordinates & texts)."""

import json
import os
from argparse import ArgumentParser

import cv2
import numpy as np
import tqdm


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--data-dir", help="Path to dir containing 'train/', 'test/', 'train.json'.")
    parser.add_argument("--transform", help="If True, crop & transform box using 4 corner points;"
                                            "crop bounding box otherwise.", action="store_true")
    return parser.parse_args()


def bounding_box_crop(image, box):
    # TODO TIP: Maybe adding some margin could help.
    x_min = np.clip(min(box[:, 0]), 0, image.shape[1])
    x_max = np.clip(max(box[:, 0]), 0, image.shape[1])
    y_min = np.clip(min(box[:, 1]), 0, image.shape[0])
    y_max = np.clip(max(box[:, 1]), 0, image.shape[0])
    return image[y_min: y_max, x_min: x_max]


def order_pts(box):
    box = np.array(box)
    rect = np.zeros((4, 2), dtype="float32")

    s = box.sum(axis=1)
    rect[0] = box[np.argmin(s)]
    rect[2] = box[np.argmax(s)]

    diff = np.diff(box, axis=1)
    rect[1] = box[np.argmin(diff)]
    rect[3] = box[np.argmax(diff)]
    return rect


def compute_max_wh(box):
    (tl, tr, br, bl) = box
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    # ...and now for the height of our new image
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    # take the maximum of the width and height values to reach
    # our final dimensions
    max_width = max(int(width_a), int(width_b))
    max_height = max(int(height_a), int(height_b))
    return max_width, max_height


def add_noise(image, box, noise):
    direction = np.array([
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1]
    ])
    noise = (np.random.random(box.shape) * noise * image.shape[:-1]).astype(int)
    box += direction * noise
    return box


def warp_perspective(image, box, noise=0.05):
    box = order_pts(box)
    box = add_noise(image, box, noise)

    max_width, max_height = compute_max_wh(box)
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]],
        dtype="float32"
    )
    transform = cv2.getPerspectiveTransform(box, dst)
    warp = cv2.warpPerspective(image, transform, (max_width, max_height))
    return warp


def main(args):
    if args.transform:
        # TODO TIP: Maybe useful to crop using corners
        # See cv2.findHomography & cv2.warpPerspective for more
        get_crop = warp_perspective
    else:
        get_crop = bounding_box_crop

    config_filename = os.path.join(args.data_dir, "train.json")
    with open(config_filename, "rt") as fp:
        config = json.load(fp)

    config_recognition = []

    for item in tqdm.tqdm(config):

        image_filename = item["file"]
        image = cv2.imread(os.path.join(args.data_dir, image_filename))
        if image is None:
            continue

        image_base, ext = os.path.splitext(image_filename)

        nums = item["nums"]
        for i, num in enumerate(nums):
            text = num["text"]
            box = np.asarray(num["box"])
            crop_filename = image_base + ".box" + str(i).zfill(2) + ext
            new_item = {"file": crop_filename, "text": text}
            if os.path.exists(crop_filename):
                config_recognition.append(new_item)
                continue

            crop = get_crop(image, box)
            cv2.imwrite(os.path.join(args.data_dir, crop_filename), crop)
            config_recognition.append(new_item)

    output_config_filename = os.path.join(args.data_dir, "train_recognition.json")
    with open(output_config_filename, "wt") as fp:
        json.dump(config_recognition, fp)


if __name__ == "__main__":
    main(parse_arguments())
