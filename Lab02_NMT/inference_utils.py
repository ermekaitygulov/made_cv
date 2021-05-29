import logging
import cv2
import numpy as np
import torch


def get_logger(filename: str) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s', '%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def prepare_for_segmentation(image, fit_size):
    """
    Scale proportionally image into fit_size and pad with zeroes to fit_size
    :return: np.ndarray image_padded shaped (*fit_size, 3), float k (scaling coef), float dw (x pad), dh (y pad)
    """
    # pretty much the same code as segmentation.transforms.Resize
    h, w = image.shape[:2]
    k = fit_size[0] / max(w, h)
    image_fitted = cv2.resize(image, dsize=None, fx=k, fy=k)
    h_, w_ = image_fitted.shape[:2]
    dw = (fit_size[0] - w_) // 2
    dh = (fit_size[1] - h_) // 2
    image_padded = cv2.copyMakeBorder(image_fitted, top=dh, bottom=dh, left=dw, right=dw,
                                      borderType=cv2.BORDER_CONSTANT, value=0.0)
    if image_padded.shape[0] != fit_size[1] or image_padded.shape[1] != fit_size[0]:
        image_padded = cv2.resize(image_padded, dsize=fit_size)
    return image_padded, k, dw, dh


def get_boxes_from_mask(mask, margin, clip=False):
    """
    Detect connected components on mask, calculate their bounding boxes (with margin) and return them (normalized).
    If clip is True, cutoff the values to (0.0, 1.0).
    :return np.ndarray boxes shaped (N, 4)
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    boxes = []
    for j in range(1, num_labels):  # j = 0 == background component
        x, y, w, h = stats[j][:4]
        x1 = int(x - margin * w)
        y1 = int(y - margin * h)
        x2 = int(x + w + margin * w)
        y2 = int(y + h + margin * h)
        box = np.asarray([x1, y1, x2, y2])
        boxes.append(box)
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes).astype(np.float)
    boxes[:, [0, 2]] /= mask.shape[1]
    boxes[:, [1, 3]] /= mask.shape[0]
    if clip:
        boxes = boxes.clip(0.0, 1.0)
    return boxes


def prepare_for_recognition(image, output_size):
    image = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)
    image = image.astype(np.float) / 255.
    return torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)


def find_min_box(mask, margin=0.01):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    direction = (np.array([
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1]
    ]) * margin * mask.shape).astype(int)
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box += direction
        boxes.append(box)
    return np.array(boxes)


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


def warp_perspective(image, box):
    box = order_pts(box)
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