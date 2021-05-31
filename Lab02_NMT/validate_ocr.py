# TODO TIP: Segmentation is just one of many approaches to object localization.
import json
import os
from argparse import ArgumentParser

import cv2
import editdistance
import numpy as np
import torch
import tqdm
import yaml

from dataset.segmentation import TRAIN_SIZE
from neural_network import REC_NN_CATALOG, SEG_NN_CATALOG
from inference_utils import prepare_for_segmentation, find_min_box, prepare_for_recognition, warp_perspective, \
    order_pts, filter_data, weight_height_odd


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", dest="config", type=str, default=None, help="path to the config")
    return parser.parse_args()


def read_config(config_name):
    with open(config_name) as fin:
        config = yaml.safe_load(fin)
    return config


def init_model(config, catalog, device):
    model_class = catalog[config['name']]
    model = model_class(**config['params'])
    with open(config['path'], "rb") as fp:
        state_dict = torch.load(fp, map_location='cpu')
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def read_val_json(data_path, json_name='train.json', ):
    with open(os.path.join(data_path, json_name)) as fin:
        train_json = json.load(fin)
    start_idx = int(len(train_json) * TRAIN_SIZE)
    val_json = train_json[start_idx:]
    return val_json


def get_ground_truth(image_info):
    ru_license = filter_data(image_info['nums'])
    if not ru_license:
        return ''
    texts = []
    for text_info in ru_license:
        box = order_pts(text_info['box'])
        _, _, _, bl = box
        text = text_info['text']
        x1 = bl[0]
        texts.append((x1, text))
    texts.sort(key=lambda x: x[0])
    texts = ' '.join([text for _, text in texts])
    return texts


def main(args):
    config = read_config(args.config)

    print("Start inference")
    w, h = config['rec_size']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    segmentation_model = init_model(config['seg_model'], SEG_NN_CATALOG, device)
    recognition_model = init_model(config['rec_model'], REC_NN_CATALOG, device)

    val_images_dir = config['data_path']
    val_json = read_val_json(config['data_path'])
    avg_ed = 0
    n = 0

    for i, image_info in enumerate(tqdm.tqdm(val_json)):
        file_name = image_info['file']
        gt_texts = get_ground_truth(image_info)
        if not gt_texts:
            continue

        image_src = cv2.imread(os.path.join(val_images_dir, file_name))
        if image_src is None:
            continue
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)

        # 1. Segmentation.
        image, k, dw, dh = prepare_for_segmentation(image_src.astype(np.float) / 255.,
                                                    tuple(config['seg_size']))
        x = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
        with torch.no_grad():
            pred = torch.sigmoid(segmentation_model(x.to(device))).squeeze().cpu().numpy()
        mask = (pred >= config['seg_threshold']).astype(np.uint8) * 255

        # 2. Extraction of detected regions.
        boxes = find_min_box(mask)
        if len(boxes) == 0:
            continue

        # 3. Text recognition for every detected bbox.
        pred_texts = []
        license_texts = ''
        for box in boxes:
            box[:, 0] -= dw
            box[:, 1] -= dh
            box /= k
            box[:, 0] = box[:, 0].clip(0, image_src.shape[1] - 1)
            box[:, 1] = box[:, 1].clip(0, image_src.shape[0] - 1)
            box = box.astype(np.int)
            #         crop = image_src[y1: y2, x1: x2, :]

            box = order_pts(box)
            wh_odd = weight_height_odd(box)
            if wh_odd < config['min_wh_odd'] or config['max_wh_odd'] < wh_odd:
                continue
            crop = warp_perspective(image_src, box)

            tensor = prepare_for_recognition(crop, (w, h)).to(device)
            with torch.no_grad():
                text = recognition_model(tensor, decode=True)[0]
            x1 = box[0][0]
            pred_texts.append((x1, text))

        # all predictions must be sorted by x1
        if pred_texts:
            pred_texts.sort(key=lambda x: x[0])
            license_texts = ' '.join([text for _, text in pred_texts])
        avg_ed += editdistance.eval(license_texts, gt_texts)
        n += 1

    avg_ed /= n
    # Generate a submission file
    print(f'AVG ED: {avg_ed}')
    print('Done')


if __name__ == "__main__":
    main(parse_arguments())
