# TODO TIP: Segmentation is just one of many approaches to object localization.
import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import tqdm
import yaml

from neural_network import REC_NN_CATALOG, SEG_NN_CATALOG
from inference_utils import prepare_for_segmentation, find_min_box, prepare_for_recognition, warp_perspective


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


def main(args):
    config = read_config(args.config)

    print("Start inference")
    w, h = config['size']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    segmentation_model = init_model(config['seg_model'], SEG_NN_CATALOG, device)
    recognition_model = init_model(config['rec_model'], REC_NN_CATALOG, device)

    test_images_dirname = os.path.join(config['data_path'], "test")
    results = []
    files = os.listdir(test_images_dirname)
    for i, file_name in enumerate(tqdm.tqdm(files)):
        image_src = cv2.imread(os.path.join(test_images_dirname, file_name))
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)

        # 1. Segmentation.
        image, k, dw, dh = prepare_for_segmentation(image_src.astype(np.float) / 255.,
                                                    (256, 256))
        x = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
        with torch.no_grad():
            pred = torch.sigmoid(segmentation_model(x.to(device))).squeeze().cpu().numpy()
        mask = (pred >= config['seg_threshold']).astype(np.uint8) * 255

        # 2. Extraction of detected regions.
        boxes = find_min_box(mask)
        if len(boxes) == 0:
            results.append((file_name, []))
            continue

        # 3. Text recognition for every detected bbox.
        texts = []
        for box in boxes:
            box[:, 0] -= dw
            box[:, 1] -= dh
            box /= k
            box[:, 0] = box[:, 0].clip(0, image_src.shape[1] - 1)
            box[:, 1] = box[:, 1].clip(0, image_src.shape[0] - 1)
            box = box.astype(np.int)
            #         crop = image_src[y1: y2, x1: x2, :]
            crop = warp_perspective(image_src, box)

            tensor = prepare_for_recognition(crop, (w, h)).to(device)
            with torch.no_grad():
                text = recognition_model(tensor, decode=True)[0]
            x1 = box[0][0]
            texts.append((x1, text))

        # all predictions must be sorted by x1
        texts.sort(key=lambda x: x[0])
        results.append((file_name, [w[1] for w in texts]))

    # Generate a submission file
    with open(config['output_file'], "wt") as wf:
        wf.write("file_name,plates_string\n")
        for file_name, texts in sorted(results, key=lambda x: int(os.path.splitext(x[0])[0])):
            wf.write(f"test/{file_name},{' '.join(texts)}\n")
    print('Done')


if __name__ == "__main__":
    main(parse_arguments())
