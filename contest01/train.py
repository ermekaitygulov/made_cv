"""Script for baseline training. Model is ResNet18 (pretrained on ImageNet). Training takes ~ 15 mins (@ GTX 1080Ti)."""

import os
import pickle
import sys
from argparse import ArgumentParser

import numpy as np
import cv2
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import tqdm
from torch.nn import functional as fnn
from torch.utils.data import DataLoader

from models import MyResNet
from transforms import NUM_PTS, CROP_SIZE, FaceHorizontalFlip, ToTensorV3
from utils import ThousandLandmarksDataset
from utils import restore_landmarks_batch, create_submission
import albumentations as A

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--name", "-n", help="Experiment name (for saving checkpoints and submits).",
                        default="baseline")
    parser.add_argument("--data", "-d", help="Path to dir with target images & landmarks.", default=None)
    parser.add_argument("--batch-size", "-b", default=512, type=int)  # 512 is OK for resnet18 finetuning @ 3GB of VRAM
    parser.add_argument("--epochs", "-e", default=1, type=int)
    parser.add_argument("--learning-rate", "-lr", default=1e-3, type=float)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--model_path", "-m", help="Path to model.", default=None)
    return parser.parse_args()


def train(model, loader, loss_fn, optimizer, device, writer, epoch):
    model.train()
    train_loss = []
    for i, batch in enumerate(tqdm.tqdm(loader, total=len(loader), desc="training...")):
        images = batch["image"].to(device)  # B x 3 x CROP_SIZE x CROP_SIZE
        landmarks = batch["landmarks"].to(device)  # B x (2 * NUM_PTS)
        landmarks = landmarks.view(landmarks.shape[0], -1)

        pred_landmarks = model(images)  # B x (2 * NUM_PTS)
        loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 10 == 0:
            writer.add_scalar('train loss', np.mean(train_loss[-10:]), i + epoch * len(loader))
            writer.flush()
    return np.mean(train_loss)


def validate(model, loader, loss_fn, device, writer, epoch):
    model.eval()
    val_loss = []
    for i, batch in enumerate(tqdm.tqdm(loader, total=len(loader), desc="validation...")):
        images = batch["image"].to(device)
        landmarks = batch["original_landmarks"]
        landmarks = landmarks.view(landmarks.shape[0], -1)
        original_shapes = batch['original_shape']

        with torch.no_grad():
            pred_landmarks = model(images).to('cpu').view(-1, NUM_PTS, 2)
        pred_landmarks = restore_landmarks_batch(pred_landmarks, original_shapes, CROP_SIZE).view(-1, NUM_PTS * 2)
        loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
        val_loss.append(loss.item())
        if (i + 1) % 10 == 0:
            writer.add_scalar('val loss', np.mean(val_loss[-10:]), i + epoch * len(loader))
            writer.flush()
    return np.mean(val_loss)


def predict(model, loader, device):
    model.eval()
    predictions = np.zeros((len(loader.dataset), NUM_PTS, 2))
    for i, batch in enumerate(tqdm.tqdm(loader, total=len(loader), desc="test prediction...")):
        images = batch["image"].to(device)

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
        pred_landmarks = pred_landmarks.numpy().reshape((len(pred_landmarks), NUM_PTS, 2))  # B x NUM_PTS x 2

        original_shapes = np.array(batch['original_shape'])
        prediction = restore_landmarks_batch(pred_landmarks, original_shapes, CROP_SIZE)  # B x NUM_PTS x 2
        predictions[i * loader.batch_size: (i + 1) * loader.batch_size] = prediction

    return predictions


def init_model():
    print("Creating model...")
    model = MyResNet(2 * NUM_PTS)
    return model


def main(args):
    os.makedirs("runs", exist_ok=True)

    # 1. prepare data & models
    train_transforms = A.Compose([
        A.RandomBrightness(limit=0.2, p=0.5),
        A.RandomContrast(limit=0.2, p=0.5),
        A.Blur(blur_limit=3, p=0.5),
        A.SmallestMaxSize(128),
        A.CenterCrop(128, 128),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV3()
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    test_transforms = A.Compose([
        A.SmallestMaxSize(128),
        A.CenterCrop(128, 128),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV3()
    ])

    print("Reading data...")
    with open('bad_images.bd') as fin:
        bad_img_names = fin.readlines()
        bad_img_names = [i.strip() for i in bad_img_names]
    train_dataset = ThousandLandmarksDataset(os.path.join(args.data, "train"), train_transforms, split="train",
                                             bad_img_names=bad_img_names)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True,
                                  shuffle=True, drop_last=True)
    val_dataset = ThousandLandmarksDataset(os.path.join(args.data, "train"), train_transforms, split="val",
                                           bad_img_names=bad_img_names)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True,
                                shuffle=False, drop_last=False)

    device = torch.device("cuda:0") if args.gpu and torch.cuda.is_available() else torch.device("cpu")
    model = init_model()
    model.to(device)
    if args.model_path:
        with open(args.model_path, "rb") as fp:
            state_dict = torch.load(fp, map_location="cpu")
            model.load_state_dict(state_dict)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
    loss_fn = fnn.mse_loss
    writer = SummaryWriter(log_dir=os.path.join('runs', f'{args.name}_tb'))

    # 2. train & validate
    print("Ready for training...")
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        train_loss = train(model, train_dataloader, loss_fn, optimizer, device=device, writer=writer, epoch=epoch)
        val_loss = validate(model, val_dataloader, loss_fn, device=device, writer=writer, epoch=epoch)
        print("Epoch #{:2}:\ttrain loss: {:5.2}\tval loss: {:5.2}".format(epoch, train_loss, val_loss))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with open(os.path.join("runs", f"{args.name}_best.pth"), "wb") as fp:
                torch.save(model.state_dict(), fp)

    # 3. predict
    test_dataset = ThousandLandmarksDataset(os.path.join(args.data, "test"), test_transforms, split="test")
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                                 shuffle=False, drop_last=False)

    with open(os.path.join("runs", f"{args.name}_best.pth"), "rb") as fp:
        best_state_dict = torch.load(fp, map_location="cpu")
        model.load_state_dict(best_state_dict)

    test_predictions = predict(model, test_dataloader, device)
    with open(os.path.join("runs", f"{args.name}_test_predictions.pkl"), "wb") as fp:
        pickle.dump({"image_names": test_dataset.image_names,
                     "landmarks": test_predictions}, fp)

    create_submission(args.data, test_predictions, os.path.join("runs", f"{args.name}_submit.csv"))


if __name__ == "__main__":
    args = parse_arguments()
    sys.exit(main(args))
