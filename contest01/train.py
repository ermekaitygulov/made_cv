"""Script for baseline training. Model is ResNet18 (pretrained on ImageNet). Training takes ~ 15 mins (@ GTX 1080Ti)."""

import os
import pickle
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import tqdm
from torch.nn import functional as fnn
from torch.utils.data import DataLoader

from models import CATALOG
from transforms import NUM_PTS, CROP_SIZE, ToTensorV3
from utils import ThousandLandmarksDataset
from utils import restore_landmarks_batch, create_submission, log_val_loss, log_shapes_loss, print_val_result
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
    parser.add_argument("--model_name", help="Model from catalog.", default='resnet')
    parser.add_argument("--model_path", help="Path to model.", default=None)
    parser.add_argument("--evaluate_only", help="Evaluate only.", action='store_true')
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
    train_loss = np.mean(train_loss)
    print(f"Train loss: {train_loss:5.2}")


def validate(model, loader, loss_fn, device, writer, epoch, best_val_loss=None):
    model.eval()
    val_loss = []
    shapes = []
    for i, batch in enumerate(tqdm.tqdm(loader, total=len(loader), desc="validation...")):
        images = batch["image"].to(device)
        landmarks = batch["original_landmarks"]
        landmarks = landmarks.view(landmarks.shape[0], -1)
        original_shapes = batch['original_shape']

        with torch.no_grad():
            pred_landmarks = model(images).to('cpu').view(-1, NUM_PTS, 2)
        pred_landmarks = restore_landmarks_batch(pred_landmarks, original_shapes, CROP_SIZE).view(-1, NUM_PTS * 2)
        loss = loss_fn(pred_landmarks, landmarks, reduction="none").numpy()

        val_loss.append(loss)
        shapes.append(original_shapes.numpy())
        if (i + 1) % 10 == 0:
            log_val_loss(val_loss[-10:], i + epoch * len(loader), writer)
            log_shapes_loss(val_loss[-10:], shapes[-10:], i + epoch * len(loader), writer)
            writer.flush()
    print_val_result(val_loss, shapes)
    if best_val_loss and np.concatenate(val_loss).mean() < best_val_loss:
        best_val_loss = np.concatenate(val_loss).mean()
        with open(os.path.join("runs", f"{args.name}_best.pth"), "wb") as fp:
            torch.save(model.state_dict(), fp)
    return best_val_loss


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


def prepare_data(args, transforms, split, **dataloader_kwargs):
    with open('bad_images.bd') as fin:
        bad_img_names = fin.readlines()
        bad_img_names = [i.strip() for i in bad_img_names]
    if split in ['train', 'val']:
        path = os.path.join(args.data, "train")
    else:
        path = os.path.join(args.data, "test")
    dataset = ThousandLandmarksDataset(path, transforms, split=split,
                                       bad_img_names=bad_img_names)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True,
                            **dataloader_kwargs)
    return dataloader


def init_model(args):
    print("Creating model...")
    model = CATALOG[args.model_name](2 * NUM_PTS)
    device = torch.device("cuda:0") if args.gpu and torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    if args.model_path:
        with open(args.model_path, "rb") as fp:
            state_dict = torch.load(fp, map_location="cpu")
            model.load_state_dict(state_dict)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
    loss_fn = fnn.mse_loss
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=3e-3, step_size_up=5,
                                            mode='triangular2')
    writer = SummaryWriter(log_dir=os.path.join('runs', f'{args.name}_tb'))
    return model, optimizer, loss_fn, scheduler, writer, device


def main(args):
    os.makedirs("runs", exist_ok=True)

    # 1. prepare data & models
    train_transforms = A.Compose([
        A.SmallestMaxSize(CROP_SIZE),
        A.CenterCrop(CROP_SIZE, CROP_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV3()
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    test_transforms = A.Compose([
        A.SmallestMaxSize(CROP_SIZE),
        A.CenterCrop(CROP_SIZE, CROP_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV3()
    ])

    print("Reading data...")
    train_dataloader = prepare_data(args, train_transforms, 'train', shuffle=True, drop_last=True)
    val_dataloader = prepare_data(args, train_transforms, 'val', shuffle=False, drop_last=False)

    print("Init model")
    model, optimizer, loss_fn, scheduler, writer, device = init_model(args)

    # 2. train & validate
    if args.evaluate_only:
        validate(model, val_dataloader, loss_fn, device=device, writer=writer, epoch=0)
        return
    print("Ready for training...")
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        print(f"Epoch #{epoch:2}:")
        train(model, train_dataloader, loss_fn, optimizer, device=device, writer=writer, epoch=epoch)
        best_val_loss = validate(model, val_dataloader, loss_fn, device=device,
                                 writer=writer, epoch=epoch, best_val_loss=best_val_loss)

        scheduler.step()
        writer.add_scalar('learning rate', optimizer.param_groups[0]["lr"], epoch)
        writer.flush()

    # 3. predict
    test_dataset = ThousandLandmarksDataset(os.path.join(args.data, "test"), test_transforms, split="test")
    test_dataloader = prepare_data(args, test_transforms, 'test', shuffle=False, drop_last=False)

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
