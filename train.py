import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torchvision import transforms

import argparse
import logging
import os
import pprint

import numpy as np
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import yaml

from dataset.dataset import SkinDataset, DSBDataset, BUSIDataset
from model.forkmamba import ForkMamba
from utils.metrics import iou_score, dice_score, recall_score, precision_score
from utils.losses import DiceLoss, WeightedConcaveRegionLoss

from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='ForkMamba: Adaptive Multi-branch Visual Mamba with Weighted Concave Region Objective Function for Skin Lesion Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--data-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]
            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count

def evaluate(model, loader):
    model.eval()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    recall_meter = AverageMeter()
    precision_meter = AverageMeter()

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            images, masks, _ = batch
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)

            dice = dice_score(preds, masks)
            iou = iou_score(preds, masks)
            recall = recall_score(preds, masks)
            precision = precision_score(preds, masks)
            dice_meter.update(dice)
            iou_meter.update(iou)
            recall_meter.update(recall)
            precision_meter.update(precision)
            
    return dice_meter.avg, iou_meter.avg, recall_meter.avg, precision_meter.avg


def main():
    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**config, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)
    
    cudnn.enabled = True
    cudnn.benchmark = True
    if config['data'] == 'busi':
        model = ForkMamba(in_channels=1, num_classes=1)
    else:
        model = ForkMamba(in_channels=3, num_classes=1)

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    loss_dict = {
    "DiceLoss": DiceLoss(smooth=1e-5),
    "WCRLoss": WeightedConcaveRegionLoss(alpha=config.get("alpha", 0.7), beta=config.get("beta", 1.0), b=config.get("b", 10), c=config.get("c", 10)),
    }

    allowed_losses = {"DiceLoss", "TverskyLoss", "FocalTverskyLoss", "BCETverskyLoss", "DiceTverskyLoss"}
    assert config["loss_type"] in allowed_losses, f"Invalid loss_type: {config['loss_type']}. Must be one of {allowed_losses}"

    criterion = loss_dict[config["loss_type"]]

    # Load dataset
    # Validate config['data'] before proceeding
    valid_datasets = ['skin', 'dsb', 'busi']
    if config['data'] not in valid_datasets:
        raise ValueError(
            f"Invalid dataset name '{config['data']}'. "
            f"Please choose one of the following: {valid_datasets}"
        )

    if config['data'] == 'skin':
        data = np.load(args.data_path)
        images, masks = data["image"], data["mask"]

        test_size = int((20 / 100) * images.shape[0])

        x_train, x_val, y_train, y_val = train_test_split(
            images, masks, test_size=test_size, random_state=42
        )

        train_dataset = SkinDataset(x_train, y_train, transform=True, typeData="train")
        val_dataset = SkinDataset(x_val, y_val, transform=False, typeData="val")

    elif config['data'] == 'dsb':
        # Already split into subsets inside the npy file
        train_dataset = DSBDataset(type='train', data_path=args.data_path, transform=False)
        val_dataset = DSBDataset(type='test', data_path=args.data_path, transform=False)

    elif config['data'] == 'busi':
        data = np.load(args.data_path)
        images, masks = data["image"], data["mask"]

        test_size = int((20 / 100) * images.shape[0])

        x_train, x_val, y_train, y_val = train_test_split(
            images, masks, test_size=test_size, random_state=42
        )

        train_dataset = BUSILoader(x_train, y_train, transform=True, typeData="train")
        val_dataset = BUSILoader(x_val, y_val, transform=False, typeData="val")


    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True,
        num_workers=4, 
        drop_last = True,
        sampler=train_sampler
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        drop_last=False,
        sampler=val_sampler
    )

    iters = 0
    total_iters = len(train_loader) * config['epochs']
    best_val_dice = 0.0
    best_epoch = 0
    epoch = -1

    for epoch in range(config["epochs"]):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f} at epoch {:}'.format(
                    epoch, lr, best_val_dice, best_epoch))

        model.train()
        total_loss = AverageMeter()
        train_sampler.set_epoch(epoch)
        
        for i, (image, mask) in enumerate(train_loader):
            image, mask = image.cuda(), mask.cuda()

            pred = model(image)
            loss = criterion(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss)
            iters = epoch * len(trainloader) + i
            lr = config['lr'] * (1 - iters / total_iters) ** 0.9

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            if (i % (max(2, len(trainloader) // 8)) == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss.avg))

        val_dice, val_iou, _, _ = evaluate(model, val_loader)
        if rank == 0:
            logger.info(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")
        is_best = val_dice >= best_val_dice

        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                best_val_dice = val_dice
                best_epoch = epoch
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))
    print("Training complete.")

if __name__ == '__main__':
    main()