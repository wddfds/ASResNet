import argparse
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from data import *
from loss import FftLoss
from metrics import psnr, ssim
from model import *


def train(opt):

    dataset_train = MyTrainDataSet(opt.dataset, patch_size=opt.patch_size)
    dataset_test = MyValueDataSet(opt.dataset)
    train_loader = DataLoader(dataset=dataset_train, batch_size=opt.batch_size, shuffle=True, drop_last=False, pin_memory=True, num_workers=6)
    val_loader = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False, drop_last=False, pin_memory=True, num_workers=6)

    model = Generator()
    model.to(opt.device)
    if opt.device == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW([{'params': model.parameters(), "lr": opt.lr, "betas": [0.9, 0.999]}])
    cosinese = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch, eta_min=opt.lr_min)

    criterion_l1 = nn.L1Loss().to(opt.device)
    criterion_fft = FftLoss()
    best_psnr = 0
    best_epoch = 0
    start_epoch = 0

    # Checkpoint loading
    checkpoint_path = os.path.join(output_path, 'checkpoint.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict)
        # Remove 'module.' prefix
        # new_state_dict = {'module.' + k : v for k, v in state_dict.items()}
        # model.load_state_dict(new_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_psnr = checkpoint['best_psnr']
        print(f"Resuming training from epoch {start_epoch} with best PSNR {best_psnr}")

    print(f'project_name:{opt.project}\tdataset:{opt.dataset}\tepoch:{opt.epoch}\tbatch_size:{opt.batch_size}\tlr:{opt.lr}--{opt.lr_min}\n')
    for epoch in range(start_epoch, opt.epoch):
        model.train(True)
        torch.cuda.empty_cache()

        print(f"epoch:{epoch+1}")
        for input_img, target in tqdm(train_loader, file=sys.stdout):
            input_img, target = input_img.to(opt.device), target.to(opt.device)

            optimizer.zero_grad()
            output = model(input_img)

            target_2 = F.interpolate(target, scale_factor=0.5)
            target_4 = F.interpolate(target, scale_factor=0.25)

            l1_loss = criterion_l1(output[2], target) + criterion_l1(output[1], target_2) + criterion_l1(output[0], target_4)
            fft_loss = criterion_fft(output[2], target) + criterion_fft(output[1], target_2) + criterion_fft(output[0], target_4)

            loss = l1_loss + 0.1 * fft_loss
            loss.backward()

            lr = optimizer.param_groups[0]['lr']
            optimizer.step()

        cosinese.step()
        torch.save(model.module.state_dict(), output_path + '/last.pth') if opt.device == 'cuda' and torch.cuda.device_count() > 1 else torch.save(model.state_dict(), output_path + '/last.pth')
        print(f"total_loss:{loss:.8f}\tlr:{lr:.8f} ")


        psnr_val = []
        ssim_val = []

        model.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():

            for image, target, img_name in tqdm(val_loader, file=sys.stdout):
                image, target = image.to(opt.device), target.to(opt.device)

                output = model(image)
                pred = output[2]

                if opt.save_image:
                    save_path = output_path + '/eval/' + img_name[0]
                    save_image(pred, save_path)

                psnr1 = psnr(pred, target)
                ssim1 = ssim(pred, target).item()
                psnr_val.append(psnr1)
                ssim_val.append(ssim1)

        avg_psnr = np.mean(psnr_val)
        avg_ssim = np.mean(ssim_val)
        print(f"PSNR:{avg_psnr:.4f} SSIM:{avg_ssim:.4f}")

        # save best
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_epoch = epoch+1
            torch.save(model.module.state_dict(), output_path + '/best.pth') if opt.device == 'cuda' and torch.cuda.device_count() > 1 else torch.save(model.state_dict(), output_path + '/best.pth')
            print(f"model saved at epoch:{epoch + 1}\n")

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.module.state_dict() if opt.device == 'cuda' and torch.cuda.device_count() > 1 else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_psnr': best_psnr,
        }, checkpoint_path)

    print(f"\n{opt.project} training finished!")
    print(f"Max psnr:{best_psnr:.4f} at epoch {best_epoch}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default='AAA', help="project name")
    parser.add_argument("--dataset", type=str, default='E:/smr/datasets/ITS', help="dataset path")
    parser.add_argument("--patch_size", type=int, default=256, help="size of the patch")
    parser.add_argument("--epoch", type=int, default=1000, help="starting epoch")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0004, help="initial learning rate")
    parser.add_argument("--lr_min", type=float, default=0.000001, help="initial learning rate")
    parser.add_argument("--device", type=str, default="cuda", help='GPU name')
    parser.add_argument("--save_image", type=bool, default=False, help="whether save the evaluate result")
    opt = parser.parse_args()

    output_path = os.path.join('experiment/', opt.project)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(output_path + '/eval'):
        os.mkdir(output_path + '/eval')

    train(opt)
