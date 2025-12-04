import torch
import torch.nn as nn


class FftLoss(nn.Module):
    def __init__(self):
        super(FftLoss, self).__init__()

        self.l1 = nn.L1Loss()

    def forward(self, restored_image, gt_image):

        label_fft1 = torch.fft.fft2(gt_image, dim=(-2, -1))
        label_fft1 = torch.stack((label_fft1.real, label_fft1.imag), -1)

        pred_fft1 = torch.fft.fft2(restored_image, dim=(-2, -1))
        pred_fft1 = torch.stack((pred_fft1.real, pred_fft1.imag), -1)

        return self.l1(label_fft1, pred_fft1)

