import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)




def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

class SSIM_PL4N(torch.nn.Module):
    "A loss layer that calculates (1-SSIM) loss. Assuming bottom[0] is output data and bottom[1] is label, meaning no back-propagation to bottom[1]."

    def __init__(self):
        super(SSIM_PL4N, self).__init__()
        self.c1 = 0.01 ** 2
        self.c2 = 0.03 ** 2
        self.sigma = 5.

    def reshape(self, img1):
        width = img1.size()
        self.w = torch.exp(-1.*torch.arange(-(width/2), width/2+1)**2/(2*self.sigma**2))
        self.w = torch.outer(self.w, self.w.reshape((width, (1))))	# extend to 2D
        self.w = self.w/torch.sum(self.w)							# normailization
        self.w = torch.reshape(self.w, (1, 1, width, width)) 		# reshape to 4D
        self.w = torch.tile(self.w, (img1.size(0), 3, 1, 1))

    def forward(ctx, self, img1, img2):
        self.mux = torch.sum(self.w * img1, dim=[2,3], keepdim=True)
        self.muy = torch.sum(self.w * img2, dim=[2,3], keepdim=True)
        self.sigmax2 = torch.sum(self.w * img1 ** 2, dim=[2,3], keepdim=True) - self.mux **2
        self.sigmay2 = torch.sum(self.w * img2 ** 2, dim=[2,3], keepdim=True) - self.muy **2
        self.sigmaxy = torch.sum(self.w * img1 * img2, dim=[2,3], keepdim=True) - self.mux * self.muy
        self.l = (2 * self.mux * self.muy + self.C1)/(self.mux ** 2 + self.muy **2 + self.C1)
        self.cs = (2 * self.sigmaxy + self.C2)/(self.sigmax2 + self.sigmay2 + self.C2)
        ctx.save_for_backward(img1, img2)
        return torch.sum(self.l * self.cs)/(img1.size(1) * img1.size(0))

    @staticmethod
    def backward(ctx, grad_output, self):
        img1, img2 = ctx.saved_tensors
        grad_input = grad_output.clone()
        self.dl = 2 * self.w * (self.muy - self.mux * self.l) / (self.mux**2 + self.muy**2 + self.C1)
        self.dcs = 2 / (self.sigmax2 + self.sigmay2 + self.C2) * self.w * ((img2 - self.muy) - self.cs * (img1 - self.mux))
        grad_input = -(self.dl * self.cs + self.l * self.dcs)/(img1.size(1) * img1.size(0))
        return grad_input  # negative sign due to -dSSIM
