import torch
from torch import nn
from .Utils import *
import matplotlib.pyplot as plt
import numpy as np
import poppy


class Camera(nn.Module):
    def __init__(self, device="cpu", N=256, lamdas=3, colors_CA=3, layers=16, zernike_terms=50):
        super(Camera, self).__init__()

        self.zi = 48e-3
        # self.z0 = 2.
        # self.f = 1 / (1 / self.zi + 1 / self.z0)
        self.f = 50e-3
        self.R = self.f * deta(torch.tensor(550e-9 * 1e6))
        self.radii = 2.58e-3
        self.D = layers  # depth values
        self.colors_CA = colors_CA
        self.pi = torch.tensor([np.pi], device=device)
        self.lamdas = lamdas
        self.device = device

        # -----parameter lens and sensor---------#

        self.N = N  # number of samples
        self.c = N // 2
        self.L_len = 2 * self.radii * 2
        # self.px = 6.22e-6
        self.px = 3.713103e-6  # For camera Mirrorless
        self.L_sen = self.px * self.N

        self.lamb = torch.tensor([640, 550, 440], device=device) * 1.e-9
        self.lamb = self.lamb.unsqueeze(-1).unsqueeze(-1)
        self.flmb = self.R / deta(self.lamb * 1e6)
        self.k = 2 * self.pi / self.lamb

        self.z = 1. / torch.linspace(1 / 0.5, 1 / 10, 16)
        self.z = torch.tensor([self.z[9]])

        self.du = self.L_len / self.N
        self.u = torch.arange(-1 * self.L_len / 2, self.L_len / 2, self.du, device=device)
        [self.X, self.Y] = torch.meshgrid(self.u, self.u, indexing="ij")
        self.XY = (self.X * self.X + self.Y * self.Y)
        [self.r, self.thetha] = cart2pol(self.X, self.Y)
        self.rad = self.r <= self.radii

        self.fx1 = torch.arange(-1 / (2 * self.du), 1 / (2 * self.du), 1 / self.L_len, device=device)
        self.fx1 = torch.fft.fftshift(self.fx1)
        [self.FX1, self.FY1] = torch.meshgrid(self.fx1, self.fx1, indexing="ij")
        self.FF = self.FX1 * self.FX1 + self.FY1 * self.FY1

        self.dx2 = self.L_sen / self.N
        self.x2 = torch.arange(-1 * self.L_sen / 2, self.L_sen / 2, self.dx2, device=device)
        [self.X2, self.Y2] = torch.meshgrid(self.x2, self.x2, indexing="ij")
        self.XY2 = self.X2 * self.X2 + self.Y2 * self.Y2
        [self.r2, self.thetha2] = cart2pol(self.X2, self.Y2)
        self.rho = self.r2 > self.px * 16

        # --- Adding Zernike Parameters --- #

        self.zernike_inits = torch.rand((zernike_terms, 1, 1), device=self.device) / 100
        self.zernike_inits[:3] = 0
        self.Zer_no_train = nn.Parameter(self.zernike_inits[:3, ...], requires_grad=False)
        self.Zer_train = nn.Parameter(self.zernike_inits[3:, ...], requires_grad=True)

        self.zernike_volume = get_zernike_volume(resolution=self.N, n_terms=zernike_terms)
        self.zernike_volume = torch.tensor(self.zernike_volume, dtype=torch.float32, device=self.device)

        # --- Adding Coded Aperture Parameters --- #
        size = (1, 1, 32, 32)
        # self.ca = torch.where(torch.rand(size=size) > 0.5, torch.ones(size), torch.zeros(size))
        self.ca = torch.rand((1, 1, 64, 64), device=self.device)
        self.ca = nn.Parameter(self.ca, requires_grad=True)

    def get_Heith_Map(self):
        zernike_coeffs_concat = torch.cat((self.Zer_no_train, self.Zer_train), 0)
        height_map = torch.sum(zernike_coeffs_concat * self.zernike_volume, dim=0)
        return height_map.unsqueeze(0)

    def get_phase_shift(self):
        return self.k * self.flmb * self.get_Heith_Map()

    def get_psf(self):
        self.psfs = None
        ca = nn.functional.interpolate(self.ca, (self.N, self.N)).squeeze(0)
        for dis in self.z:
            t = compl_exp(-(self.k / (2 * self.flmb)) * self.XY) * ca
            focus = compl_exp((self.k / (2 * dis)) * self.XY)

            ph = torch.mul(self.rad, torch.mul(t, focus)) * compl_exp(self.get_phase_shift())

            vu = torch.mul(ph, compl_exp((self.pi / (self.lamb * self.zi * self.L_len) * (self.L_len - self.L_sen)) * self.XY))
            vu = torch.fft.fft2(torch.fft.fftshift(vu, dim=(-2, -1)))

            vu = torch.mul(vu, compl_exp(-(self.pi * self.lamb * self.zi * self.L_len / self.L_sen) * self.FF))

            vu = torch.fft.ifftshift(torch.fft.ifft2(vu), dim=(-2, -1))
            vu = (self.L_sen / self.L_len) * torch.multiply(vu, compl_exp(-(self.pi / (self.lamb * self.zi * self.L_sen) *
                                                                            (self.L_len - self.L_sen)) * self.XY2))

            psf = torch.square(torch.abs(vu * ((self.du * self.du) / (self.dx2 * self.dx2))))
            psf = psf / torch.sum(psf)

            if self.psfs is not None:
                self.psfs = torch.cat([self.psfs, psf.unsqueeze(0)], dim=0)
            else:
                self.psfs = psf.unsqueeze(0)

        return self.psfs

    def forward(self, img):
        psf = self.get_psf()
        psf = torch.roll(psf, shifts=(-self.c, -self.c), dims=(-2, -1))
        img_sensor = conv2D(img, psf)
        return img_sensor/img_sensor.max()
