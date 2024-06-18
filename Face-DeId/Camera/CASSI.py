import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from PIL import Image
import urllib.request
import matplotlib.pyplot as plt


class CASSI(nn.Module):
    def __init__(self, M=256, N=256, L=15, Tram=0.5, device='cpu'):
        super().__init__()

        self.M = M
        self.N = N
        self.L = L
        self.Tram = Tram
        self.device = device

        self.ca = nn.Parameter(self.generate_code(), requires_grad=False)

    def generate_code(self):
        code = np.random.rand(self.M, self.N)
        code = 1. * (code < self.Tram)
        code = torch.tensor(code, dtype=torch.float32, device=self.device)
        code = torch.unsqueeze(code, 0)
        return code

    def forward(self, inputs):

        y = torch.multiply(inputs, self.ca)

        # ------ Dispersion -------
        C = inputs.size(1)  # Number of color channels
        paddings = (0, 0, 0, C - 1, 0, 0)
        y = F.pad(y, paddings, mode="constant")

        y = torch.stack([torch.roll(temp, shifts=i, dims=2) for temp, i in zip(torch.unbind(y, dim=1), range(C))], dim=1)
        # ------ Integration ------
        y = torch.mean(y, dim=1)

        rgb = torch.zeros_like(y).unsqueeze(1).repeat((1, 3, 1, 1))
        rgb[:, 0, ::2, ::2] = y[:, ::2, ::2]
        rgb[:, 1, ::2, 1::2] = y[:, ::2, 1::2]
        rgb[:, 2, 1::2, 1::2] = y[:, 1::2, 1::2]

        return rgb[:, :, :-2, :]


'''url = "https://www.ripponmedicalservices.co.uk/images/easyblog_articles/89/b2ap3_large_ee72093c-3c01-433a-8d25-701cca06c975.jpg"
urllib.request.urlretrieve(url, "geeksforgeeks.png")

# Opening the image and displaying it (to confirm its presence)
img = Image.open(r"geeksforgeeks.png")
img = np.asarray(img.resize((256, 256)))
img_t = torch.from_numpy(img[:, :, :3]).permute(2, 0, 1)
img_t = torch.stack([img_t, img_t], dim=0)
cam = CASSI()
measure = cam(img_t)
plt.imshow(measure[0].permute(1, 2, 0).squeeze()), plt.axis('off'), plt.show()'''
