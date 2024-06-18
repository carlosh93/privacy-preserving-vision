from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from Optics import Camera
from Utils import plot_PSFs

device = torch.device('cuda')
camera = Camera(device=device, zernike_terms=300).to(device)
# psfs = camera()
# plot_PSFs(psfs.cpu())

I = Image.open('img.jpg')
I = I.resize((256, 256))
x = torch.tensor(np.array(I), dtype=torch.float, device=device).permute(2, 0, 1).unsqueeze(0)
x = x / x.max()

optimizer = torch.optim.Adam(params=camera.parameters(), lr=5e-3)
criterion = torch.nn.MSELoss()
camera.train()

for i in range(50000):

    optimizer.zero_grad()
    y = camera(x)
    loss = 10 * criterion(x, y)  # torch.mul(x, y).mean() + 4e6 * camera.centering_loss  # + camera.loss_rad * 1.4e4
    # loss -= torch.mean(torch.square(torch.abs(torch.fft.fftn(x - y))), dim=[2, 3]).mean() * 1e-5
    print('Iter: {:.0f}\t Loss: {:.4f}\t {:.4f} \t {:.4f}'.format(i, loss.item(), camera.Zer_train[0].item(), camera.Zer_train[-1].item()))
    # camera.Zer_train.data.clamp_(-1, 1)
    loss.backward()
    optimizer.step()

    if i % 1000 == 0:
        # torch.save({'models': basic_model.state_dict()}, './Model.pth')
        res = y.cpu().detach().permute(0, 2, 3, 1)[0]
        org = x.cpu().detach().permute(0, 2, 3, 1)[0]
        psf = camera.psfs
        plt.subplot(231)
        plt.title('Iter = {:.0f}'.format(i))
        plt.imshow(res / res.max()), plt.axis('off')
        plt.subplot(232)
        plt.imshow(org / org.max()), plt.axis('off')
        plt.subplot(233)
        plt.imshow((psf / psf.max()).squeeze().permute(1, 2, 0).cpu().data.numpy()), plt.axis('off')
        plt.subplot(234)
        plt.imshow(camera.get_Heith_Map().squeeze().cpu().data, cmap='jet'), plt.colorbar(), plt.axis('off')
        plt.subplot(235)
        plt.imshow(camera.ca.squeeze().cpu().data, cmap='gray'), plt.colorbar(), plt.axis('off')
        plt.show()

torch.save({'camera': camera.state_dict()}, 'Cam_focus.pth')
