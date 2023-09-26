from PIL import Image
from pytorch_ssim import *
from Lens import OpticsZernike
import matplotlib.pyplot as plt
import os

device = torch.device("cpu")
camera = OpticsZernike(input_shape=[None, 256, 256, 3], device=device, zernike_terms=350, patch_size=256,
                           height_tolerance=2e-8, sensor_distance=0.025, wave_resolution=[896, 896],
                           sample_interval=3e-06, upsample=False)


if __name__ == '__main__':
    torch.manual_seed(0)  # for repeatable results
    camera.to(device)
    if os.path.isfile('Last_Model.pth'):
        ckpt = torch.load('./Last_Model.pth')
        camera.optics.state_dict()['zernike_coeffs_train'].copy_(
            ckpt['model']['optics.zernike_coeffs_train'])
    camera_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, camera.parameters()),
                                        lr=1e-3)
    camera.train()
    I = Image.open('paisaje.jpeg')
    I = I.resize((256, 256))
    inp = np.array(I)
    x = torch.tensor(inp, dtype=torch.float)
    x = x.to(device)
    x = x.unsqueeze(0)
    x = x.permute(0, 3, 1, 2)
    criterion = torch.nn.MSELoss()
    # criterion = torch.nn.Sigmoid()
    #ssim = ssim()

    for iter in range(5000):
        camera.train()
        camera_optimizer.zero_grad()
        y, a, b,c = camera(x)
        # loss = criterion(x, y)
        s = ssim(x, y)
        loss = s
        print('Iter: {:.0f}\t Loss: {:.4f} SSIM: {:.4f}'.format(iter, loss.item(), s))
        loss.backward()
        camera_optimizer.step()

        camera.zernike_coeffs_train[1:].data.clamp_(-1, 1)

        if iter % 10 == 0:
            torch.save({'model': basic_model.state_dict()}, './damage5.pth')
            res = y.cpu().detach().permute(0, 2, 3, 1)[0]
            org = x.cpu().detach().permute(0, 2, 3, 1)[0]
            plt.subplot(121)
            plt.imshow(res / res.max())
            plt.subplot(122)
            plt.imshow(org / org.max())
            plt.show()

    torch.save({'model': basic_model.state_dict()}, './Last_Model.pth')
    res = y.cpu().detach().permute(0, 2, 3, 1)[0]
    org = x.cpu().detach().permute(0, 2, 3, 1)[0]
    plt.subplot(121)
    plt.imshow(res / res.max())
    plt.subplot(122)
    plt.imshow(org / org.max())
    plt.show()