import torch
import matplotlib.pyplot as plt
import poppy
from torch.fft import rfftn, irfftn


def conv2D(img, kernel):
    img_fft = rfftn(img, dim=(-2, -1))
    kernel_fft = rfftn(kernel, dim=(-2, -1))
    conv = torch.multiply(img_fft, kernel_fft)
    img_conv = irfftn(conv, dim=(-2, -1))
    return img_conv


def ifftshift(x, dim=None):
    if dim is None:
        dim = tuple(range(x.dim()))
    elif not isinstance(dim, tuple):
        dim = (dim,)
    shift = tuple(x.size(d) // 2 for d in dim)
    return torch.roll(x, shift, dim)


def fftshift(x, dim=None):
    if dim is None:
        dim = tuple(range(x.dim()))
    elif not isinstance(dim, tuple):
        dim = (dim,)
    shift = tuple(-(x.size(d) // 2) for d in dim)
    return torch.roll(x, shift, dim)


def deta(Lb):
    IdLens = torch.sqrt(1 + (0.6961663 * (Lb ** 2) / ((Lb ** 2) - 0.0684043 ** 2) + 0.4079426 * (Lb ** 2) / (
            (Lb ** 2) - 0.1162414 ** 2) + 0.8974794 * (Lb ** 2) / ((Lb ** 2) - 9.896161 ** 2)))

    # IdLens = 1.5375 + 0.00829045 * (Lb ** -2) - 0.000211046 * (Lb ** -4)
    IdAir = 1 + 0.05792105 / (238.0185 - Lb ** -2) + 0.00167917 / (57.362 - Lb ** -2)
    val = torch.abs(IdLens - IdAir)
    return val


def cart2pol(x, y):
    rho = torch.sqrt(x ** 2 + y ** 2)
    phi = torch.atan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    x = rho * torch.cos(phi)
    y = rho * torch.sin(phi)
    return x, y


def compl_exp(phase):
    """Complex exponent via Euler's formula since CUDA doesn't have a GPU kernel for that."""
    return torch.complex(torch.cos(phase), torch.sin(phase))


def get_zernike_volume(resolution, n_terms, scale_factor=1e-6, height_tolerance=2e-8):
    zernike_volume = poppy.zernike.zernike_basis(nterms=n_terms, npix=resolution, outside=0.0)
    # zernike_volume *= np.random.uniform(low=-height_tolerance, high=height_tolerance, size=(zernike_volume.shape[1], zernike_volume.shape[2]))
    return zernike_volume * scale_factor


def plot_PSFs(psfs):
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.flatten()
    for i, image in enumerate(psfs):
        axes[i].imshow((image.permute(1, 2, 0) / image.max()).data.numpy())
        axes[i].axis('off')

    # Adjust the spacing between subplots
    plt.tight_layout()
    plt.show()
