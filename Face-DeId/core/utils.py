import os
from os.path import join as ospj
import json
import glob
from shutil import copyfile
from matplotlib.colors import ListedColormap

from tqdm import tqdm
import ffmpeg

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torch.linalg as LA
import re

jet_cmap = plt.get_cmap('jet')


def save_json(json_file, filename):
    with open(filename, 'w') as f:
        json.dump(json_file, f, indent=4, sort_keys=False)


def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    # print(network)
    print("Number of parameters of %s: %i" % (name, num_params))


def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def save_image(x, ncol, filename):
    # x = denormalize(x)
    x = x.clamp_(0, 1)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)


def save_tensor_as_svg(tensor, filename, nrow=9):
    # Assuming tensor is a PyTorch tensor of size (81, 3, 256, 256)

    # Convert the tensor to a NumPy array and normalize the values to [0, 1]
    tensor = tensor.clamp(0, 1)  # Ensure values are between 0 and 1
    image_data = tensor.permute(0, 2, 3, 1).numpy()  # Convert to NumPy and rearrange channels

    # Create a new figure
    fig = plt.figure(figsize=(40, 40))  # Modify the figsize as needed

    # Calculate the number of rows and columns needed for the subplots
    batch_size = image_data.shape[0]
    ncol = int(batch_size / nrow)

    try:
        # Loop through the images and create subplots
        for i in range(batch_size):
            plt.subplot(ncol, nrow,  i + 1)
            plt.imshow(image_data[i])
            plt.axis('off')  # Hide axes ticks and labels
    except:
        ncol += 1
        # Loop through the images and create subplots
        for i in range(batch_size):
            plt.subplot(ncol, nrow, i + 1)
            plt.imshow(image_data[i])
            plt.axis('off')  # Hide axes ticks and labels

    # Adjust the layout for better spacing
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.0, hspace=0.0)

    # Save the figure as an SVG file
    plt.savefig(filename, format='svg', bbox_inches='tight', pad_inches=0)

    # Close the figure to free up resources
    plt.close(fig)


@torch.no_grad()
def translate_and_reconstruct(nets, args, x_src, y_src, x_ref, y_ref, filename):
    N, C, H, W = x_src.size()
    s_ref = nets.style_encoder(x_ref, y_ref)
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    x_fake = nets.generator(x_src, s_ref, masks=masks)
    s_src = nets.style_encoder(x_src, y_src)
    masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    x_rec = nets.generator(x_fake, s_src, masks=masks)
    x_concat = [x_src, x_ref, x_fake, x_rec]
    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename)
    del x_concat


@torch.no_grad()
def translate_using_latent(nets, args, x_src, y_trg_list, z_trg_list, psi, camera, filename):
    N, C, H, W = x_src.size()
    latent_dim = z_trg_list[0].size(1)
    x_concat = [x_src]
    if camera is not None:
        x_src = camera(x_src)
        # x_src = F.interpolate(x_src, (16, 16)).detach()
        # x_src = F.interpolate(x_src, (256, 256)).detach()
        masks = nets.fan_priv.get_heatmap(x_src, Privacy=True) if args.w_hpf > 0 else None

    else:
        masks = nets.fan.get_heatmap(x_src, Privacy=False) if args.w_hpf > 0 else None

    mask_rgb = jet_cmap(masks[0].cpu().squeeze(1).data.numpy())
    mask_rgb = torch.from_numpy(mask_rgb[:, :, :, :3]).permute(0, 3, 1, 2).to(x_src.device)

    x_concat += [x_src]
    x_concat += [mask_rgb]

    for i, y_trg in enumerate(y_trg_list):
        z_many = torch.randn(10000, latent_dim).to(x_src.device)
        y_many = torch.LongTensor(10000).to(x_src.device).fill_(y_trg[0])
        s_many = nets.mapping_network(z_many, y_many)
        s_avg = torch.mean(s_many, dim=0, keepdim=True)
        s_avg = s_avg.repeat(N, 1)

        for z_trg in z_trg_list:
            s_trg = nets.mapping_network(z_trg, y_trg)
            s_trg = torch.lerp(s_avg, s_trg, psi)
            x_fake = nets.generator(x_src, s_trg, masks=masks)
            x_concat += [x_fake]

    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename)
    save_tensor_as_svg(x_concat.cpu().data, filename.replace('jpg', 'svg'), nrow=N)


@torch.no_grad()
def translate_using_reference(nets, args, x_src, x_ref, y_ref, filename, camera=None):
    N, C, H, W = x_src.size()
    wb = torch.ones(1, C, H, W).to(x_src.device)
    x_org = x_src.clone()
    # y_ref = torch.randint_like(y_ref, 0, 2)
    if camera is not None:
        x_src = camera(x_src)
        # x_src = F.interpolate(x_src, (16, 16)).detach() # Uncomment if you want to try the low-resolution model
        # x_src = F.interpolate(x_src, (256, 256)).detach()
        masks = nets.fan_priv.get_heatmap(x_src, Privacy=True) if args.w_hpf > 0 else None

    else:
        masks = nets.fan.get_heatmap(x_src, Privacy=False) if args.w_hpf > 0 else None

    mask_rgb = jet_cmap(masks[0].cpu().squeeze(1).data.numpy())
    mask_rgb = torch.from_numpy(mask_rgb[:, :, :, :3]).permute(0, 3, 1, 2).to(x_src.device)

    x_src_with_wb = torch.cat([wb, x_src], dim=0)
    x_org_with_wb = torch.cat([wb, x_org], dim=0)
    mask_with_wb = torch.cat([wb, mask_rgb], dim=0)

    s_ref = nets.style_encoder(x_ref, y_ref)
    s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1)
    x_concat = [x_org_with_wb]
    x_concat += [mask_with_wb]
    x_concat += [x_src_with_wb]
    pattern = r'Final_expr/WS_Model_X_ref_1/Results_Video/Ref_/'
    # save_image(x_src, N + 1, filename)
    for i, s_ref in enumerate(s_ref_list):
        x_fake = nets.generator(x_src, s_ref, masks=masks)
        # save_image(x_fake, N + 1, filename)
        x_fake_with_ref = torch.cat([x_ref[i:i + 1], x_fake], dim=0)
        x_concat += [x_fake_with_ref]
        folder_name = pattern.replace('Ref_', 'Ref_'+str(i + 1))
        os.makedirs(folder_name, exist_ok=True)
        [save_image(x_fake[k], 1, folder_name + '/frame_{:04d}.png'.format((8*filename) + k)) for k in range(N)]

    folder_name = pattern.replace('Ref_', 'Priv')
    os.makedirs(folder_name, exist_ok=True)
    [save_image(x_src[k], 1, folder_name + '/frame_{:.0f}.png'.format((8 * filename) + k)) for k in range(N)]

    # x_concat = torch.cat(x_concat, dim=0)
    # save_image(x_concat, N + 1, filename)
    # save_tensor_as_svg(x_concat.cpu().data, filename.replace('png', 'svg'), nrow=N + 1) # +3
    del x_concat

@torch.no_grad()
def translate_using_reference_val(nets, args, x_src, x_ref, y_ref, camera=None):
    N, C, H, W = x_src.size()
    if camera is not None:
        x_src = camera(x_src)
        # x_src = F.interpolate(x_src, (16, 16)).detach()
        # x_src = F.interpolate(x_src, (256, 256)).detach()
        masks = nets.fan_priv.get_heatmap(x_src, Privacy=True) if args.w_hpf > 0 else None

    else:
        masks = nets.fan.get_heatmap(x_src, Privacy=False) if args.w_hpf > 0 else None

    s_ref = nets.style_encoder(x_ref, y_ref)
    s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1)
    x_concat = []

    for i, s_ref in enumerate(s_ref_list):
        x_fake = nets.generator(x_src, s_ref, masks=masks)
        x_concat += [x_fake]

    # x_concat = torch.cat(x_concat, dim=0)
    x_concat = torch.stack(x_concat, dim=1)
    return x_concat

@torch.no_grad()
def translate_using_reference_lab(nets, args, x_src, x_ref, y_ref):
    N, C, H, W = x_src.size()
    masks = nets.fan_priv.get_heatmap(x_src, Privacy=True) if args.w_hpf > 0 else None

    s_ref = nets.style_encoder(x_ref, y_ref)
    s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1)
    x_concat = []

    for i, s_ref in enumerate(s_ref_list):
        x_fake = nets.generator(x_src, s_ref, masks=masks)
        x_concat += [x_fake]

    x_concat = torch.stack(x_concat, dim=1)
    return x_concat


@torch.no_grad()
def debug_image(nets, args, inputs, camera,  step):
    x_src, y_src = inputs.x, inputs.y

    device = x_src.device
    N = x_src.size(0)

    # latent-guided image synthesis
    y_trg_list = [torch.tensor(y).repeat(N).to(device) for y in range(min(args.num_domains, 5))]
    z_trg_list = torch.randn(args.num_outs_per_domain, 1, args.latent_dim).repeat(1, N, 1).to(device)
    for psi in [0.5, 0.7, 1.0]:
        filename = ospj(args.result_dir, '%06d_latent_psi_%.1f.jpg' % (step, psi))
        translate_using_latent(nets, args, x_src, y_trg_list, z_trg_list, psi, camera, filename)


# ======================= #
# Video-related functions #
# ======================= #


def sigmoid(x, w=1):
    return 1. / (1 + np.exp(-w * x))


def get_alphas(start=-5, end=5, step=0.5, len_tail=10):
    return [0] + [sigmoid(alpha) for alpha in np.arange(start, end, step)] + [1] * len_tail


def interpolate(nets, args, x_src, s_prev, s_next, masks):
    ''' returns T x C x H x W '''
    B = x_src.size(0)
    frames = []
    # masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    alphas = get_alphas()

    for alpha in alphas:
        s_ref = torch.lerp(s_prev, s_next, alpha)
        x_fake = nets.generator(x_src, s_ref, masks=masks)
        x_fake = (x_fake - x_fake.min()) / (x_fake.max() - x_fake.min())
        # print(x_fake.min(), x_fake.max())
        entries = torch.cat([x_src.cpu(), x_fake.cpu()], dim=2)
        frame = torchvision.utils.make_grid(entries, nrow=B, padding=0, pad_value=-1).unsqueeze(0)
        frames.append(frame)
    frames = torch.cat(frames)
    return frames


def slide(entries, margin=32):
    """Returns a sliding reference window.
    Args:
        entries: a list containing two reference images, x_prev and x_next, 
                 both of which has a shape (1, 3, 256, 256)
    Returns:
        canvas: output slide of shape (num_frames, 3, 256*2, 256+margin)
    """
    _, C, H, W = entries[0].shape
    alphas = get_alphas()
    T = len(alphas)  # number of frames

    canvas = - torch.ones((T, C, H * 2, W + margin))
    merged = torch.cat(entries, dim=2)  # (1, 3, 512, 256)
    for t, alpha in enumerate(alphas):
        top = int(H * (1 - alpha))  # top, bottom for canvas
        bottom = H * 2
        m_top = 0  # top, bottom for merged
        m_bottom = 2 * H - top
        canvas[t, :, top:bottom, :W] = merged[:, :, m_top:m_bottom, :]
    return canvas


@torch.no_grad()
def video_ref(nets, args, x_src, x_ref, y_ref, fname, camera=None):
    video = []
    if camera is None:
        masks = nets.fan.get_heatmap(x_src, Privacy=False) if args.w_hpf > 0 else None
    else:
        # x_ref = camera(x_ref)
        x_src = camera(x_src)
        masks = nets.fan_priv.get_heatmap(x_src, Privacy=True) if args.w_hpf > 0 else None
    s_ref = nets.style_encoder(x_ref, y_ref)
    s_prev = None
    for data_next in tqdm(zip(x_ref, y_ref, s_ref), 'video_ref', len(x_ref)):
        x_next, y_next, s_next = [d.unsqueeze(0) for d in data_next]
        if s_prev is None:
            x_prev, y_prev, s_prev = x_next, y_next, s_next
            continue
        if y_prev != y_next:
            x_prev, y_prev, s_prev = x_next, y_next, s_next
            continue

        interpolated = interpolate(nets, args, x_src, s_prev, s_next, masks)
        entries = [x_prev, x_next]
        slided = slide(entries)  # (T, C, 256*2, 256)
        frames = torch.cat([slided, interpolated], dim=3).cpu()  # (T, C, 256*2, 256*(batch+1))
        video.append(frames)
        x_prev, y_prev, s_prev = x_next, y_next, s_next

    # append last frame 10 time
    for _ in range(10):
        video.append(frames[-1:])
    video = tensor2ndarray255(torch.cat(video))
    save_video(fname, video)


@torch.no_grad()
def video_latent(nets, args, x_src, y_list, z_list, psi, fname):
    latent_dim = z_list[0].size(1)
    s_list = []
    for i, y_trg in enumerate(y_list):
        z_many = torch.randn(10000, latent_dim).to(x_src.device)
        y_many = torch.LongTensor(10000).to(x_src.device).fill_(y_trg[0])
        s_many = nets.mapping_network(z_many, y_many)
        s_avg = torch.mean(s_many, dim=0, keepdim=True)
        s_avg = s_avg.repeat(x_src.size(0), 1)

        for z_trg in z_list:
            s_trg = nets.mapping_network(z_trg, y_trg)
            s_trg = torch.lerp(s_avg, s_trg, psi)
            s_list.append(s_trg)

    s_prev = None
    video = []
    # fetch reference images
    for idx_ref, s_next in enumerate(tqdm(s_list, 'video_latent', len(s_list))):
        if s_prev is None:
            s_prev = s_next
            continue
        if idx_ref % len(z_list) == 0:
            s_prev = s_next
            continue
        frames = interpolate(nets, args, x_src, s_prev, s_next).cpu()
        video.append(frames)
        s_prev = s_next
    for _ in range(10):
        video.append(frames[-1:])
    video = tensor2ndarray255(torch.cat(video))
    save_video(fname, video)


def save_video(fname, images, output_fps=30, vcodec='libx264', filters=''):
    assert isinstance(images, np.ndarray), "images should be np.array: NHWC"
    num_frames, height, width, channels = images.shape
    stream = ffmpeg.input('pipe:', format='rawvideo',
                          pix_fmt='rgb24', s='{}x{}'.format(width, height))
    stream = ffmpeg.filter(stream, 'setpts', '2*PTS')  # 2*PTS is for slower playback
    stream = ffmpeg.output(stream, fname, pix_fmt='yuv420p', vcodec=vcodec, r=output_fps)
    stream = ffmpeg.overwrite_output(stream)
    process = ffmpeg.run_async(stream, pipe_stdin=True)
    for frame in tqdm(images, desc='writing video to %s' % fname):
        process.stdin.write(frame.astype(np.uint8).tobytes())
    process.stdin.close()
    process.wait()


def tensor2ndarray255(images):
    # images = torch.clamp(images * 0.5 + 0.5, 0, 1)
    return images.cpu().numpy().transpose(0, 2, 3, 1) * 255


@torch.no_grad()
def save_video_from_images(root_path, name, fps=24):
    import cv2
    import os

    files = sorted(os.listdir(root_path))

    # Leer una imagen para obtener sus dimensiones
    primer_imagen = cv2.imread(os.path.join(root_path, files[0]))
    alto, ancho, _ = primer_imagen.shape

    # Crear el objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec de video
    video_salida = cv2.VideoWriter(name, fourcc, fps, (ancho, alto))

    # Recorrer los archivos de la carpeta y agregarlos al video
    for archivo in files:
        if archivo.endswith('.jpg') or archivo.endswith('.png'):
            ruta_imagen = os.path.join(root_path, archivo)
            imagen = cv2.imread(ruta_imagen)
            if imagen is not None:
                video_salida.write(imagen)

    # Liberar recursos
    video_salida.release()
    cv2.destroyAllWindows()

    print(f"Video saved as: {name}")


def dice_coefficient_batch(predictions, targets):
    smooth = 1.0
    intersections = torch.sum(predictions * targets, dim=(2, 3))
    unions = torch.sum(predictions, dim=(2, 3)) + torch.sum(targets, dim=(2, 3))
    dice_scores = (2.0 * intersections + smooth) / (unions + smooth)
    loss = 1.0 - dice_scores.mean()  # Invert the Dice coefficient to use as a loss
    return loss


class loss_RAFT:
    def __init__(self, DEVICE):
        # super().__init__()
        from RAFT.core.raft import RAFT
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', help="restore checkpoint", default='RAFT/models/raft-things.pth')
        parser.add_argument('--small', action='store_true', help='use small model', default=False)
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision', default=False)
        parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation', default=False)
        args = parser.parse_args()
        args.model = '../RAFT/models/raft-things.pth'

        model = RAFT(args)
        ckpt = torch.load(args.model)
        ckpt_1 = {key.replace('module.', ''): val for key, val in ckpt.items() if 'module.' in key}
        model.load_state_dict(ckpt_1, strict=True)

        self.model = model
        self.model.to(DEVICE)
        self.model.eval()
        del ckpt, ckpt_1

    def __call__(self, frame1, frame2):
        rstl = [self.model(frame1[x, None, ...], frame2[x, None, ...], iters=20, test_mode=True).mean().abs() for x in range(frame1.shape[0])]
        return sum(rstl)
