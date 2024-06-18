import numpy as np
import torch
from core.data_loader import get_eval_loader
import argparse
import os
import random
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from core.model import build_model
from Camera.Optics import Camera
from core import utils
from metrics.fid import calculate_fid_given_paths
from torch.nn import functional as F
import shutil
from tqdm import tqdm


def process_images(loader_src, camera, device, nets_ema, loader_ref, y, args, task):
    path_fake = os.path.join('../data/FID/', task)
    shutil.rmtree(path_fake, ignore_errors=True)
    os.makedirs(path_fake)

    for i, img_source in enumerate(tqdm(loader_src, total=len(loader_src))):

        iter_ref = iter(loader_ref)
        x_ref = next(iter_ref).float().to(device)

        img_source = img_source.float().to(device)

        imgs_protected = utils.translate_using_reference_val(nets_ema, args, img_source, x_ref, y, camera=camera)

        for k, img_protected in enumerate(imgs_protected):
            # save generated images to calculate FID later
            for j in range(args.num_ref):
                filename = os.path.join(path_fake, '%.4i_%.2i.png' % (i * 8 + (k + 1), j + 1))
                utils.save_image(img_protected[j], ncol=1, filename=filename)


def iteration(args):
    torch.manual_seed(777)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loader_src_male = get_eval_loader(root=args.val_img_dir + '/male/', img_size=args.img_size, batch_size=args.val_batch_size, imagenet_normalize=False, drop_last=True)
    loader_src_female = get_eval_loader(root=args.val_img_dir + '/female/', img_size=args.img_size, batch_size=args.val_batch_size, imagenet_normalize=False, drop_last=True)
    loader_ref_male = get_eval_loader(root=args.val_img_dir + '/male/', img_size=args.img_size, batch_size=args.num_ref, imagenet_normalize=False, drop_last=True)
    loader_ref_female = get_eval_loader(root=args.val_img_dir + '/female/', img_size=args.img_size, batch_size=args.num_ref, imagenet_normalize=False, drop_last=True)

    nets_ema, camera = load_models(args)
    camera = camera.to(device)

    y_f = torch.zeros(args.num_ref, device=device, dtype=torch.long)
    y_m = torch.ones(args.num_ref, device=device, dtype=torch.long)

    process_images(loader_src_male, camera, device, nets_ema, loader_ref_female, y_f, args, 'male2female')
    process_images(loader_src_female, camera, device, nets_ema, loader_ref_male, y_m, args, 'female2male')

    fid_male2female = calculate_fid_given_paths(paths=['../data/celeba_hq/train/female/', '../data/FID/male2female'], img_size=args.img_size, batch_size=args.val_batch_size)
    fid_female2male = calculate_fid_given_paths(paths=['../data/celeba_hq/train/male/', '../data/FID/female2male'], img_size=args.img_size, batch_size=args.val_batch_size)

    print("\n\t-------------------")
    print('\t FID male2female is: {:.4f}'.format(fid_male2female))
    print('\t FID female2male: {:.4f}'.format(fid_female2male))
    print("\t-------------------")

    return fid_male2female, fid_female2male


def load_ref(args):
    transform = transforms.ToTensor()
    files = os.listdir(args.val_img_dir + '/male/')
    random.shuffle(files)
    male_files = files[:args.num_ref]
    male_files = [args.val_img_dir + '/male/' + x for x in male_files]

    files = os.listdir(args.val_img_dir + '/female/')
    random.shuffle(files)
    female_files = files[:args.num_ref]
    female_files = [args.val_img_dir + '/female/' + x for x in female_files]

    male_imgs = [transform(Image.open(x).resize((args.img_size, args.img_size))) for x in male_files]
    female_imgs = [transform(Image.open(x).resize((args.img_size, args.img_size))) for x in female_files]

    bacth_ref = torch.stack(male_imgs + female_imgs, dim=0)
    return bacth_ref


def load_models(args):
    _, nets_ema = build_model(args)
    ckpt = torch.load(args.checkpoint_load_dir)
    nets_ema.fan.load_state_dict({'module.' + key: val for key, val in ckpt['fan'].items()})
    nets_ema.generator.load_state_dict({'module.' + key: val for key, val in ckpt['generator'].items()}, strict=False)
    nets_ema.mapping_network.load_state_dict({'module.' + key: val for key, val in ckpt['mapping_network'].items()})
    nets_ema.style_encoder.load_state_dict({'module.' + key: val for key, val in ckpt['style_encoder'].items()})
    nets_ema.fan_priv.load_state_dict({'module.' + key: val for key, val in ckpt['fan_priv'].items()})

    camera = Camera(device='cuda', zernike_terms=300)
    checkpoint_Cam_Wing = torch.load(args.camera_ckpt, map_location='cuda')
    camera.load_state_dict(checkpoint_Cam_Wing['Camera'], strict=True)
    nets_ema.fan_priv.load_state_dict(checkpoint_Cam_Wing['Decoder'], strict=True)
    return nets_ema, camera


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=256, help='Image resolution')
    parser.add_argument('--num_domains', type=int, default=2, help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16, help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=64, help='Style code dimension')
    parser.add_argument('--w_hpf', type=float, default=1.0, help='weight for high-pass filtering')

    # directory for testing
    parser.add_argument('--val_img_dir', type=str, default='../data/celeba_hq/train/', help='Directory containing validation images')
    parser.add_argument('--val_batch_size', type=int, default=8, help='Batch size for validation')
    parser.add_argument('--checkpoint_load_dir', type=str, default='../Final_expr/Model_FFHQ/Checkpoints/150000_nets_ema.ckpt',
                        help='Directory to load network checkpoints')
    parser.add_argument('--wing_path', type=str, default='../checkpoints/wing.ckpt')
    parser.add_argument('--camera_ckpt', type=str, default='../checkpoints/Model_wing.pth')
    # parser.add_argument('--camera_ckpt', type=str, default='../checkpoints/Wing_LR_16.pth')
    parser.add_argument('--num_ref', type=int, default=10, help='Number of domains')

    args = parser.parse_args()

    vgg_m2m = []
    vgg_f2f = []

    rstl = iteration(args)
    vgg_f2f.append(rstl[1])
    vgg_m2m.append(rstl[0])

    fids = vgg_m2m + vgg_f2f

    print('\n\n\t-----------------------------------------------------------------------------------------------------\n\n')
    print('The Final FID mean is: {:.4f}'.format(sum(fids) / len(fids)))

    tensor2 = torch.tensor(vgg_m2m)
    tensor4 = torch.tensor(vgg_f2f)

    tensor_dict = {
        'vgg_m2m': tensor2,
        'vgg_f2f': tensor4,
    }
    # torch.save(tensor_dict, './Results/FID_FFHQ_Celeb_lr_120.pt')
