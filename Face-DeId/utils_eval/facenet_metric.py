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
import dlib
from imutils import face_utils
import concurrent.futures
from facenet_pytorch import InceptionResnetV1
from torch.nn import functional as F
from tqdm import tqdm


def process_images(loader_src, camera, device, nets_ema, loader_ref, y, model):
    loss = []

    for i, img_source in enumerate(tqdm(loader_src, total=len(loader_src))):

        img_source = img_source.float().to(device)
        iter_ref = iter(loader_ref)
        x_ref = next(iter_ref).to(device)

        imgs_protected = utils.translate_using_reference_val(nets_ema, args, img_source, x_ref, y, camera=camera)
        imgs_protected = imgs_protected.clip(0, 1)

        for k, img_protected in enumerate(imgs_protected):
            embedding_org = model(img_source[k, None])
            embedding_priv = model(img_protected)
            aux = torch.cat([embedding_priv, embedding_org], 0)
            dists = [[(e1 - e2).norm().item() for e2 in aux] for e1 in aux]
            dists = dists[-1][:-1]
            loss.append(sum(dists) / len(dists))
    return loss


def process_images_Lab(loader_src, camera, device, nets_ema, loader_ref, y, model):
    loss = []

    for i, source in enumerate(tqdm(loader_src, total=len(loader_src))):

        img_source = source[0].float().to(device)
        img_priv = source[1].float().to(device)
        img_priv = F.interpolate(img_source, (16, 16))
        img_priv = F.interpolate(img_priv, (256, 256))

        iter_ref = iter(loader_ref)
        x_ref = next(iter_ref).to(device)

        imgs_protected = utils.translate_using_reference_lab(nets_ema, args, img_priv, x_ref, y)
        imgs_protected = imgs_protected.clip(0, 1)

        for k, img_protected in enumerate(imgs_protected):
            embedding_org = model(img_source[k, None])
            embedding_priv = model(img_protected)
            aux = torch.cat([embedding_priv, embedding_org], 0)
            dists = [[(e1 - e2).norm().item() for e2 in aux] for e1 in aux]
            dists = dists[-1][:-1]
            loss.append(sum(dists) / len(dists))
    return loss


def iteration(args):
    torch.manual_seed(777)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loader_src_male = get_eval_loader(root=args.val_img_dir + '/male/', img_size=args.img_size, batch_size=args.val_batch_size, imagenet_normalize=False,
                                      drop_last=True)
    loader_src_female = get_eval_loader(root=args.val_img_dir + '/female/', img_size=args.img_size, batch_size=args.val_batch_size, imagenet_normalize=False,
                                        drop_last=True)
    loader_ref_male = get_eval_loader(root='../data/celeba_hq/val' + '/male/', img_size=args.img_size, batch_size=args.num_ref, imagenet_normalize=False,
                                      drop_last=True)
    loader_ref_female = get_eval_loader(root='../data/celeba_hq/val' + '/female/', img_size=args.img_size, batch_size=args.num_ref, imagenet_normalize=False,
                                        drop_last=True)

    nets_ema, camera = load_models(args)
    camera = camera.to(device)

    resnet_vgg = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    resnet_casia = InceptionResnetV1(pretrained='casia-webface').eval().to(device)

    y_f = torch.zeros(args.num_ref, device=device, dtype=torch.long)
    y_m = torch.ones(args.num_ref, device=device, dtype=torch.long)

    # vgg_m2f = process_images_Lab(loader_src_male, camera, device, nets_ema, loader_ref_female, y_f, resnet_vgg)
    # vgg_f2m = process_images_Lab(loader_src_female, camera, device, nets_ema, loader_ref_male, y_m, resnet_vgg)
    vgg_m2f = process_images(loader_src_male, camera, device, nets_ema, loader_ref_female, y_f, resnet_vgg)
    vgg_f2m = process_images(loader_src_female, camera, device, nets_ema, loader_ref_male, y_m, resnet_vgg)

    print("\n\t-------------------")
    print('\t Ended VGG Iteration')
    print("\t-------------------")

    casia_m2f = process_images(loader_src_male, camera, device, nets_ema, loader_ref_female, y_f, resnet_casia)
    casia_f2m = process_images(loader_src_female, camera, device, nets_ema, loader_ref_male, y_m, resnet_casia)
    # casia_m2f = process_images_Lab(loader_src_male, camera, device, nets_ema, loader_ref_female, y_f, resnet_casia)
    # casia_f2m = process_images_Lab(loader_src_female, camera, device, nets_ema, loader_ref_male, y_m, resnet_casia)

    mean_vgg_m2f = sum(vgg_m2f) / len(vgg_m2f)
    mean_vgg_f2m = sum(vgg_f2m) / len(vgg_f2m)

    mean_casia_m2f = sum(casia_m2f) / len(casia_m2f)
    mean_casia_f2m = sum(casia_f2m) / len(casia_f2m)

    print("\n\t-------------------")
    print('\t Ended CASIA Iteration')
    print("\t-------------------")

    print('\n\n\t-----------------------------------------------------------------------------------------------------\n\n')
    print('The Final identity loss using VGG m2f is: {:.4f}'.format(mean_vgg_m2f))
    print('The Final identity loss using VGG f2f is: {:.4f}'.format(mean_vgg_f2m))

    print('The Final identity loss using CASIA m2f is: {:.4f}'.format(mean_casia_m2f))
    print('The Final identity loss using CASIA f2f is: {:.4f}'.format(mean_casia_f2m))

    return vgg_m2f, vgg_f2m, casia_m2f, casia_f2m


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
    nets_ema.generator.load_state_dict({'module.' + key: val for key, val in ckpt['generator'].items()})
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

    # directory for testing  #
    # parser.add_argument('--val_img_dir', type=str, default='/home/jhon/Desktop/Data_Sets/FACE2FACE_LAB/Frames_Crop/Org/val/')
    parser.add_argument('--val_img_dir', type=str, default='../data/celeba_hq/train/')
    parser.add_argument('--val_batch_size', type=int, default=8, help='Batch size for validation')
    parser.add_argument('--checkpoint_load_dir', type=str, default='../Final_expr/Model_X_ref_1/Checkpoints/110000_nets_ema.ckpt',
                        help='Directory to load network checkpoints')
    parser.add_argument('--wing_path', type=str, default='../checkpoints/wing.ckpt')
    parser.add_argument('--camera_ckpt', type=str, default='../checkpoints/Model_wing.pth')  # '../Final_expr/Model_Lab/Wing_Lab_lr.pth')
    # parser.add_argument('--camera_ckpt', type=str, default='../checkpoints/Wing_LR_16.pth')
    parser.add_argument('--num_ref', type=int, default=10, help='Number of domains')

    args = parser.parse_args()

    # model = UNet(3, 3)
    # train_unet(model, 1000, 5e-4, args)

    rstl = iteration(args)
    vgg_m2f = rstl[0]
    vgg_f2m = rstl[1]

    casia_m2f = rstl[2]
    casia_f2m = rstl[3]

    tensor1 = torch.tensor(vgg_m2f)
    tensor3 = torch.tensor(vgg_f2m)
    tensor5 = torch.tensor(casia_m2f)
    tensor7 = torch.tensor(casia_f2m)

    tensor_dict = {
        'vgg_m2f': tensor1,
        'vgg_f2m': tensor3,
        'casia_m2f': tensor5,
        'casia_f2m': tensor7,

    }
    # torch.save(tensor_dict, './Results/Identity_FFHQ_Celeb_LR_120.pt')
