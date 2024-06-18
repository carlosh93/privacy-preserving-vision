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
from tqdm import tqdm
import face_recognition
from torch.nn.functional import interpolate


def process_images(loader_src, camera, device, nets_ema, loader_ref_male, loader_ref_female, y, pil_transform):
    embeddind = []

    for i, img_source in enumerate(tqdm(loader_src, total=len(loader_src))):

        img_source = img_source.float().to(device)

        iter_ref = iter(loader_ref_male)
        x_ref_male = next(iter_ref).to(device)
        iter_ref_1 = iter(loader_ref_female)
        x_ref_female = next(iter_ref_1).to(device)
        bacth_ref = torch.cat([x_ref_male, x_ref_female], dim=0)

        imgs_protected = utils.translate_using_reference_val(nets_ema, args, img_source, bacth_ref, y, camera=camera)
        imgs_protected = imgs_protected.clip(0, 1)

        for k, img_protected in enumerate(imgs_protected):
            img = np.asarray(pil_transform(img_source[k]))
            # plt.imshow(img), plt.show()
            e1 = face_recognition.face_encodings(img)
            for x in range(img_protected.shape[0]):
                img_p = np.asarray(pil_transform(img_protected[x]))
                # plt.imshow(img_p), plt.show()
                e2 = face_recognition.face_encodings(img_p)
                if len(e1) > 0 and len(e2) > 0:
                    embeddind.append(np.linalg.norm(e1[0] - e2[0]))

    return embeddind


def process_images_lab(loader_src, camera, device, nets_ema, loader_ref_male, loader_ref_female, y, pil_transform):
    embeddind = []

    for i, source in enumerate(tqdm(loader_src, total=len(loader_src))):

        img_source = source[0].float().to(device)
        img_priv = source[1].float().to(device)
        img_priv = interpolate(img_source, (16, 16))
        img_priv = interpolate(img_priv, (256, 256))

        iter_ref = iter(loader_ref_male)
        x_ref_male = next(iter_ref).to(device)
        iter_ref_1 = iter(loader_ref_female)
        x_ref_female = next(iter_ref_1).to(device)
        bacth_ref = torch.cat([x_ref_male, x_ref_female], dim=0)

        imgs_protected = utils.translate_using_reference_lab(nets_ema, args, img_priv, bacth_ref, y)
        imgs_protected = imgs_protected.clip(0, 1)

        for k, img_protected in enumerate(imgs_protected):
            img = np.asarray(pil_transform(img_source[k]))
            # plt.imshow(img), plt.show()
            e1 = face_recognition.face_encodings(img)
            for x in range(img_protected.shape[0]):
                img_p = np.asarray(pil_transform(img_protected[x]))
                # plt.imshow(img_p), plt.show()
                e2 = face_recognition.face_encodings(img_p)
                if len(e1) > 0 and len(e2) > 0:
                    embeddind.append(np.linalg.norm(e1[0] - e2[0]))

    return embeddind


def iteration(args):
    torch.manual_seed(777)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bacth_ref = load_ref(args).to(device)
    y_1 = torch.zeros(args.num_ref, device=device, dtype=torch.long)
    y_2 = torch.ones(args.num_ref, device=device, dtype=torch.long)
    y = torch.cat([y_1, y_2], dim=0)

    loader_src_male = get_eval_loader(root=args.val_img_dir + '/male/', img_size=args.img_size, batch_size=args.val_batch_size, imagenet_normalize=False)
    loader_src_female = get_eval_loader(root=args.val_img_dir + '/female/', img_size=args.img_size, batch_size=args.val_batch_size, imagenet_normalize=False)
    loader_ref_male = get_eval_loader(root='../data/celeba_hq/val' + '/male/', img_size=args.img_size, batch_size=args.num_ref, imagenet_normalize=False, shuffle=True)
    loader_ref_female = get_eval_loader(root='../data/celeba_hq/val' + '/female/', img_size=args.img_size, batch_size=args.num_ref, imagenet_normalize=False,
                                        shuffle=True)

    nets_ema, camera = load_models(args)
    camera = camera.to(device)

    pil_transform = transforms.ToPILImage()
    """with_concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_male = executor.submit(process_images, loader_src_male, camera, device, nets_ema, bacth_ref, y, pil_transform)
        future_female = executor.submit(process_images, loader_src_female, camera, device, nets_ema, bacth_ref, y, pil_transform)

        embeddings_male = future_male.result()
        embeddings_female = future_female.result()
    embeddings = embeddings_female + embeddings_male """

    embeddings_female = process_images(loader_src_male, camera, device, nets_ema, loader_ref_male, loader_ref_female, y, pil_transform)
    embeddings_male = process_images(loader_src_female, camera, device, nets_ema, loader_ref_male, loader_ref_female, y, pil_transform)
    # embeddings_female = process_images_lab(loader_src_male, camera, device, nets_ema, loader_ref_male, loader_ref_female, y, pil_transform)
    # embeddings_male = process_images_lab(loader_src_female, camera, device, nets_ema, loader_ref_male, loader_ref_female, y, pil_transform)
    embeddings = embeddings_female + embeddings_male

    print('The embedding distance is: {:.4f}'.format(sum(embeddings) / len(embeddings)))

    return embeddings


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
    # camera.load_state_dict(checkpoint_Cam_Wing['Camera'], strict=True)
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
    # parser.add_argument('--val_img_dir', type=str, default='/home/jhon/Desktop/Data_Sets/FACE2FACE_LAB/Frames_Crop/Org/val/', help='Directory containing validation images')
    parser.add_argument('--val_img_dir', type=str, default='../data/celeba_hq/train/', help='Directory containing validation images')
    parser.add_argument('--val_batch_size', type=int, default=8, help='Batch size for validation')
    parser.add_argument('--checkpoint_load_dir', type=str, default='../Final_expr/Ablations/FFHQ_Celeb_lr/120000_nets_ema.ckpt',
                        help='Directory to load network checkpoints')
    parser.add_argument('--wing_path', type=str, default='../checkpoints/wing.ckpt')
    # parser.add_argument('--camera_ckpt', type=str, default='../checkpoints/Model_wing.pth')  # '../Final_expr/Model_Lab/Wing_Lab_lr.pth')
    parser.add_argument('--camera_ckpt', type=str, default='../checkpoints/Wing_LR_16.pth')
    parser.add_argument('--num_ref', type=int, default=5, help='Number of domains')

    args = parser.parse_args()
    rstl = iteration(args)

    print('\n\n\t-----------------------------------------------------------------------------------------------------\n\n')
    print('The Maximum Final embedding distance using Face recognition is: {:.4f}'.format(max(rstl)))
    print('The Minimum Final embedding distance using Face recognition is: {:.4f}'.format(min(rstl)))

    tensor1 = torch.tensor(rstl)
    tensor_dict = {
        'Face_recognition': tensor1,
    }
    torch.save(tensor_dict, 'Results/FR_FFHQ_Celeb_LR_120.pt')
