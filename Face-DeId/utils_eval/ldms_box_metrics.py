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
from facenet_pytorch import MTCNN
from tqdm import tqdm


def process_images(loader_src, camera, device, nets_ema, loader_ref, y, dlib_model, mtcnn_model, pil_transform):
    landmarks_loss = []
    bounding_loss = []

    landmarks_loss_m = []
    bounding_loss_m = []

    for i, img_source in enumerate(tqdm(loader_src, total=len(loader_src))):

        iter_ref = iter(loader_ref)
        x_ref = next(iter_ref).to(device)

        img_source = img_source.float().to(device)
        imgs_protected = utils.translate_using_reference_val(nets_ema, args, img_source, x_ref, y, camera=camera)
        imgs_protected = imgs_protected.clip(0, 1)

        for k, img_protected in enumerate(imgs_protected):

            img = np.asarray(pil_transform(img_source[k]))
            shape_dlib, box_dlib = dlib_model.find_landmarks(img)
            shape_mtcnn, box_mtcnn = mtcnn_model.find_landmarks(img)
            # plt.imshow(img), plt.show()

            for x in range(img_protected.shape[0]):

                img_p = np.asarray(pil_transform(img_protected[x]))
                shape_p_dlib, box_p_dlib = dlib_model.find_landmarks(img_p)
                shape_p_mtcnn, box_p_mtcnn = mtcnn_model.find_landmarks(img_p)
                # plt.imshow(img_p), plt.show()

                if shape_dlib is not None and shape_p_dlib is not None:
                    # shape[:17, :] = 0
                    # shape_p[:17, :] = 0
                    # landmarks_loss.append(np.mean((shape_p - shape) ** 2))
                    # landmarks_loss.append(np.linalg.norm(shape_p - shape, axis=1).mean())
                    landmarks_loss.append(np.mean(np.abs(shape_dlib - shape_p_dlib)))

                    center_box = [(box_dlib[0] + box_dlib[2]) / 2, (box_dlib[1] + box_dlib[3]) / 2]
                    center_box_p = [(box_p_dlib[0] + box_p_dlib[2]) / 2, (box_p_dlib[1] + box_p_dlib[3]) / 2]
                    distance = np.sqrt((center_box[0] - center_box_p[0]) ** 2 + (center_box[1] - center_box_p[1]) ** 2)
                    bounding_loss.append(distance)

                if shape_mtcnn is not None and shape_p_mtcnn is not None:
                    # landmarks_loss_m.append(np.mean((shape_p - shape) ** 2))
                    # landmarks_loss_m.append(np.linalg.norm(shape_p - shape, axis=1).mean())
                    landmarks_loss_m.append(np.mean(np.abs(shape_mtcnn - shape_p_mtcnn)))

                    if box_mtcnn.shape[0] == 4 and box_p_mtcnn.shape[0] == 4:
                        center_box = [(box_mtcnn[0] + box_mtcnn[2]) / 2, (box_mtcnn[1] + box_mtcnn[3]) / 2]
                        center_box_p = [(box_p_mtcnn[0] + box_p_mtcnn[2]) / 2, (box_p_mtcnn[1] + box_p_mtcnn[3]) / 2]
                        distance = np.sqrt((center_box[0] - center_box_p[0]) ** 2 + (center_box[1] - center_box_p[1]) ** 2)
                        bounding_loss_m.append(distance)

    return landmarks_loss, bounding_loss, landmarks_loss_m, bounding_loss_m


def iteration(args):
    torch.manual_seed(777)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loader_src_male = get_eval_loader(root=args.val_img_dir + '/male/', img_size=args.img_size, batch_size=args.val_batch_size, imagenet_normalize=False,
                                      drop_last=True)
    loader_src_female = get_eval_loader(root=args.val_img_dir + '/female/', img_size=args.img_size, batch_size=args.val_batch_size, imagenet_normalize=False,
                                        drop_last=True)
    loader_ref_male = get_eval_loader(root=args.val_img_dir + '/male/', img_size=args.img_size, batch_size=args.num_ref, imagenet_normalize=False,
                                      drop_last=True)
    loader_ref_female = get_eval_loader(root=args.val_img_dir + '/female/', img_size=args.img_size, batch_size=args.num_ref, imagenet_normalize=False,
                                        drop_last=True)

    nets_ema, camera = load_models(args)
    camera = camera.to(device)

    pil_transform = transforms.ToPILImage()
    dlib_model = LandmarksDlib()
    mtcnn_model = LandmarksMTCNN()

    y_f = torch.zeros(args.num_ref, device=device, dtype=torch.long)
    y_m = torch.ones(args.num_ref, device=device, dtype=torch.long)

    landmarks_loss_male, bounding_loss_male, landmarks_loss_male_m, bounding_loss_male_m = process_images(loader_src_male, camera, device, nets_ema,
                                                                                                          loader_ref_female, y_f, dlib_model,
                                                                                                          mtcnn_model, pil_transform)

    landmarks_loss_female, bounding_loss_female, landmarks_loss_female_m, bounding_loss_female_m = process_images(loader_src_female, camera, device, nets_ema,
                                                                                                                  loader_ref_male, y_m,
                                                                                                                  dlib_model, mtcnn_model, pil_transform)

    landmarks_loss = landmarks_loss_male + landmarks_loss_female
    bounding_loss = bounding_loss_male + bounding_loss_female

    landmarks_loss_m = landmarks_loss_male_m + landmarks_loss_female_m
    bounding_loss_m = bounding_loss_male_m + bounding_loss_female_m

    print('The loss landmarks using DLIB is: {:.4f}'.format(sum(landmarks_loss) / len(landmarks_loss)))
    print('The loss Boundary using DLIB is: {:.4f}'.format(sum(bounding_loss) / len(bounding_loss)))
    print('The loss landmarks using MTCNN is: {:.4f}'.format(sum(landmarks_loss_m) / len(landmarks_loss_m)))
    print('The loss Boundary using MTCNN is: {:.4f}'.format(sum(bounding_loss_m) / len(bounding_loss_m)))

    return landmarks_loss, bounding_loss, landmarks_loss_m, bounding_loss_m


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


class LandmarksDlib:
    def __init__(self):
        # Initialize the DLIB shape predictor and face detector
        self.predictor = dlib.shape_predictor("../checkpoints/shape_predictor_68_face_landmarks.dat")
        self.detector = dlib.get_frontal_face_detector()

    def find_landmarks(self, image):
        # Detect faces in the image
        rects = self.detector(image)
        shape = None

        # loop over the face detections
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the landmark (x, y)-coordinates to a NumPy array
            x, y, width, height = (rect.left(), rect.top(), rect.width(), rect.height())
            shape = self.predictor(image, rect)
            shape = face_utils.shape_to_np(shape)

        if shape is None:
            return shape, []
        elif shape.shape[0] != 68:
            return shape, []
        else:
            return shape, [x, y, width, height]


class LandmarksMTCNN:
    def __init__(self):
        # Initialize the DLIB shape predictor and face detector
        self.mtcnn = MTCNN(keep_all=True, device='cuda:0')

    def find_landmarks(self, image):
        # Detect faces in the image
        boxes, probs, landmarks = self.mtcnn.detect(image, landmarks=True)
        if boxes is None:
            return None, None

        return np.round(landmarks[0].astype(float)).astype(int).squeeze(), np.round(boxes[0].astype(float)).astype(int).squeeze()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=256, help='Image resolution')
    parser.add_argument('--num_domains', type=int, default=2, help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16, help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=64, help='Style code dimension')
    parser.add_argument('--w_hpf', type=float, default=1.0, help='weight for high-pass filtering')

    # directory for testing
    parser.add_argument('--val_img_dir', type=str, default='../data/celeba_hq/val/', help='Directory containing validation images')
    parser.add_argument('--val_batch_size', type=int, default=8, help='Batch size for validation')
    parser.add_argument('--checkpoint_load_dir', type=str, default='../Final_expr/Model_X_ref_1/Checkpoints/110000_nets_ema.ckpt',
                        help='Directory to load network checkpoints')
    parser.add_argument('--wing_path', type=str, default='../checkpoints/wing.ckpt')
    # parser.add_argument('--camera_ckpt', type=str, default='../checkpoints/Wing_LR_16.pth')
    parser.add_argument('--camera_ckpt', type=str, default='../checkpoints/Model_wing.pth')
    parser.add_argument('--num_ref', type=int, default=10, help='Number of domains')

    args = parser.parse_args()
    collect_ldm = []
    collect_box = []
    collect_ldm_m = []
    collect_box_m = []
    collect_box_embd = []

    rstl = iteration(args)
    tensor1 = torch.tensor(rstl[0])
    tensor2 = torch.tensor(rstl[1])
    tensor3 = torch.tensor(rstl[2])
    tensor4 = torch.tensor(rstl[3])

    tensor_dict = {
        'Landmarks_dlib': tensor1,
        'Boxes_dlib': tensor2,
        'Landmarks_mtcnn': tensor3,
        'Boxes_mtcnn': tensor4,
    }
    # torch.save(tensor_dict, './Results/LDMs_BBX_FFHQ_Celeb_LR_120.pt')
