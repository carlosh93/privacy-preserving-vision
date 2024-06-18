import os
import argparse

from munch import Munch
from torch.backends import cudnn
import torch

from core.data_loader import get_train_loader
from core.data_loader import get_test_loader
from core.solver import Solver


# from core.solver_org import Solver


def str2bool(v):
    return v.lower() in 'true'


def subdirs(dname):
    return [d for d in os.listdir(dname)
            if os.path.isdir(os.path.join(dname, d))]


def main(args):
    print(args)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    # Check if the folder exist
    if os.path.exists(args.checkpoint_save_dir):
        # List files folder
        files_in_folder = os.listdir(args.checkpoint_save_dir)
        for archivo in files_in_folder:
            if int(archivo[:6]) > args.resume_iter:
                args.resume_iter = int(archivo[:6])
                args.checkpoint_dir = args.checkpoint_save_dir

        print("\n\nThere is a checkpoint for this run from iter {:.0f}".format(args.resume_iter))
    else:
        print("\n\nNo checkpoint found in {}....  Start training from scratch.".format(args.checkpoint_save_dir))

    if args.use_wandb:
        try:
            import wandb
            args_wandb = vars(args)
            wandb.init(name=args_wandb['wandb_name'], dir=args_wandb['wandb_dir'], project=args_wandb['wandb_project'], config=args_wandb,
                       resume=args_wandb['resume_wandb'])

            main_script_path = "./core/solver.py"
            # Create an artifact
            artifact = wandb.Artifact("main_solver", type="code")
            # Add the main.py file to the artifact
            artifact.add_file(main_script_path)

            # Log the artifact
            wandb.log_artifact(artifact)

        except ImportError:
            print("WandB is not installed. Please install it using 'pip install wandb'.")
            raise SystemExit(1)

    solver = Solver(args)

    if args.mode == 'train':
        assert len(subdirs(args.train_img_dir)) == args.num_domains
        assert len(subdirs(args.val_img_dir)) == args.num_domains
        loaders = Munch(src=get_train_loader(root=args.train_img_dir,
                                             which='source',
                                             img_size=args.img_size,
                                             batch_size=args.batch_size,
                                             prob=args.randcrop_prob,
                                             num_workers=args.num_workers),
                        ref=get_train_loader(root=args.train_img_dir,
                                             which='reference',
                                             img_size=args.img_size,
                                             batch_size=args.batch_size,
                                             prob=args.randcrop_prob,
                                             num_workers=args.num_workers),
                        val=get_test_loader(root=args.val_img_dir,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers))
        solver.train(loaders)

    elif args.mode == 'finetune':
        assert len(subdirs(args.train_img_dir)) == args.num_domains
        assert len(subdirs(args.val_img_dir)) == args.num_domains
        loaders = Munch(src=get_train_loader(root=args.train_img_dir,
                                             which='source',
                                             img_size=args.img_size,
                                             batch_size=args.batch_size,
                                             prob=args.randcrop_prob,
                                             num_workers=args.num_workers),
                        ref=get_train_loader(root=args.train_img_dir,
                                             which='reference',
                                             img_size=args.img_size,
                                             batch_size=args.batch_size,
                                             prob=args.randcrop_prob,
                                             num_workers=args.num_workers),
                        val=get_test_loader(root=args.val_img_dir,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers))
        solver.finetune_lab(loaders)

    elif args.mode == 'sample':
        assert len(subdirs(args.src_dir)) == args.num_domains
        assert len(subdirs(args.ref_dir)) == args.num_domains
        loaders = Munch(src=get_test_loader(root=args.src_dir,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers),
                        ref=get_test_loader(root=args.ref_dir,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers))
        solver.sample(loaders)
    elif args.mode == 'eval':
        solver.evaluate()
    elif args.mode == 'align':
        from core.wing import align_faces
        align_faces(args, args.inp_dir, args.out_dir)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image resolution')
    parser.add_argument('--num_domains', type=int, default=2,
                        help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=64,
                        help='Style code dimension')

    # weight for objective functions
    parser.add_argument('--lambda_reg', type=float, default=1,
                        help='Weight for R1 regularization')
    parser.add_argument('--lambda_cyc', type=float, default=7,  # value of 7 for Privacy consistency
                        help='Weight for cyclic consistency loss')
    parser.add_argument('--lambda_sty', type=float, default=1,
                        help='Weight for style reconstruction loss')
    parser.add_argument('--lambda_ds', type=float, default=1,
                        help='Weight for diversity sensitive loss')
    parser.add_argument('--ds_iter', type=int, default=100000,
                        help='Number of iterations to optimize diversity sensitive loss')
    parser.add_argument('--w_hpf', type=float, default=1.0,
                        help='weight for high-pass filtering')

    # training arguments
    parser.add_argument('--randcrop_prob', type=float, default=0.5,
                        help='Probabilty of using random-resized cropping')
    parser.add_argument('--total_iters', type=int, default=300000,
                        help='Number of total iterations')
    parser.add_argument('--resume_iter', type=int, default=1,
                        help='Iterations to resume training/testing')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=8,
                        help='Batch size for validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for D, E and G')
    parser.add_argument('--f_lr', type=float, default=1e-6,
                        help='Learning rate for F')
    parser.add_argument('--beta1', type=float, default=0.,
                        help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_outs_per_domain', type=int, default=10,
                        help='Number of generated images per domain during sampling')

    # misc
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'finetune', 'sample', 'eval', 'align'],
                        help='This argument is used in solver')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')

    # directory for training
    parser.add_argument('--train_img_dir', type=str, default='data/celeba_hq/train/',
                        help='Directory containing training images')
    parser.add_argument('--val_img_dir', type=str, default='data/celeba_hq/val/',
                        help='Directory containing validation images')
    parser.add_argument('--sample_dir', type=str, default='expr/model_1/Samples/',
                        help='Directory for saving generated images')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/',
                        help='Directory for original network checkpoints')
    parser.add_argument('--checkpoint_save_dir', type=str, default='expr/model_1/Checkpoints/',
                        help='Directory to save network checkpoints')

    # directory for calculating metrics
    parser.add_argument('--eval_dir', type=str, default='expr/model_1/Eval',
                        help='Directory for saving metrics, i.e., FID and LPIPS')
    parser.add_argument('--debug_dir', type=str, default='expr/model_1/Debug/')
    # directory for testing
    parser.add_argument('--result_dir', type=str, default='',
                        help='Directory for saving generated images and videos')
    parser.add_argument('--src_dir', type=str, default='assets/representative/celeba_hq/src',
                        help='Directory containing input source images')
    parser.add_argument('--ref_dir', type=str, default='assets/representative/celeba_hq/ref',
                        help='Directory containing input reference images')
    parser.add_argument('--inp_dir', type=str, default='', help='input directory when aligning faces')
    parser.add_argument('--out_dir', type=str, default='', help='output directory when aligning faces')

    # face alignment
    parser.add_argument('--wing_path', type=str, default='checkpoints/wing.ckpt')
    parser.add_argument('--lm_path', type=str, default='checkpoints/celeba_lm_mean.npz')

    # step size
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=1e6)
    parser.add_argument('--save_every', type=int, default=10000)
    parser.add_argument('--eval_every', type=int, default=1e6)
    parser.add_argument('--debug_every', type=int, default=100)
    parser.add_argument('--masked_up_to', type=int, default=0)
    parser.add_argument('--patch_size', type=int, default=8)

    # Camera configuration
    parser.add_argument('--camera_ckpt', type=str, default='checkpoints/Model_wing.pth')

    # Add weights for the loss LPIPs and MSE mask
    parser.add_argument('--lpips', type=float, default=2000)
    parser.add_argument('--flow', type=float, default=10)

    # Wandb configuration
    parser.add_argument('--use_wandb', action='store_true', default=False, help='Enable wandb logger')
    parser.add_argument('--resume_wandb', action='store_true', default=False, help='Resume wandb logger')
    parser.add_argument('--wandb_name', type=str, default='')
    parser.add_argument('--wandb_dir', type=str, default='./')
    parser.add_argument('--wandb_project', type=str, default='')

    args = parser.parse_args()
    main(args)