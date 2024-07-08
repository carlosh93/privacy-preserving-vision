import os
from os.path import join as ospj
import time
import datetime
from munch import Munch

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import build_model
from core.checkpoint import CheckpointIO
from core.data_loader import InputFetcher
import core.utils as utils
from metrics.eval import calculate_metrics
from Camera.Optics import Camera
import matplotlib.pyplot as plt
from metrics.lpips import LPIPS
from core.utils import loss_RAFT


class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')

        self.nets, self.nets_ema = build_model(args)
        self.camera = Camera(device=self.device, zernike_terms=300).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.lpips = LPIPS().eval().to(self.device)
        self.raft = loss_RAFT(self.device)

        if args.use_wandb:
            try:
                import wandb
            except ImportError:
                print("WandB is not installed. Please install it using 'pip install wandb'.")
                raise SystemExit(1)

        self.seed = args.seed

        if args.camera_ckpt:
            self.Airi_disk = torch.load('./Camera/Airy_Disk_Chromatic.pt')['psf'].to(self.device)
            self.checkpoint_Cam_Wing = torch.load(args.camera_ckpt, map_location=self.device)
            if not 'lr' in args.camera_ckpt.lower():
                self.camera.load_state_dict(self.checkpoint_Cam_Wing['Camera'], strict=True)

        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        if args.mode == 'train':
            self.optims = Munch()

            for net in self.nets.keys():
                if net == 'fan' or net == 'fan_priv':
                    continue
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.f_lr if net == 'mapping_network' else args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay)

            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), data_parallel=True, **self.nets),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_optims.ckpt'), **self.optims),
            ]

        else:
            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), data_parallel=True, **self.nets),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema)]

        self.to(self.device)
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ('ema' not in name) and ('fan' not in name):
                print('Initializing %s...' % name)
                network.apply(utils.he_init)

    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)
        torch.save({'Camera': self.camera.state_dict(), 'Mask': self.nets['fan_priv']}, self.args.checkpoint_save_dir + '/Camera_{:.0f}.pth'.format(step))

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            try:
                ckptio.load(step)
            except:
                print('No Loaded Checkpoint from: ', ckptio.fname_template)
            ckptio.fname_template = ckptio.fname_template.replace('checkpoints', self.args.checkpoint_save_dir)
        self.nets_ema.fan_priv.load_state_dict(self.checkpoint_Cam_Wing['Decoder'], strict=True) if self.args.w_hpf > 0 else None
        # ckpt = torch.load('checkpoints/wing_seg.pth')
        # self.nets_ema.fan.load_state_dict(ckpt['Decoder'], strict=True)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def train(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        # Use the camera only for inference
        self.camera.eval()

        # fetch random validation images for debugging
        fetcher = InputFetcher(loaders.src, loaders.ref, args.latent_dim, 'train')
        fetcher_val = InputFetcher(loaders.val, None, args.latent_dim, 'val')
        inputs_val = next(fetcher_val)

        # resume training if necessary
        if args.resume_iter > 0 or self.args.checkpoint_dir is not None:
            self._load_checkpoint(args.resume_iter)

        print('Start training...')
        start_time = time.time()
        if args.resume_iter >= args.ds_iter:
            args.lambda_ds = 0

        else:
            args.lambda_ds -= args.resume_iter / args.ds_iter * args.lambda_ds

        # remember the initial value of ds weight
        initial_lambda_ds = args.lambda_ds

        for i in range(args.resume_iter, args.total_iters + 1):
            # fetch images and labels
            inputs = next(fetcher)
            x_real_org, y_org = inputs.x_src, inputs.y_src
            x_ref, x_ref2, y_trg = inputs.x_ref, inputs.x_ref2, inputs.y_ref
            z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2

            # ------------------------
            x_real = self.camera(x_real_org).detach()
            # x_real = interpolate(x_real_org, (16, 16)).detach()  # Uncomment if you want to try the low-resolution model
            # x_real = interpolate(x_real, (256, 256)).detach()  # Uncomment if you want to try the low-resolution model
            masks = nets.fan_priv.get_heatmap(x_real, Privacy=True) if args.w_hpf > 0 else None

            # train the discriminator
            d_loss, d_losses_latent = compute_d_loss(nets, args, x_real, y_trg, y_trg, z_trg=z_trg, masks=masks, x_real_org=x_ref)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            d_loss, d_losses_ref = compute_d_loss(nets, args, x_real, y_trg, y_trg, x_ref=x_ref, masks=masks, x_real_org=x_ref)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            # train the generator
            g_loss, g_losses_latent, x_f = compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2], masks=masks, train=True)
            mask_org = nets_ema.fan.get_heatmap(x_real_org, Privacy=False, delimiter=True)[0] > 0.5
            optical_flow = self.raft((x_real_org * mask_org * 255.), (x_f[0] * mask_org * 255.)).requires_grad_(True) * args.flow
            g_losses_latent['Flow'] = optical_flow.item()
            g_loss += optical_flow  # + ldm
            masks_fake = nets.fan.get_heatmap(x_f[0]) if args.w_hpf > 0 else None
            mse = F.l1_loss(masks_fake[0], masks[0]).requires_grad_(True) * 1000
            g_loss += mse
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()
            optims.mapping_network.step()
            optims.style_encoder.step()

            g_loss, g_losses_ref, x_f = compute_g_loss(nets, args, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2], masks=masks, train=True)
            lpips = self.lpips(x_ref, x_f[0]).abs().requires_grad_(True) * args.lpips
            mask_org = nets_ema.fan.get_heatmap(x_real_org, Privacy=False, delimiter=True)[0] > 0.5
            optical_flow = self.raft((x_real_org * mask_org * 255.), (x_f[0] * mask_org * 255.)).requires_grad_(True) * args.flow
            g_losses_ref['Flow'] = optical_flow.item()
            g_losses_ref['LPIPS'] = lpips.item()
            g_loss += lpips + optical_flow
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()

            # compute moving average of network parameters
            moving_average(nets.generator, nets_ema.generator, beta=0.999)
            moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
            moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)

            # decay weight for diversity sensitive loss
            if args.lambda_ds > 0:
                args.lambda_ds -= (initial_lambda_ds / args.ds_iter)

            # print out log info
            if (i + 1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i + 1, args.total_iters)
                all_losses = dict()
                for loss, prefix in zip([d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
                                        ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                all_losses['G/lambda_ds'] = args.lambda_ds
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)
                if self.args.use_wandb:
                    wandb.log(all_losses, step=i)

            # save model checkpoints
            if (i + 1) % args.save_every == 0:
                os.makedirs(args.checkpoint_save_dir, exist_ok=True)
                self._save_checkpoint(step=i + 1)

            if i % args.debug_every == 0:
                with torch.no_grad():
                    masks = nets.fan_priv.get_heatmap(x_real, Privacy=True) if args.w_hpf > 0 else None
                    masks_fake = nets.fan.get_heatmap(x_f[0]) if args.w_hpf > 0 else None
                    masks_ref = nets.fan.get_heatmap(x_ref, Privacy=False) if args.w_hpf > 0 else None
                    masks_rec = nets.fan_priv.get_heatmap(x_f[1], Privacy=True) if args.w_hpf > 0 else None
                    masks_1 = nets.fan.get_heatmap(x_real_org, Privacy=False) if args.w_hpf > 0 else None
                    plt.subplot(251)
                    plt.imshow((x_real_org[0] / x_real_org[0].max()).permute(1, 2, 0).cpu().data.numpy()), plt.title('Org'), plt.axis('off')
                    plt.subplot(252)
                    plt.imshow((x_real[0] / x_real[0].max()).permute(1, 2, 0).cpu().data.numpy()), plt.title('Priv'), plt.axis('off')
                    plt.subplot(253)
                    plt.imshow((x_f[0][0] / x_f[0][0].max()).permute(1, 2, 0).cpu().data.numpy()), plt.title('Fake'), plt.axis('off')
                    plt.subplot(254)
                    plt.imshow((x_f[1][0] / x_f[1][0].max()).permute(1, 2, 0).cpu().data.numpy()), plt.title('Rec'), plt.axis('off')
                    plt.subplot(255)
                    plt.imshow((x_ref[0] / x_ref[0].max()).permute(1, 2, 0).cpu().data.numpy()), plt.title('Ref'), plt.axis('off')
                    plt.subplot(256)
                    plt.imshow(masks_1[0][0, 0].cpu().data, cmap='jet'), plt.axis('off')
                    plt.subplot(257)
                    plt.imshow(masks[0][0, 0].cpu().data, cmap='jet'), plt.axis('off')
                    plt.subplot(258)
                    plt.imshow(masks_fake[0][0, 0].cpu().data, cmap='jet'), plt.axis('off')
                    plt.subplot(259)
                    plt.imshow(masks_rec[0][0, 0].cpu().data, cmap='jet'), plt.axis('off')
                    plt.subplot(2, 5, 10), plt.imshow(masks_ref[0][0, 0].cpu().data, cmap='jet'), plt.axis('off')
                    plt.tight_layout()
                    if self.args.use_wandb:
                        wandb.log({'Debug': plt}, step=i)  # If wandb is not available, log images locally
                    else:
                        os.makedirs(args.debug_dir, exist_ok=True)
                        plt.savefig(args.debug_dir + '/Img_{:.0f}.svg'.format(i))
                    plt.close()

    @torch.no_grad()
    def sample(self, loaders):
        args = self.args
        nets_ema = self.nets_ema
        camera = self.camera
        os.makedirs(args.result_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)
        fetcher_src = InputFetcher(loaders.src, None, args.latent_dim, 'test')
        fetcher_ref = InputFetcher(loaders.ref, None, args.latent_dim, 'test')

        """for i in range(loaders.src.__len__()):
            src = next(fetcher_src)
            utils.debug_image(nets_ema, args, inputs=src, camera=camera, step=i + 1)"""

        for i in range(loaders.src.__len__()):
            src = next(fetcher_src)
            for j in range(loaders.ref.__len__()):
                ref = next(fetcher_ref)
                fname = ospj(args.result_dir, 'reference_{:.0f}_{:.0f}.png'.format(i, j))
                # fname = ospj(args.result_dir, 'ours_{:.0f}{:.0f}.png'.format(i, j))
                print('Working on {}...'.format(fname))
                # utils.translate_using_reference(nets_ema, args, src.x, ref.x, ref.y, fname)
                utils.translate_using_reference(nets_ema, args, src.x, ref.x, ref.y, i, camera=camera)

        # fname = ospj(args.result_dir, 'video_ref_{:.0f}.mp4'.format(i))
        # print('Working on {}...'.format(fname))
        # utils.video_ref(nets_ema, args, src.x, ref.x, ref.y, fname)
        # utils.video_ref(nets_ema, args, src.x, ref.x, ref.y, fname, camera=camera)
        # utils.save_video_from_images(args.result_dir, args.result_dir + '/Proof_video.mp4', 24)

    @torch.no_grad()
    def evaluate(self):
        args = self.args
        nets_ema = self.nets_ema
        resume_iter = args.resume_iter
        self._load_checkpoint(args.resume_iter)
        calculate_metrics(nets_ema, args, step=resume_iter, mode='latent', camera=self.camera)
        calculate_metrics(nets_ema, args, step=resume_iter, mode='reference', camera=self.camera)
        # calculate_metrics(nets_ema, args, step=resume_iter, mode='latent')
        # calculate_metrics(nets_ema, args, step=resume_iter, mode='reference')


def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, masks=None, x_real_org=None):
    assert (z_trg is None) != (x_ref is None)
    # with real images
    if x_real_org is None:
        x_real.requires_grad_()
        out = nets.discriminator(x_real, y_org)
        loss_real = adv_loss(out, 1)
        loss_reg = r1_reg(out, x_real) * args.lambda_reg
    else:
        x_real_org.requires_grad_()
        out = nets.discriminator(x_real_org, y_org)
        loss_real = adv_loss(out, 1)
        loss_reg = r1_reg(out, x_real_org) * args.lambda_reg
    # with fake images
    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_ref, y_trg)

        x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + loss_reg
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())


def compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=None, x_refs=None, masks=None, train=False):
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # adversarial loss
    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else:
        s_trg = nets.style_encoder(x_ref, y_trg)

    x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # style reconstruction loss
    s_pred = nets.style_encoder(x_fake, y_trg)
    loss_sty = args.lambda_sty * torch.mean(torch.abs(s_pred - s_trg))

    # diversity sensitive loss
    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trg2, y_trg)
    else:
        s_trg2 = nets.style_encoder(x_ref2, y_trg)

    x_fake2 = nets.generator(x_real, s_trg2, masks=masks)

    x_fake2 = x_fake2.detach()
    loss_ds = args.lambda_ds * torch.mean(torch.abs(x_fake - x_fake2))

    # cycle-consistency loss
    masks = nets.fan.get_heatmap(x_fake, Privacy=False) if args.w_hpf > 0 else None
    s_org = nets.style_encoder(x_real, y_org)
    x_rec = nets.generator(x_fake, s_org, masks=None)
    loss_cyc = args.lambda_cyc * torch.mean(torch.abs(x_rec - x_real))

    loss = loss_adv + loss_sty - loss_ds + loss_cyc
    if train:
        return loss, Munch(adv=loss_adv.item(), sty=loss_sty.item(), ds=loss_ds.item(), cyc=loss_cyc.item()), [x_fake, x_rec, masks]
    else:
        return loss, Munch(adv=loss_adv.item(), sty=loss_sty.item(), ds=loss_ds.item(), cyc=loss_cyc.item())


def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = \
        torch.autograd.grad(outputs=d_out.sum(), inputs=x_in, create_graph=True, retain_graph=True, only_inputs=True,
                            allow_unused=True)[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg
