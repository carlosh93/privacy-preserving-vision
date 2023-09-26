import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
from Camera.Lens import OpticsZernike
import pytorch_ssim
import os
import wandb


# Data parameters
name = 'coco'
data_folder = './data/' + name + '/'  # folder with data files saved by create_input_files.py
data_name = name + '_5_cap_per_img_5_min_word_freq'  # base name shared by data files
# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.3

device = torch.device("cuda:1")  # sets device for model and PyTorch tensors
#device = torch.device("cpu")

cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 100  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 64
workers = 0  # for data-loading; right now, only 1 works with h5py
encoder_lr, decoder_lr, camera_lr = 1e-4, 5e-4, 5e-7
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now --

wandbs = False
camera_train = True
fine_tune_encoder = True  # fine-tune encoder?
reset_bleu = True
prueba_psf = "3"
clamp_zer = True
clamp_size = 1
name_exp = "GPU_final"
exp = './results/'+name_exp+'/'
if not os.path.exists(exp):
    os.mkdir(exp)
#checkpoint = './results/GPU_final/checkpoint_coco_5_cap_per_img_5_min_word_freq.pth'
checkpoint = None # path to checkpoint, None if none
total_steps = 0
print_freq, lim_train, lim_val = 50, 1000000, 3000000
# LOSSES
camera_loss = 'MSE'


def init_camera():

    camera = OpticsZernike(input_shape=[None, 256, 256, 3], device=device, zernike_terms=350, patch_size=256,
                           height_tolerance=2e-8, sensor_distance=0.025, wave_resolution=[896, 896],
                           sample_interval=3e-06, upsample=False)
    camera_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, camera.parameters()),
                                        lr=camera_lr)


    checkpoint_camera = torch.load('./Camera/Model.pth', map_location=device)
    checkpoint_camera['model']['zernike_coeffs_no_train'] = checkpoint_camera['model'][
        'optics.zernike_coeffs_no_train']
    checkpoint_camera['model']['zernike_coeffs_train'] = checkpoint_camera['model'][
        'optics.zernike_coeffs_train']
    del checkpoint_camera['model']['optics.zernike_coeffs_train']
    del checkpoint_camera['model']['optics.zernike_coeffs_no_train']
    camera.load_state_dict(checkpoint_camera['model'])


    return camera, camera_optimizer

def main():

    global criterion, camera_loss, best_bleu4, lim_train, lim_val, checkpoint_icip, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map, total_steps, prueba_psf, clamp_zer,clamp_size, camera_train, name_exp

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    if checkpoint is None:
        print("NO CP.... Exp Name....")
        print(name_exp)
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       encoder_dim=2048,
                                       vocab_size=len(word_map),
                                       dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None


        decoder.load_state_dict(torch.load('./Checkpoint/Decoder.pth', map_location=device), strict=False)
        encoder.load_state_dict(torch.load('./Checkpoint/Encoder.pth', map_location=device), strict=False)
        camera, camera_optimizer = init_camera()

        #camera = OpticsZernike(input_shape=[None, 256, 256, 3], device=device, zernike_terms=350, patch_size=256,
        #                       height_tolerance=2e-8, sensor_distance=0.025, wave_resolution=[896, 896],
        #                       sample_interval=3e-06, upsample=False)
        #camera_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, camera.parameters()),
        #                                    lr=camera_lr)

    else:

        print('Loading Checkpoint....')
        print(checkpoint)
        print("Exp Name....")
        print(name_exp)

        checkpoint = torch.load(checkpoint, map_location=device)

        #checkpoint_icip = torch.load(checkpoint_icip, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        if reset_bleu is True:
            epochs_since_improvement = 0
            best_bleu4 = 0.20
        else:
            epochs_since_improvement = checkpoint['epochs_since_improvement']
            best_bleu4 = checkpoint['bleu-4']

        camera = OpticsZernike(input_shape=[None, 256, 256, 3], device=device, zernike_terms=350, patch_size=256,
                               height_tolerance=2e-8, sensor_distance=0.025, wave_resolution=[896, 896],
                               sample_interval=3e-06, upsample=False)
        camera.load_state_dict(checkpoint['camera'].state_dict())
        camera_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, camera.parameters()),
                                        lr=camera_lr)
        #camera_optimizer.load_state_dict(checkpoint['camera_optimizer'].state_dict())
        #reset_learning_rate(camera_optimizer, camera_lr)

        encoder = checkpoint['encoder']
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                            lr=encoder_lr)
        #encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'].state_dict())
        #reset_learning_rate(encoder_optimizer, encoder_lr)

        decoder = checkpoint['decoder']
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                           lr=decoder_lr)
        #decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'].state_dict())
        #reset_learning_rate(decoder_optimizer, decoder_lr)
        #reset_learning_rate(camera_optimizer, camera_lr)
        # camera, camera_optimizer = init_camera()

        print('Checkpoint loaded successfully, start from epoch {:.1f}...'.format(start_epoch))
        del checkpoint
        #del checkpoint_icip


    # Move to GPU, if available

    camera = camera.to(device)
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)
    if camera_loss == 'SSIM':
        noise_loss = pytorch_ssim.SSIM()
    elif camera_loss == 'MSE':
        noise_loss = torch.nn.MSELoss()

    # Custom dataloaders

    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN'),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL'),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    if wandbs:
        wandb.init(project="test-project", entity="", name=name_exp,
                   config={"start_epoch": start_epoch, "epochs": epochs, "batch_size": batch_size, "camera_lr": camera_lr,
                           "encoder_lr": encoder_lr, "decoder_lr": decoder_lr, "train_camera": camera_train, })


    for epoch in range(start_epoch, epochs):

    #    if epoch % 5 == 0:

    #        adjust_learning_rate(camera_optimizer, 0.8)
        #if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:

            #if fine_tune_encoder:
                #(encoder_optimizer, 0.5)

        total_steps, loss_camera = train(train_loader=train_loader,
                                      encoder=encoder,
                                      decoder=decoder,
                                      camera=camera,
                                      encoder_optimizer=encoder_optimizer,
                                      decoder_optimizer=decoder_optimizer,
                                      camera_optimizer=camera_optimizer,
                                      epoch=epoch,
                                      total_steps=total_steps,
                                      noise_loss=noise_loss,
                                      criterion=criterion)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                camera=camera,
                                noise_loss=noise_loss,
                                criterion=criterion)

        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if wandbs:
            wandb.log({"Bleu-4": recent_bleu4})
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
        wish_bleu = 0.20
        terrible = recent_bleu4 < wish_bleu

        if not terrible:
            save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, camera, encoder_optimizer,
                            decoder_optimizer, camera_optimizer, recent_bleu4, is_best, exp)
        else:
            print("STOP STOP STOP STOP STOP STOP -- B L E U < 20.0 -- STOP STOP STOP STOP STOP STOP ")
            print("STOP STOP STOP STOP STOP STOP -- B L E U < 20.0 -- STOP STOP STOP STOP STOP STOP ")




def train(train_loader, encoder, decoder, camera, encoder_optimizer, decoder_optimizer, camera_optimizer, epoch, total_steps, noise_loss, criterion):

    encoder.train()
    decoder.train()
    if camera_train:
        camera.train()
    else:
        camera.eval()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy
    loss_c = 0
    start = time.time()

    for i, (imgs, caps, caplens) in enumerate(train_loader):

        if i < lim_train:

            data_time.update(time.time() - start)
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            aux = imgs[0]
            aux2 = imgs

            sensor, psf, zernike_coeffs_train, loss_psf = camera(imgs, None, prueba_psf)

            encoder_out = encoder(sensor)

            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(encoder_out, caps, caplens)
            targets = caps_sorted[:, 1:]

            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            loss_ce = criterion(scores.data, targets.data)
            loss_dsr = alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
            loss_decoder = loss_ce + loss_dsr
            # lambda_loss = 8*(2.2+math.cos(total_steps/4+math.pi))
            loss_camera = 1 - noise_loss(aux2, sensor)
            if camera_train:
                loss = (0.4 * loss_decoder) + (6 * loss_camera) + (30 * loss_psf)
            else:
                loss = 10 * loss_decoder
            loss_c = loss_camera

            if wandbs:
                if i % print_freq == 0:

                    wandb.log({"loss": loss, "Camera loss": loss_camera, "Decoder loss": loss_decoder, "PSF loss": loss_psf,
                               "Original": wandb.Image(aux / aux.max()), "Sensor": wandb.Image(sensor[0] / sensor[0].max()),
                               "PSF": wandb.Image((psf / psf.max()).permute(3, 1, 2, 0).squeeze(3)),
                               "4th Zernike Pol": zernike_coeffs_train.squeeze().cpu().data[3],
                               "5th Zernike Pol": zernike_coeffs_train.squeeze().cpu().data[4],
                               "6th Zernike Pol": zernike_coeffs_train.squeeze().cpu().data[5]
                               })
                    total_steps += 1

            decoder_optimizer.zero_grad()
            if camera_train:
                camera_optimizer.zero_grad()
            if encoder_optimizer is not None:
                encoder_optimizer.zero_grad()
            loss.backward()
            if camera_train:
                camera_optimizer.step()

            if grad_clip is not None:
                clip_gradient(decoder_optimizer, grad_clip)
                if encoder_optimizer is not None:
                    clip_gradient(encoder_optimizer, grad_clip)


            decoder_optimizer.step()
            if encoder_optimizer is not None:
                encoder_optimizer.step()

            if clamp_zer:
                camera.zernike_coeffs_train[1:].data.clamp_(-clamp_size, clamp_size)

            top5 = accuracy(scores.data, targets.data, 5)
            losses.update(loss.item(), sum(decode_lengths))
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)
            start = time.time()
            if loss_psf is None:
                loss_psf = 12345

            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'loss_camera {loss_camera:.4f}\t'
                      'Loss decoder {loss_decoder:.4f}\t'
                      'Loss psf {loss_psf:.4f}\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                              batch_time=batch_time,
                                                                              data_time=data_time, loss=losses,
                                                                              loss_camera=loss_camera,
                                                                              loss_decoder=loss_decoder,
                                                                              loss_psf=loss_psf,
                                                                              top5=top5accs))

        else:
            break

    return total_steps, loss_c


def validate(val_loader, encoder, decoder, camera, noise_loss, criterion):

    encoder.eval()
    decoder.eval()
    camera.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()
    hypotheses = list()

    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            if i < lim_val:
                imgs = imgs.to(device)
                caps = caps.to(device)
                caplens = caplens.to(device)
                aux2 = imgs

                sensor, psf, zernike_coeffs_train, loss_psf = camera(imgs, None, prueba_psf)

                encoder_out = encoder(sensor)

                scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(encoder_out, caps, caplens)
                targets = caps_sorted[:, 1:]

                # Remove timesteps that we didn't decode at, or are pads
                # pack_padded_sequence is an easy trick to do this
                scores_copy = scores.clone()
                scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
                targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

                loss_ce = criterion(scores.data, targets.data)
                loss_dsr = alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
                loss_decoder = loss_ce + loss_dsr
                # lambda_loss = 8*(2.2+math.cos(total_steps/4+math.pi))
                loss_camera = 1 - noise_loss(aux2, sensor)
                if camera_train:
                    loss = (0.4 * loss_decoder) + (6 * loss_camera) + (30 * loss_psf)
                else:
                    loss = 10 * loss_decoder
                loss_c = loss_camera

                losses.update(loss.item(), sum(decode_lengths))
                top5 = accuracy(scores.data, targets.data, 5)
                top5accs.update(top5, sum(decode_lengths))
                batch_time.update(time.time() - start)

                start = time.time()

                if loss_psf is None:
                    loss_psf = 12345

                if i % print_freq == 0:
                    print('Validation: [{0}/{1}]\t'
                          'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Loss images {loss_camera:.4f}\t'
                          'Loss decoder {loss_decoder:.4f}\t'
                          'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
                                                                                    batch_time=batch_time,
                                                                                    loss=losses, loss_camera=loss_camera,
                                                                                    loss_decoder=loss_decoder, top5=top5accs))


                allcaps = allcaps.cpu()[sort_ind.cpu()]

                for j in range(allcaps.shape[0]):
                    img_caps = allcaps[j].tolist()
                    img_captions = list(
                        map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                            img_caps))
                    references.append(img_captions)

                _, preds = torch.max(scores_copy, dim=2)
                preds = preds.tolist()
                temp_preds = list()
                for j, p in enumerate(preds):
                    temp_preds.append(preds[j][:decode_lengths[j]])
                preds = temp_preds
                hypotheses.extend(preds)

                assert len(references) == len(hypotheses)

            else:
                break

        bleu4 = corpus_bleu(references, hypotheses)
        print(
            '\n * LOSS  {loss.avg:.3f}\t TOP-5 ACCURACY {top5.avg:.3f}\t BLEU-4  {bleu}\n'.format(loss=losses, top5=top5accs, bleu=bleu4))

    return bleu4

if __name__ == '__main__':
    main()
