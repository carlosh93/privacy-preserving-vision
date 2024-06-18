import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from datasets import *
from models import Encoder, DecoderWithAttention
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
from cider_metric.cider import Cider
import json
import os
from collections import defaultdict
#nltk.download('wordnet')
#nltk.download('omw-1.4')

name = 'coco'
# Parameters
batch_size = 1
pretrained_default = False
data_folder = '../data/' + name  # folder with data files saved by create_input_files.py
data_name = name + '_5_cap_per_img_5_min_word_freq'  # base name shared by data files
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
device = torch.device("cuda")  # sets device for model and PyTorch tensors

cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
decoder = DecoderWithAttention(attention_dim=512, embed_dim=512, decoder_dim=512, vocab_size=9490, dropout=0.5)
encoder = Encoder()

encoder = encoder.to(device)
decoder = decoder.to(device)

def evaluate(beam_size, path):
    """
    Evaluation
    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # DataLoader

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()
    aux = 0


    # For each image
    for batch_idx, (image, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):
        if batch_idx < 10000000:

            k = beam_size

            # Move to GPU device, if available
            image = image.to(device)  # (1, 3, 256, 256)


            info_str = ""
            if aux < limite and (batch_idx + 1) % 50 == 0:

                info_str += "\t {:.0f} ".format(aux)
                plt.imshow(image[0].squeeze().permute(1, 2, 0).detach().cpu().data), plt.axis('off'), plt.savefig(
                    imgs_path + '/org_{:.0f}.png'.format(aux))

            # Encode
            encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
            enc_image_size = encoder_out.size(1)
            encoder_dim = encoder_out.size(3)

            # Flatten encoding
            encoder_out = encoder_out.reshape(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
            num_pixels = encoder_out.size(1)

            # We'll treat the problem as having a batch size of k
            encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

            # Tensor to store top k previous words at each step; now they're just <start>
            k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

            # Tensor to store top k sequences; now they're just <start>
            seqs = k_prev_words  # (k, 1)

            # Tensor to store top k sequences' scores; now they're just 0
            top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

            # Lists to store completed sequences and scores
            complete_seqs = list()
            complete_seqs_scores = list()

            # Start decoding
            step = 1
            h, c = decoder.init_hidden_state(encoder_out)

            # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
            while True:

                embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

                awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

                gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
                awe = gate * awe

                h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

                scores = decoder.fc(h)  # (s, vocab_size)
                scores = F.log_softmax(scores, dim=1)

                # Add
                scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

                # For the first step, all k points will have the same scores (since same k previous words, h, c)
                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                else:
                    # Unroll and find top scores, and their unrolled indices
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

                # Convert unrolled indices to actual indices of scores
                prev_word_inds = top_k_words / vocab_size  # (s)
                next_word_inds = top_k_words % vocab_size  # (s)

                # Add new words to sequences
                seqs = torch.cat([seqs[prev_word_inds.long()], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

                # Which sequences are incomplete (didn't reach <end>)?
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                   next_word != word_map['<end>']]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

                # Set aside complete sequences
                if len(complete_inds) > 0:
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                k -= len(complete_inds)  # reduce beam length accordingly

                # Proceed with incomplete sequences
                if k == 0:
                    break
                seqs = seqs[incomplete_inds]
                h = h[prev_word_inds[incomplete_inds].long()]
                c = c[prev_word_inds[incomplete_inds].long()]
                encoder_out = encoder_out[prev_word_inds[incomplete_inds].long()]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

                # Break if things have been going on too long
                if step > 50:
                    break
                step += 1

            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]

            # References
            img_caps = allcaps[0].tolist()
            img_captions = list(
                map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                    img_caps))  # remove <start> and pads
            references.append(img_captions)

            # Hypotheses
            hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

            assert len(references) == len(hypotheses)

            if aux < limite and (batch_idx + 1) % 50 == 0:

                for j in range(5):
                    org = [rev_word_map[x] for x in references[batch_idx][j]]
                    org = listToString(org)
                    est = [rev_word_map[x] for x in hypotheses[batch_idx]]
                    est = listToString(est)
                    info_str_1 = info_str + "\t {} \t {}\n".format(org, est)
                    f.write(info_str_1)
                aux += 1
                f.write("\n")
        else:
            break


    with open(path + '/Number_ref.json', 'w') as f1:
        json.dump(references, f1)
    with open(path + '/Number_hyp.json', 'w') as f1:
        json.dump(hypotheses, f1)

    # Calculate BLEU-4 scores
    weights = (1.0 / 1.0,)
    bleu1 = corpus_bleu(references, hypotheses, weights)
    weights = (1.0 / 2., 1.0 / 2.,)
    bleu2 = corpus_bleu(references, hypotheses, weights)
    weights = (1.0 / 3., 1.0 / 3., 1.0 / 3.,)
    bleu3 = corpus_bleu(references, hypotheses, weights)
    bleu4 = corpus_bleu(references, hypotheses)

    aux_ref, aux_hyp = num2wordCap(references, hypotheses, rev_word_map)

    with open(path + '/Text_ref.json', 'w') as f1:
        json.dump(aux_ref, f1)
    with open(path + '/Text_hyp.json', 'w') as f1:
        json.dump(aux_hyp, f1)

    meteor = est_Meteor(aux_ref, aux_hyp)

    scorer = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True, split_summaries=True)

    suma = 0
    a = [*range(0, len(aux_ref) + 1, 100)]
    for l in range(len(a) - 1):
        ref, hyp = "", ""
        for i in range(a[l], a[l + 1]):
            hyp += '\n'
            hyp += [" ".join(aux_hyp[i])][0]
            if i % 5 == 0:
                for j in range(5):
                    ref += '\n'
                    ref += " ".join(aux_ref[i][j])
        rouge_value = scorer.score(ref, hyp)['rougeLsum'][2]
        suma += rouge_value
        print("rango: ", a[l], a[l + 1], "rouge_value: ", rouge_value, "len: ", len(ref))
    rouge_value = suma / (len(a) - 1)
    print("total:", rouge_value)


    ref2, hyp2 = [], []
    for i in range(len(aux_ref)):
        hyp2.append([" ".join(aux_hyp[i])][0])
        if i % 5 == 0:
            for j in range(5):
                ref2.append(" ".join(aux_ref[i][j]))

    nose_hyp = []
    nose_ref = []
    keys = ["image_id", "caption"]
    num = 0
    for i in range(len(hyp2)):
        if i % 5 == 0:
            num = num + 1

        nose_hyp.append(dict(zip(keys, ["img_" + str(num) + ".jpg", hyp2[i]])))
        nose_ref.append(dict(zip(keys, ["img_" + str(num) + ".jpg", ref2[i]])))

    ref_list = nose_ref
    cand_list = nose_hyp
    gts = defaultdict(list)
    for l in ref_list:
        gts[l['image_id']].append(l['caption'])

    res = defaultdict(list)
    i = 0
    for l in cand_list:
        if i % 5 == 0:
            res[l['image_id']].append(l['caption'])
        i = i +1

    scorer = Cider()
    score, scores = scorer.compute_score(gts,res)
    #scorer = ciderEval(gts, res, "corpus")
    #scores = scorer.evaluate()
    # scorer += (hypo[0], ref1)
    print('cider = %s' % scorer)

    file_path_M = path + '/Metrics.txt'

    with open(file_path_M, 'w', buffering=1) as ff:
        ff.write("\n\n------------------------------------------------------------------------\n")
        ff.write("\nBLEU-1 score @ beam size of %d is %.4f." % (beam_size, bleu1))
        ff.write("\nBLEU-2 score @ beam size of %d is %.4f." % (beam_size, bleu2))
        ff.write("\nBLEU-3 score @ beam size of %d is %.4f." % (beam_size, bleu3))
        ff.write("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, bleu4))
        ff.write("\nMeteor score @ beam size of %d is %.4f." % (beam_size, meteor))
        ff.write("\nRouge score @ beam size of %d is %.4f." % (beam_size, rouge_value))
        ff.write("\nCider score @ beam size of %d is %.4f." % (beam_size, score))
        ff.write("\n\n------------------------------------------------------------------------\n")

    return bleu4, meteor, rouge_value, score


path = '../results/baseline/'
"""
#path = '../Checkpoint/'
dir_checkpoint = path + 'BEST_checkpoint_'+name+'_5_cap_per_img_5_min_word_freq.pth'
#dir_checkpoint = '../Checkpoint/checkpoint_ICIP_2022.pth'
checkpoint = torch.load(dir_checkpoint, map_location=device)
print('Evaluating Previous Checkpoint at: {} from epoch : {:.0f}'.format(dir_checkpoint, checkpoint['epoch']))
decoder.load_state_dict(checkpoint['decoder'].state_dict())
encoder.load_state_dict(checkpoint['encoder'].state_dict())
del checkpoint
"""
decoder.load_state_dict(torch.load('../Checkpoint/Decoder.pth', map_location=device), strict=False)
encoder.load_state_dict(torch.load('../Checkpoint/Encoder.pth', map_location=device), strict=False)
loader = torch.utils.data.DataLoader(
    CaptionDataset(data_folder, data_name, 'TEST', transform=None),
    batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


encoder = encoder.to(device)
decoder = decoder.to(device)
encoder.eval()
decoder.eval()

# Load word map (word2ix)
word_map_file = '../data/' + name + '/WORDMAP_' + data_name + '.json'

with open(word_map_file, 'r') as jj:
    word_map = json.load(jj)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)


imgs_path = path+'/Imgs/'
if not os.path.exists(imgs_path):
    os.makedirs(imgs_path)

file_path = path + '/Captions.txt'
f = open(file_path, 'w', buffering=1)
f.write('\tIndex \t\t\t\t ORG \t\t\t\t\t EST \n\n')
limite = 100

beam_size = 5
bleu4, meteor, rouge_value, score = evaluate(beam_size, path)
print("FINISH ----- bleu4: ", bleu4, "meteor: ", meteor, "rouge: ", rouge_value, "cider: ", score)
