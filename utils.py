import json
import logging
import os
from pprint import pformat
from importlib import import_module
from vocab import Vocab
from preprocess_data import dann as dann_woz
from preprocess_data import dann_dstc# preprocess_dstc import dann_dstc
import torch
import numpy as np
from torch.autograd import Variable
import random

fix = {'centre': 'center', 'areas': 'area', 'post code': 'postcode', 'dontcare': 'dont care', 'addressess': 'address', 'addresses': 'address', 'gastropub' : 'gastro pub', 'addre': 'address', 'signapore':'singapore', 'vegitarian':'vegetarian', 'catalanian':'catalonian', 'europene':'european', 'portugeuse':'portuguese', 'euorpean':'european', 'asian ori':'asian oriental', 'malyasian':'malaysian', 'restuarant':'restaurant', 'chineese':'chinese', 'chinses':'chinese', 'itailian':'italian', 'signaporian':'singaporean', 'malyasian':'malaysian', 'earetree':'mediterranean', 'adddress':'address', 'nymber':'number', 'arotrian':'eritrean', 'thatll':'that', 'derately':'moderately', 'tailand':'thailand', 'moroccon':'moroccan', 'foo':'food', 'addrss':'address', 'moderat':'moderate', 'eartrain':'eritrean', 'bristish':'british', 'restauran':'restaurant', 'cantonates':'cantonese', 'spani':'spanish', 'scandanavian':'scandinavian', 'ori':'oriental', 'earatree':'mediterranean', 'gasper': 'gastro', 'earatrain':'eritrean', 'ran': 'range', 'restaraunt': 'restaurant', 'ffood':'food', 'pri':'priced', 'halo':'halal', 'canope':'canapes', 'modreately':'moderately', 'mediteranian':'mediterranean', 'endonesian': 'indonesian', 'europ':'european', 'ostro':'australian', 'rerestaurant':'restaurant', 'airitran': 'eritrean', 'turkiesh':'turkish', 'medetanian':'mediterranean', 'restaurnt':'restaurant', 'airatarin':'eritrean', 'vietna':'vietnamese', 'signaporean': 'singaporean', 'medterranean':'mediterranean', 'modereate': 'moderate', 'baskey':'basque', 'modertley':'moderately', 'jamcian': 'jamaican', 'carraibean': 'carribean', 'jamcian':'jamaican', 'fdod':'food', 'veitnamese':'vietnamese', 'addresseses':'address', 'venetian':'venesian', 'brazillian':'brazilian', 'europea':'european', 'fus':'fusion', 'unusal':'unusual', 'fre':'french', 'austral':'austria', 'canopus':'canapes', 'ye':'yes', 'yea':'yeah', 'enlish':'english', 'pricerange': 'price range', 'bask': 'basque', 'vinesha':'venetian','labenese': 'labanese'}

def load_dataset(dataset, splits=('train', 'dev', 'test', 'test_asr')):
    if dataset=='dstc':
        from dataset_dstc_clean_asr import Dataset, Ontology
        dann = dann_dstc
    else:
        from dataset import Dataset, Ontology
        dann = dann_woz

    with open(os.path.join(dann, 'ontology.json')) as f:
        ontology = Ontology.from_dict(json.load(f))
    with open(os.path.join(dann, 'vocab.json')) as f:
        vocab = Vocab.from_dict(json.load(f))
    with open(os.path.join(dann, 'emb.json')) as f:
        E = json.load(f)
    dataset = {}
    for split in splits:
        print(split, os.path.join(dann, '{}.json'.format(split)), os.path.isfile(os.path.join(dann, '{}.json'.format(split))))
        if split == 'test_asr' and not os.path.isfile(os.path.join(dann, '{}.json'.format(split))):
            print('test_asr.json not found')
            continue
        with open(os.path.join(dann, '{}.json'.format(split))) as f:
            logging.warning('loading split {}'.format(split))
            dataset[split] = Dataset.from_dict(json.load(f))
    logging.info('dataset sizes: {}'.format(pformat({k: len(v) for k, v in dataset.items()})))
    return dataset, ontology, vocab, E


def get_models():
    return [m.replace('.py', '') for m in os.listdir('models') if not m.startswith('_') and m != 'model']


def load_model(model, *args, **kwargs):
    print('loading model')
    Model = import_module('models.{}'.format(model)).Model
    print('loaded model')
    print('args', args)
    model = Model(*args, **kwargs)
    logging.info('loaded model {}'.format(Model))
    return model

###### TO-DO: BEST PASS FOR CONFNET AFTER PREPROCESSING ##################
def get_cnet_best_pass(cnet, eos, null_, sos):
    asr1_from_cnet = []
    cnet_score = 0.0
    for par_arcs in cnet:
        max_score = float("-inf")
        max_score_index = 0
        if isinstance(par_arcs[0][0], str):
            null_ = '!null'
            eos = '</s>'
            sos = '<s>'
        for i, a in enumerate(par_arcs):
            if float(a[1]) > max_score:
                max_score_index = i
                max_score = float(a[1])
        cnet_score += max_score
        if par_arcs[max_score_index][0] == eos:
            break
        elif par_arcs[max_score_index][0] != null_ and par_arcs[max_score_index][0] != sos:
            w = par_arcs[max_score_index][0]
            asr1_from_cnet.append(w)
    if len(asr1_from_cnet) == 0:
        asr1_from_cnet = [null_]
    return asr1_from_cnet#' '.join(asr1_from_cnet).strip()

def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = max(sequence_length)
    batch_size = len(sequence_length)
    seq_range = torch.range(0, max_len-1).long()
    seq_len= torch.tensor(np.array(sequence_length)).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    sequence_length = torch.tensor(sequence_length).long()
    seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand
    

def get_cleaned_cnet(cnet, scores, k, threshold, null_, vocab):
    interjections = vocab.word2index(['erm', 'aha', 'uhm' , 'mmm' , 'uhh', 'umm',  'ahh', 'hmm', 'oh', 'em', 'er', 'eh', 'uh', 'mm', 'ah','um', 'oops', 'haha'])
    cleaned_cnet = []
    scores_from_cnet = []
    par_arc_lens = []
    for par_arcs, sc in zip(cnet, scores):
        # Remove arcs with score less than threshold or contains an interjection
        sorted_index = [i[0] for i in sorted(enumerate(sc), key=lambda x:x[1], reverse=True)
                        if par_arcs[i[0]] not in interjections and np.exp(float(sc[i[0]])) > threshold]
        sorted_arcs = [par_arcs[i] for i in sorted_index[:k]]

        # if all parallel arcs are null, remove them to shorten cnet length
        include_in_cnet = any([True if word != null_ else False for word in sorted_arcs])
        if include_in_cnet:
            cleaned_cnet.append(sorted_arcs)
            par_arc_lens.append(len(sorted_arcs))
            scores_from_cnet.append([np.exp(float(sc[i])) for i in sorted_index[:k]])

    return cleaned_cnet, scores_from_cnet, par_arc_lens

def pad_confnet(batch, emb, device, max_par_arc, pad=1, vocab=None):
    max_pararcs_size = max_par_arc
    max_length_sentences = 0
    threshold = 0.001
    padded = []
    scores = []
    lens = []
    null_ = vocab.word2index('!null')
    pad = vocab.word2index('<pad>')
    cnets = []
    topk_scores_list = []
    all_par_arc_lens = []
    final_all_par_arc_lens = []

    # get max cnet length in batch
    for cnet in batch:
        pararc_encode = [[arc[0] for arc in par_arcs] for par_arcs in cnet]
        pararc_scores = [[float(arc[1]) for arc in par_arcs] for par_arcs in cnet]
        cnet, topk_scores, par_arc_lens = get_cleaned_cnet(pararc_encode, pararc_scores, max_pararcs_size, threshold, null_, vocab)

        all_par_arc_lens.append(par_arc_lens)
        cnets.append(cnet)
        l = len(cnet)
        max_length_sentences = l if l > max_length_sentences else max_length_sentences
        lens.append(l)
        topk_scores_list.append(topk_scores)


    for cnet, topk_scores, par_arc_lens in zip(cnets, topk_scores_list, all_par_arc_lens):

        # for OOV words, cnet score of word is set to 0 and word is treated as pad
        cnet = [[word if vocab.index2word(word) in vocab.counts else pad for word in par_arcs] for par_arcs in cnet]
        pararc_scores = [[float(s) if vocab.index2word(word) in vocab.counts else 0.0 for word, s in zip(par_arcs, sc)]
                         for par_arcs, sc in zip(cnet, topk_scores)]

        # padding parallel arcs to max_par_arc_size
        for par_arcs, sent_sc, par_arc_len in zip(cnet, pararc_scores, par_arc_lens):
            sent_sc.extend([0.0] * (max_pararcs_size - len(par_arcs)))
            par_arcs.extend([pad]*(max_pararcs_size - len(par_arcs)))

        # Padding cnet length to max cnet length
        extended_sentences = [[pad]*max_pararcs_size]*(max_length_sentences - len(cnet))
        extended_sentence_scores = [[0.0]*max_pararcs_size]*(max_length_sentences - len(cnet))
        extended_par_arc_lens = [0]*(max_length_sentences - len(cnet))
        cnet.extend(extended_sentences)
        pararc_scores.extend(extended_sentence_scores)
        par_arc_lens.extend(extended_par_arc_lens)

        # Remove parallel arcs beyond max_pararcs_size
        cnet = [sentences[:max_pararcs_size] for sentences in cnet][:max_length_sentences]
        topk_scores = [sentences[:max_pararcs_size] for sentences in pararc_scores][:max_length_sentences]
        cnet = np.stack(arrays=cnet, axis=0)
        topk_scores = np.stack(arrays=topk_scores, axis=0)

        padded.append(cnet)
        scores.append(topk_scores)
        final_all_par_arc_lens.append(par_arc_lens)
    padded = torch.LongTensor(padded).to(device)
    scores = torch.DoubleTensor(scores).to(device)
    final_all_par_arc_lens = torch.LongTensor(final_all_par_arc_lens).to(device)
    return padded, scores, lens, final_all_par_arc_lens#, embedding


def pad_asr(seqs, emb, device, pad=0):
    transpose_seqs = list(zip(*seqs))
    samples = [[list(samp[0]) for samp in batch] for batch in transpose_seqs]
    scores = [torch.tensor([float(samp[1]) for samp in batch]).to(device) for batch in transpose_seqs]
    lens = [[len(samp) for samp in batch] for batch in samples]
    max_lens = [max(samp) for samp in lens]
    padded = []
    for batch_samp, batch_score, len_, max_len in zip(samples, scores, lens, max_lens):
        extended_samples = [samp + [pad] * (max_len - l) for samp, l in zip(batch_samp, len_)]
        padded.append(extended_samples)

    embedded = [emb(torch.LongTensor(p).to(device)) for p in padded]
    return embedded, lens, scores

def pad(seqs, emb, device, pad=0, word_dropout=0.3):
    lens = [len(s) for s in seqs]
    max_len = max(lens)
    padded = torch.LongTensor([s + (max_len - l) * [pad] for s, l in zip(seqs, lens)])
    return emb(padded.to(device)), lens

if __name__ == '__main__':
	dataset, ontology, vocab, E = load_dataset('dstc_clean_confnet') 
