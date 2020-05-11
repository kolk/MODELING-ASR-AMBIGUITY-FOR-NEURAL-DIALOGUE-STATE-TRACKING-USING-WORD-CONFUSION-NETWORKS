import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
import logging
import os
import re
import json
from collections import defaultdict
from pprint import pformat
import numpy as np
import random
from utils import sequence_mask, get_cnet_best_pass, get_cleaned_cnet, pad_confnet, pad, pad_asr
import time
from models.confnet_enc import ConfnetEncoder

def run_rnn(rnn, inputs, lens):
    # sort by lens
    order = np.argsort(lens)[::-1].tolist()
    reindexed = inputs.index_select(0, inputs.data.new(order).long())
    reindexed_lens = [lens[i] for i in order]
    packed = nn.utils.rnn.pack_padded_sequence(reindexed, reindexed_lens, batch_first=True)
    outputs, _ = rnn(packed)
    padded, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, padding_value=0.)
    reverse_order = np.argsort(order).tolist()
    recovered = padded.index_select(0, inputs.data.new(reverse_order).long())
    return recovered


def attend(seq, cond, lens):
    """
    attend over the sequences `seq` using the condition `cond`.
    """
    scores = cond.unsqueeze(1).expand_as(seq).mul(seq).sum(2)
    max_len = max(lens)
    for i, l in enumerate(lens):
        if l < max_len:
            scores.data[i, l:] = -np.inf
    scores = F.softmax(scores, dim=1)
    context = scores.unsqueeze(2).expand_as(seq).mul(seq).sum(1)
    return context, scores


class FixedEmbedding(nn.Embedding):
    """
    this is the same as `nn.Embedding` but detaches the result from the graph and has dropout after lookup.
    """

    def __init__(self, *args, dropout=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout = dropout

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        out.detach_()
        return F.dropout(out, self.dropout, self.training)


class SelfAttention(nn.Module):
    """
    scores each element of the sequence with a linear layer and uses the normalized scores to compute a context over the sequence.
    """

    def __init__(self, d_hid, dropout=0.):
        super().__init__()
        self.scorer = nn.Linear(d_hid, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, lens):
        batch_size, seq_len, d_feat = inp.size()
        inp = self.dropout(inp)
        scores = self.scorer(inp.contiguous().view(-1, d_feat)).view(batch_size, seq_len)
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores.data[i, l:] = -np.inf
        scores = F.softmax(scores, dim=1)
        context = scores.unsqueeze(2).expand_as(inp).mul(inp).sum(1)
        return context

class GLADEncoder(nn.Module):
    """
    the GLAD encoder described in https://arxiv.org/abs/1805.09655.
    """

    def __init__(self, din, dhid, slots, local_dropout=None, global_dropout=None, selfattn_dropout=0.):#dropout=None):
        super().__init__()
        #self.dropout = dropout or {}
        self.local_dropout=local_dropout
        self.global_dropout=global_dropout
        self.selfattn_dropout=selfattn_dropout
        self.global_rnn = nn.LSTM(din, dhid, bidirectional=True, batch_first=True)
        #self.global_rnn = self.global_rnn.cuda()
        self.global_selfattn = SelfAttention(2 * dhid, dropout=self.selfattn_dropout)
        for s in slots:
            setattr(self, '{}_rnn'.format(s), nn.LSTM(din, dhid, bidirectional=True, batch_first=True, dropout=0.))
            setattr(self, '{}_selfattn'.format(s), SelfAttention(din, dropout=self.selfattn_dropout))
        self.slots = slots
        self.beta_raw = nn.Parameter(torch.Tensor(len(slots)))
        nn.init.uniform_(self.beta_raw, -0.01, 0.01)

    def beta(self, slot):
        return torch.sigmoid(self.beta_raw[self.slots.index(slot)])

    def forward(self, x, x_len, slot, utterance2=None, utterance2_len=None, default_dropout=0.2):
        local_rnn = getattr(self, '{}_rnn'.format(slot))
        local_selfattn = getattr(self, '{}_selfattn'.format(slot))
        beta = self.beta(slot)
        #local_rnn = local_rnn.cuda()        
        local_h = run_rnn(local_rnn, x, x_len)
        global_h = run_rnn(self.global_rnn, x, x_len)
        local_dropout = self.local_dropout if self.local_dropout else default_dropout
        global_dropout = self.global_dropout if self.global_dropout else default_dropout
        h = F.dropout(local_h, local_dropout, self.training) * beta + F.dropout(global_h, global_dropout, self.training) * (1-beta)

        h2 = None
        c2 = None
        if utterance2 is not None and utterance2_len is not None:
            local_h2 = run_rnn(local_rnn, utterance2, utterance2_len)
            global_h2 = run_rnn(self.global_rnn, utterance2, utterance2_len)
            h2 = F.dropout(local_h2, local_dropout, self.training) * beta + F.dropout(global_h2, global_dropout, self.training) * (1-beta)
            c2 = F.dropout(local_selfattn(h2, utterance2_len), local_dropout, self.training) * beta + F.dropout(self.global_selfattn(h2, utterance2_len), global_dropout, self.training) * (1-beta)
        c = F.dropout(local_selfattn(h, x_len), local_dropout, self.training) * beta + F.dropout(self.global_selfattn(h, x_len), global_dropout, self.training) * (1-beta)
        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()

        return h, c, h2, c2


class Model(nn.Module):
    """
    the GLAD model described in https://arxiv.org/abs/1805.09655.
    """

    def __init__(self, args, ontology, vocab):
        super().__init__()
        self.optimizer = None
        self.args = args
        self.vocab = vocab
        self.ontology = ontology

        self.l2 = torch.nn.CosineEmbeddingLoss()
        self.l2_norm_loss = torch.nn.MSELoss()
        self.emb_fixed = FixedEmbedding(len(vocab), args.demb, dropout=args.emb_dropout)#args.dropout.get('emb', 0.2))
        self.confnet_encoder = ConfnetEncoder(hidden_size=int(args.demb/2), device=self.device)
        self.utt_encoder = GLADEncoder(args.demb, args.dhid, self.ontology.slots, local_dropout=args.local_dropout, global_dropout=args.global_dropout, selfattn_dropout=args.selfattn_dropout)#dropout=args.dropout)
        self.act_encoder = GLADEncoder(args.demb, args.dhid, self.ontology.slots, local_dropout=args.local_dropout, global_dropout=args.global_dropout, selfattn_dropout=args.selfattn_dropout)#dropout=args.dropout)
        self.ont_encoder = GLADEncoder(args.demb, args.dhid, self.ontology.slots, local_dropout=args.local_dropout, global_dropout=args.global_dropout, selfattn_dropout=args.selfattn_dropout) #dropout=args.dropout)
        self.utt_scorer = nn.Linear(2 * args.dhid, 1)
        self.score_weight = nn.Parameter(torch.Tensor([0.5]))

    @property
    def device(self):
        if self.args.gpu is not None and torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def set_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.args.lr)#, eps=0.1)

    def load_emb(self, Eword):
        new = self.emb_fixed.weight.data.new
        self.emb_fixed.weight.data.copy_(new(Eword))

    def infer_with_asr(self, batch, asr_utterance, asr_utterance_len, acts, ontology, asr_scores, args):
        lambda_ = 0.01
        method = args.asr_average_method
        ys = dict()#defaultdict(torch.tensor)
        for i, (utterance, utterance_len, asr_score) in enumerate(zip(asr_utterance, asr_utterance_len, asr_scores)):
            if i >= int(args.asr_number):
                break
            for j, s in enumerate(self.ontology.slots):
                # for each slot, compute the scores for each value
                H_utt, c_utt, H_utt2, c_utt2 = self.utt_encoder(utterance, utterance_len, slot=s)
                _, C_acts, _, _ = list(zip(*[self.act_encoder(a, a_len, slot=s) for a, a_len in acts]))
                _, C_vals, _, _ = self.ont_encoder(ontology[s][0], ontology[s][1], slot=s)

                # compute the utterance score
                q_utts = []
                for c_val in C_vals:
                    q_utt, _ = attend(H_utt, c_val.unsqueeze(0).expand(len(batch), *c_val.size()), lens=utterance_len)
                    q_utts.append(q_utt)
                y_utts = self.utt_scorer(torch.stack(q_utts, dim=1)).squeeze(2)

                # compute the previous action score
                q_acts = []
                for i, C_act in enumerate(C_acts):
                    q_act, _ = attend(C_act.unsqueeze(0), c_utt[i].unsqueeze(0), lens=[C_act.size(0)])
                    q_acts.append(q_act)
                y_acts = torch.cat(q_acts, dim=0).mm(C_vals.transpose(0, 1))

                # Aggregate the scores from k ASR utterances
                temp = y_utts + self.score_weight * y_acts
                if method == 'sum':
                    ys.setdefault(s, torch.zeros_like(temp))
                    ys[s]+=temp
                elif method == 'weighted_sum':
                    ys.setdefault(s, torch.zeros_like(temp))
                    asr_score_ = asr_score.repeat(temp.size()[1], 1).t()
                    ys[s] += temp * asr_score_
                else:
                    ys[s] = y_utts + self.score_weight * y_acts

        if self.training:
            # create label variable and compute loss
            labels = {s: [len(self.ontology.values[s]) * [0] for i in range(len(batch))] for s in self.ontology.slots}
            for p, e in enumerate(batch):
                for s, v in e.turn_label:
                    labels[s][p][self.ontology.values[s].index(v)] = 1
            labels = {s: torch.Tensor(m).to(self.device) for s, m in labels.items()}
        
        loss = 0
        for s in self.ontology.slots:
            ys[s] = torch.sigmoid(ys[s])
            if self.training:
                loss += F.binary_cross_entropy(ys[s], labels[s])
                if args.joint_training:
                    # l2 = self.l2(c_utt, c_utt2, torch.tensor(1.0).cuda()
                    loss += lambda_*self.l2_norm_loss(c_utt, c_utt2)
        if not self.training:
            loss = torch.Tensor([0]).to(self.device)

        return loss, ys

    def infer_with_transcript(self, batch, utterance, utterance_len, acts, ontology, args, utterance2=None, utterance2_len=None):
        ys = {}

        lambda_ = 0.01#1e-1#1e-06#0.000001
        for s in self.ontology.slots:
            # for each slot, compute the scores for each value
            _, C_acts, _, _ = list(zip(*[self.act_encoder(a, a_len, slot=s) for a, a_len in acts]))
            _, C_vals, _, _ = self.ont_encoder(ontology[s][0], ontology[s][1], slot=s)
            H_utt, c_utt, H_utt2, c_utt2 = self.utt_encoder(utterance, utterance_len, slot=s, utterance2=utterance2, utterance2_len=utterance2_len)

            # compute the utterance score
            q_utts = []
            cnt_diff = []
            for c_val in C_vals:
                q_utt, _ = attend(H_utt, c_val.unsqueeze(0).expand(len(batch), *c_val.size()), lens=utterance_len)
                q_utt2, _ = attend(H_utt2, c_val.unsqueeze(0).expand(len(batch), *c_val.size()), lens=utterance2_len)
                cnt_diff.append(torch.dist(q_utt, q_utt2, p=2))
                q_utts.append(q_utt)
            y_utts = self.utt_scorer(torch.stack(q_utts, dim=1)).squeeze(2)
            
            # compute the previous action score
            q_acts = []
            for i, C_act in enumerate(C_acts):
                q_act, _ = attend(C_act.unsqueeze(0), c_utt[i].unsqueeze(0), lens=[C_act.size(0)])
                q_acts.append(q_act)
            y_acts = torch.cat(q_acts, dim=0).mm(C_vals.transpose(0, 1))

            # combine the scores
            ys[s] = torch.sigmoid(y_utts + self.score_weight * y_acts)
        if self.training:
            # create label variable and compute loss
            labels = {s: [len(self.ontology.values[s]) * [0] for i in range(len(batch))] for s in self.ontology.slots}
            for i, e in enumerate(batch):
                for s, v in e.turn_label:
                    labels[s][i][self.ontology.values[s].index(v)] = 1
            labels = {s: torch.Tensor(m).to(self.device) for s, m in labels.items()}
            loss = 0
            for s in self.ontology.slots:
                loss += F.binary_cross_entropy(ys[s], labels[s])
                if args.joint_training:
                    # l2 = self.l2(c_utt, c_utt2, torch.tensor(1.0).cuda()
                    loss += lambda_*self.l2_norm_loss(c_utt, c_utt2)#context_diff
        else:
            loss = torch.Tensor([0]).to(self.device)
        return loss, ys


    def forward(self, batch, args, logger):
        # convert to method == 'variables and look up embeddings
        eos = self.vocab.word2index('<eos>')
        acts = [pad(e.num['system_acts'], self.emb_fixed, self.device, pad=eos) for e in batch]
        ontology = {s: pad(v, self.emb_fixed, self.device, pad=eos) for s, v in self.ontology.num.items()}
         
        if self.training:
            """ 
            if args.train_using == 'confnet_best_pass':
                utterance, utterance_len = pad([get_cnet_best_pass(e.num['cnet'], self.vocab.word2index('</s>'), self.vocab.word2index('!null'), self.vocab.word2index('<s>')) for e in batch], self.emb_fixed, self.device, pad=eos)
                loss, ys = self.infer_with_transcript(batch, utterance, utterance_len, acts, ontology, args)
                return loss, {s: v.data.tolist() for s, v in ys.items()}
            """
            # train using a confusion-network
            if args.train_using == 'confnet' or args.train_using == 'aug_confnet':
                # pad the confusion network to max parallel arc size
                padded_confnet, scores, sent_lens, all_par_arc_lens = pad_confnet([e.num['cnet'] for e in batch], self.emb_fixed, self.device,  args.max_par_arc, vocab=self.vocab)
                utterance, utterance_len = pad([e.num['transcript'] for e in batch], self.emb_fixed, self.device, pad=eos)
                output_list = torch.tensor([]).to(self.device)

                padded_confnet_ = padded_confnet.permute(1, 0, 2) # (max_sent_len, batch, max_par_arc_len)
                scores_ = scores.permute(1, 0, 2) # (max_sent_len, batch, max_par_arc_len)
                all_par_arc_lens_ = all_par_arc_lens.permute(1,0) #(lens, batch)
                for i, sc, par_arc_lens in zip(padded_confnet_, scores_, all_par_arc_lens_): # for each par_arcs at time i in the batch
                    #output_confnet, most_attentive_arc, all_attention_arcs, most_attentive_arc_weights = self.confnet_encoder(i, sc, logger, self.emb_fixed, par_arc_lens)
                    output_confnet=self.confnet_encoder(i, sc, logger, self.emb_fixed, par_arc_lens, args).contiguous()
                    output_list = torch.cat((output_list, output_confnet.unsqueeze(0)), dim=0)

                output_confnet = output_list.permute(1,0,2) #(batch, max_sent_len, hid_dim)
                # the confusion network is represented by a 1D sequence of hidden representation
                loss, ys = self.infer_with_transcript(batch, output_confnet, sent_lens, acts, ontology, args, utterance, utterance_len)
                return loss, {s: v.data.tolist() for s, v in ys.items()}
            
            """
            elif args.train_using == 'transcript':
                utterance, utterance_len = pad([e.num['transcript'] for e in batch], self.emb_fixed, self.device, pad=eos)
                batch_size, max_len, dim = utterance.size()
                # word dropout if infer with asr during evaluation
                if self.infer_with_asr:
                    for s in range(batch_size):
                        for w in range(max_len):
                            if random.random() <= self.args.word_dropout:
                                utterance[s,w,:] = torch.zeros_like(utterance[s,w,:])
                loss, ys = self.infer_with_transcript(batch, utterance, utterance_len, acts, ontology, args)
                return loss, {s: v.data.tolist() for s, v in ys.items()}

            
            elif args.train_using == 'asr' or args.train_using == 'aug_asr':
                asr_utterance, asr_utterance_len, asr_scores = pad_asr([e.num['cnet_asr'] for e in batch], self.emb_fixed, self.device, pad=eos)
                loss, ys = self.infer_with_asr(batch, asr_utterance, asr_utterance_len, acts, ontology, asr_scores, args)
                return loss, {s: v.data.tolist() for s, v in ys.items()}
            """
        else:
            # Evaluation
            """
            if args.infer_with_asr:
                asr_utterance, asr_utterance_len, asr_scores = pad_asr([e.num['cnet_asr'] for e in batch], self.emb_fixed, self.device, pad=eos)
                loss, ys = self.infer_with_asr(batch, asr_utterance, asr_utterance_len, acts, ontology, asr_scores, args)
                return loss, {s: v.data.tolist() for s, v in ys.items()}, None, None, None, None
            elif args.infer_with_confnet_best_pass:
                utterance, utterance_len = pad([get_cnet_best_pass(e.num['cnet'], self.vocab.word2index('</s>'), self.vocab.word2index('!null'), self.vocab.word2index('<s>')) for e in batch], self.emb_fixed, self.device, pad=eos)
                loss, ys = self.infer_with_transcript(batch, utterance, utterance_len, acts, ontology, args)
                return loss, {s: v.data.tolist() for s, v in ys.items()}
            """
            if args.infer_with_confnet:
                utterance, utterance_len = pad([e.num['transcript'] for e in batch], self.emb_fixed, self.device, pad=eos)
                padded_confnet, scores, sent_lens, all_par_arc_lens = pad_confnet([e.num['cnet'] for e in batch], self.emb_fixed, self.device, args.max_par_arc, pad=eos, vocab=self.vocab)
                output_list = torch.tensor([]).to(self.device)
                padded_confnet = padded_confnet.permute(1, 0, 2) # (max_sent_len, batch, max_par_arc_len)
                scores = scores.permute(1, 0, 2)
                all_par_arc_lens = all_par_arc_lens.permute(1,0)


                # most attended words
                attention_best_pass = torch.tensor([], dtype=torch.long).to(self.device)#torch.zeros((padded_confnet.shape(1), padded_confnet.shape(0), dtype=np.float)
                # all attention weights
                batch_all_attention_arcs = torch.tensor([], dtype=torch.float).to(self.device)
                batch_most_attentive_arc_weights = torch.tensor([], dtype=torch.float).to(self.device)
                for i, sc, par_arc_lens in zip(padded_confnet, scores, all_par_arc_lens):
                    #output_confnet, best_arc_indices, all_attention_arcs, most_attentive_arc_weights  = self.confnet_encoder(i, sc, logger, self.emb_fixed, par_arc_lens)
                    output_confnet = self.confnet_encoder(i, sc, logger, self.emb_fixed, par_arc_lens, args)
                    output_list = torch.cat((output_list, output_confnet.unsqueeze(0)), dim=0)

                    if args.visualize_attention:
                        attention_best_pass = torch.cat((attention_best_pass, i[torch.tensor(torch.arange(best_arc_indices.size(0)),
                                     dtype=torch.long), best_arc_indices].unsqueeze(0)), dim=0)
                        batch_all_attention_arcs = torch.cat((batch_all_attention_arcs, all_attention_arcs.unsqueeze(0)),
                                                         dim=0)
                        batch_most_attentive_arc_weights = torch.cat((batch_most_attentive_arc_weights, most_attentive_arc_weights.unsqueeze(0)), dim=0)

                output_confnet = output_list.permute(1,0,2) #(batch, max_sent_len, hid_dim)
                loss, ys = self.infer_with_transcript(batch, output_confnet, sent_lens, acts, ontology, args, utterance, utterance_len)

                attention_best_pass_sents = []
                padded_confnet_words = []
                batch_most_attentive_arc_weights = []
                batch_all_attention_arcs = []
                if args.visualize_attention:
                    attention_best_pass = attention_best_pass.permute(1,0)
                    batch_most_attentive_arc_weights = batch_most_attentive_arc_weights.permute(1,0)
                    batch_all_attention_arcs = batch_all_attention_arcs.permute(1,0,2)
                    padded_confnet = padded_confnet.permute(1,0,2)
                    for sent in attention_best_pass:#.view( padded_confnet.size(0), -1):
                        best_sent = [self.vocab.index2word(word.item()) for word in sent]
                        attention_best_pass_sents.append(' '.join(best_sent))

                    padded_confnet_words = [[[self.vocab.index2word(word.item()) for word in par_arcs]
                                             for par_arcs in sent] for sent in padded_confnet]

                return loss, {s: v.data.tolist() for s, v in ys.items()}, attention_best_pass_sents, batch_most_attentive_arc_weights, batch_all_attention_arcs, padded_confnet_words
            else:
                utterance, utterance_len = pad([e.num['transcript'] for e in batch], self.emb_fixed, self.device, pad=eos)
                loss, ys, loss2 = self.infer_with_transcript(batch, utterance, utterance_len, acts, ontology, args)
                return loss, loss2, {s: v.data.tolist() for s, v in ys.items()}
            

    def get_train_logger(self):
        logger = logging.getLogger('train-{}'.format(self.__class__.__name__))
        formatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s')
        file_handler = logging.FileHandler(os.path.join(self.args.dout, 'train.log'))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger


    def get_test_logger(self):
        logger = logging.getLogger('test-{}'.format(self.__class__.__name__))
        formatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s')
        file_handler = logging.FileHandler(os.path.join(self.args.dout, 'test.log'))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger


    def run_train(self, train, dev, args):
        track = defaultdict(list)
        iteration = 0
        best = {}
        logger = self.get_train_logger()
        if self.optimizer is None:
            self.set_optimizer()

        for epoch in range(args.epoch):
            logger.info('starting epoch {}'.format(epoch))

            # train and update parameters
            self.train()
            for batch in train.batch(batch_size=args.batch_size, shuffle=True, vocab=self.vocab):
                iteration += 1
                self.zero_grad()
                loss, scores = self.forward(batch, args, logger)
                loss.backward()
                self.optimizer.step()
                track['loss'].append(loss.item())

            # evalute on train and dev
            summary = {'iteration': iteration, 'epoch': epoch}
            for k, v in track.items():
                summary[k] = sum(v) / len(v)
            summary.update({'eval_train_{}'.format(k): v for k, v in self.run_eval(train, args).items()})
            summary.update({'eval_dev_{}'.format(k): v for k, v in self.run_eval(dev, args).items()})

            # do early stopping saves
            stop_key = 'eval_dev_{}'.format(args.stop)
            train_key = 'eval_train_{}'.format(args.stop)
            if best.get(stop_key, 0) <= summary[stop_key]:
                best_dev = '{:f}'.format(summary[stop_key])
                best_train = '{:f}'.format(summary[train_key])
                best.update(summary)
                self.save(
                    best,
                    identifier='lr={lr},eps={eps},threshold={threshold},wdout={word_dropout},adout={selfattn_dropout},bsz={batch_size},epoch={epoch},iter={iteration},train_{key}={train},dev_{key}={dev}'.format(
                        lr=args.lr, eps=args.eps, threshold=args.threshold, word_dropout=args.word_dropout, selfattn_dropout=args.selfattn_dropout, batch_size=args.batch_size, epoch=epoch, iteration=iteration, train=best_train, dev=best_dev, key=args.stop,
                    )
                )
                self.prune_saves()
                preds, all_attention_best_pass, all_most_attentive_arc_weights, all_attention_arcs, all_padded_confnet_words =self.run_pred(dev, self.args)
                dev.record_preds(
                    preds=preds,
                    to_file=os.path.join(self.args.dout, 'dev.pred.json'),
                )
            summary.update({'best_{}'.format(k): v for k, v in best.items()})
            logger.info(pformat(summary))
            track.clear()

    def extract_predictions(self, scores, threshold=0.5):
        batch_size = len(list(scores.values())[0])
        predictions = [set() for i in range(batch_size)]
        for s in self.ontology.slots:
            if s == '':
                continue
            for i, p in enumerate(scores[s]):
                triggered = [(s, v, p_v) for v, p_v in zip(self.ontology.values[s], p) if p_v > threshold]
                if s == 'request':
                    # we can have multiple requests predictions
                    predictions[i] |= set([(s, v) for s, v, p_v in triggered])
                elif triggered:
                    # only extract the top inform prediction
                    sort = sorted(triggered, key=lambda tup: tup[-1], reverse=True)
                    predictions[i].add((sort[0][0], sort[0][1]))
        return predictions

    def run_pred(self, dev, args):
        logger = self.get_test_logger()
        self.eval()
        predictions = []
        all_attention_best_pass = []#torch.tensor([], dtype=torch.long).cuda()
        all_most_attentive_arc_weights = [] 
        all_attention_arcs = []
        all_padded_confnet_words = []
        for batch in dev.batch(batch_size=args.batch_size, vocab=self.vocab):
            loss, scores, attention_best_pass, batch_most_attentive_arc_weights, batch_all_attention_arcs, batch_padded_confnet_words = self.forward(batch, args, logger)
            if args.infer_with_confnet and args.visualize_attention:
                all_attention_best_pass.extend(attention_best_pass)#torch.cat((all_attention_best_pass, attention_best_pass), dim=0)
                all_most_attentive_arc_weights.extend(batch_most_attentive_arc_weights)
                all_attention_arcs.extend(batch_all_attention_arcs)
                all_padded_confnet_words.extend(batch_padded_confnet_words)
            predictions += self.extract_predictions(scores)
        return predictions, all_attention_best_pass, all_most_attentive_arc_weights, all_attention_arcs, all_padded_confnet_words

    def run_eval(self, dev, args):
        predictions, attention_best_pass, all_most_attentive_arc_weights, all_attention_arcs, all_padded_confnet_words = self.run_pred(dev, args)
        return dev.evaluate_preds(predictions, self.vocab, attention_best_pass, all_most_attentive_arc_weights, all_attention_arcs, all_padded_confnet_words)

    def save_config(self):
        fname = '{}/config.json'.format(self.args.dout)
        with open(fname, 'wt') as f:
            logging.info('saving config to {}'.format(fname))
            json.dump(vars(self.args), f, indent=2)

    @classmethod
    def load_config(cls, fname, ontology, **kwargs):
        with open(fname) as f:
            logging.info('loading config from {}'.format(fname))
            args = object()
            for k, v in json.load(f):
                setattr(args, k, kwargs.get(k, v))
        return cls(args, ontology)

    def save(self, summary, identifier):
        fname = '{}/{}.t7'.format(self.args.dout, identifier)
        logging.info('saving model to {}'.format(fname))
        state = {
            'args': vars(self.args),
            'model': self.state_dict(),
            'summary': summary,
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, fname)

    def load(self, fname):
        logging.info('loading model from {}'.format(fname))
        state = torch.load(fname, map_location=torch.device("cpu"))
        self.load_state_dict(state['model'])
        self.set_optimizer()
        self.optimizer.load_state_dict(state['optimizer'])

    def get_saves(self, directory=None):
        if directory is None:
            directory = self.args.dout
        files = [f for f in os.listdir(directory) if f.endswith('.t7')]
        scores = []
        for fname in files:
            re_str = r'dev_{}=([0-9\.]+)'.format(self.args.stop)
            dev_acc = re.findall(re_str, fname)
            if dev_acc:
                score = float(dev_acc[0].strip('.'))
                scores.append((score, os.path.join(directory, fname)))
        if not scores:
            raise Exception('No files found!')
        scores.sort(key=lambda tup: tup[0], reverse=True)
        return scores

    def prune_saves(self, n_keep=5):
        scores_and_files = self.get_saves()
        if len(scores_and_files) > n_keep:
            for score, fname in scores_and_files[n_keep:]:
                os.remove(fname)

    def load_best_save(self, directory):
        if directory is None:
            directory = self.args.dout

        scores_and_files = self.get_saves(directory=directory)
        if scores_and_files:
            assert scores_and_files, 'no saves exist at {}'.format(directory)
            score, fname = scores_and_files[0]
            self.load(fname)
