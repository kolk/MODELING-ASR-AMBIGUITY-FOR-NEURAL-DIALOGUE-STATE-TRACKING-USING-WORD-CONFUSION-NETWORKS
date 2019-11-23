#!/usr/bin/env python3
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from utils import load_dataset, get_models, load_model
import os
import logging
import numpy as np
from pprint import pprint
import torch
from random import seed


def run(args):
    pprint(args)
    logging.basicConfig(level=logging.INFO)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    seed(args.seed)

    dataset, ontology, vocab, Eword = load_dataset(args.dataset)
    model = load_model(args.model, args, ontology, vocab)
    model.save_config()
    model.load_emb(Eword)

    model = model.to(model.device)
    if not args.test and not args.resume:
        print('Starting train', torch.cuda.is_available())
        model.run_train(dataset['train'], dataset['dev'], args)
    if args.resume:
        model.load_best_save(directory=args.resume)#args.dout)
        model.run_train(dataset['train'], dataset['dev'], args)
    else:
        model.load_best_save(directory=args.dout)
    model = model.to(model.device)
    logging.info('Running dev evaluation')
    dev_out = model.run_eval(dataset['test'], args)
    pprint(dev_out)


def get_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dexp', help='root experiment folder', default='exp')
    parser.add_argument('--model', help='which model to use', default='glad', choices=get_models())
    parser.add_argument('--epoch', help='max epoch to run for', default=150, type=int)
    parser.add_argument('--demb', help='word embedding size', default=400, type=int)
    parser.add_argument('--dhid', help='hidden state size', default=200, type=int)
    parser.add_argument('--batch_size', help='batch size', default=50, type=int)
    parser.add_argument('--lr', help='learning rate', default=1e-3, type=float)
    parser.add_argument('--eps', help='adam eplison', default=1e-8, type=float)
    parser.add_argument('--stop', help='slot to early stop on', default='joint_goal')
    parser.add_argument('--resume', help='save directory to resume from', type=str)#action='store_true')
    parser.add_argument('-n', '--nick', help='nickname for model', default='default')
    parser.add_argument('--seed', default=42, help='random seed', type=int)
    parser.add_argument('--test', action='store_true', help='run in evaluation only mode')
    parser.add_argument('--gpu', type=int, help='which GPU to use')
    parser.add_argument('--emb_dropout', type=float, default=0.2, help='embedding dropout')
    parser.add_argument('--local_dropout', type=float, default=0.2, help='local dropout')
    parser.add_argument('--global_dropout', type=float, default=0.2, help='global dropout')
    parser.add_argument('--selfattn_dropout', type=float, default=0.0, help='self attention dropout')
    #parser.add_argument('--dropout', nargs='*', help='dropout rates', default=['emb=0.2', 'local=0.2', 'global=0.2'])
    parser.add_argument("--word_dropout", help='rate at which word embedding is set to 0', default=0.3, type=float)
    parser.add_argument('--dataset', type=str, help='dataset to use: woz or dstc', default='woz')
    parser.add_argument('--threshold', help='sigmoid threshold', default=0.5, type=float)
    parser.add_argument('--infer_with_asr', action='store_true', help='use asr for inference')
    parser.add_argument('--infer_with_confnet', action='store_true', help='use confnet for inference')
    parser.add_argument('--asr_number', type=int, default=1, help='number of asr utterance used during inference')
    parser.add_argument('--asr_average_method', type=str, default='sum', help='method to accumulate ASR utterances: sum, wrighted_sum, mean')
    parser.add_argument('--train_using', type=str, default='transcript', help='train using [asr, transcript, confnet]')
    parser.add_argument('--max_par_arc', type=int, default=5, help='max number of parallel arcs in the confnet')
    parser.add_argument('--infer_with_confnet_best_pass', action='store_true', help='use confnet for inference')
    parser.add_argument('--visualize_attention', action='store_true', help='Visualize the attention weights during eval')
    parser.add_argument('--joint_training', action='store_true', help='Train asr or confnet embeddings jointly with transcript')

    args = parser.parse_args()
    args.dout = os.path.join(args.dexp, args.model, args.nick)
    if args.resume:
        args.resume = os.path.join(args.resume)
    #args.dropout = {d.split('=')[0]: float(d.split('=')[1]) for d in args.dropout}
    if not os.path.isdir(args.dout):
        os.makedirs(args.dout)
    return args


if __name__ == '__main__':
    args = get_args()
    run(args)
