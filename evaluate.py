import os
import json
import logging
from argparse import ArgumentParser, Namespace
from pprint import pprint
from utils import load_dataset, load_model
import time

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dsave', help='save location of model')
    parser.add_argument('--split', help='split to evaluate on', default='dev')
    parser.add_argument('--gpu', type=int, help='gpu to use', default=None)
    parser.add_argument('--fout', help='optional save file to store the predictions')
    parser.add_argument('--dataset', help='dstc or woz dataset', default='woz')
    parser.add_argument('--infer_with_asr', action='store_true', help='inference with asr')
    parser.add_argument('--infer_with_confnet_best_pass', action='store_true', help='inference with confet best pass')
    parser.add_argument('--asr_number', help='number of asr utterances', default=True)
    parser.add_argument('--asr_average_method', help='method for accumulating ASR utterances', default='sum')
    parser.add_argument('--max_par_arc', type=int, default=5, help='max number of parallel arcs in the confnet')
    parser.add_argument('--infer_with_confnet', action='store_true', help='use confnet for inference')
    parser.add_argument('--visualize_attention', action='store_true', help='Visualize the attention weights during eval')
    parser.add_argument('--forward_pass_time', action='store_true', help='forward pass time')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--old_encoder', action="store_true", help="use old encoder")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(os.path.join(args.dsave, 'config.json')) as f:
        j = json.load(f)
        args_save = Namespace(**j)
        print('args_save', type(args_save))
        args_save.gpu = args.gpu
        args_save.forward_pass_time = args.forward_pass_time
        args_save.batch_size = args.batch_size
        args_save.old_encoder = args.old_encoder
    pprint(args_save)

    dataset, ontology, vocab, Eword = load_dataset(args.dataset)

    model = load_model(args_save.model, args_save, ontology, vocab)
    model.load_best_save(directory=args.dsave)
    if args.gpu is not None:
        model.cuda(args.gpu)

    print(dataset.keys())
    if args.split not in dataset.keys():
        print(splits + ' file not found')
    
    #dataset[args.split].dialogues = dataset[args.split].dialogues[:1117]
    logging.info('Making predictions for {} dialogues and {} turns'.format(len(dataset[args.split]), len(list(dataset[args.split].iter_turns()))))
    start = time.time()
    preds, attention_best_pass, most_attentive_arc_weights, all_attention_arcs, padded_confnet_words = model.run_pred(dataset[args.split], args_save)
    print('inference time', (time.time()-start))
    pprint(dataset[args.split].evaluate_preds(preds, vocab, attention_best_pass, most_attentive_arc_weights, all_attention_arcs, padded_confnet_words))

    if args.fout:
        with open(args.fout, 'wt') as f:
            # predictions is a list of sets, need to convert to list of lists to make it JSON serializable
            json.dump([list(p) for p in preds], f, indent=2)
