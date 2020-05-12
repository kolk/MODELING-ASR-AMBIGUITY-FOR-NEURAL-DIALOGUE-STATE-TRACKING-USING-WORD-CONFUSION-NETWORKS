import json
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from stanza.nlp.corenlp import CoreNLPClient
import itertools
from jiwer import wer
from utils import get_cnet_best_pass
import csv
from math import exp
from functools import reduce
import time

client = None
def cnet_best_n_paths(confusion_network,n,paths):
    """Prints n best paths in list format with each element as a pair of string and log probability.Takes actual probability as input"""
    confusion_network=[list(sorted(i,key=lambda x:x[1],reverse=True)) for i in confusion_network]
    if confusion_network:
        if paths:
            new_addition=[[[l[0]],l[1]] for l in confusion_network[0][:n]]
            paths=list(itertools.product(paths,new_addition))
            paths=[[reduce(lambda x,y:x[0]+y[0],path),reduce(lambda x,y:x[1]+y[1],path)] for path in paths]
            paths=list(sorted(paths,key=lambda x:x[1],reverse=True))[:n]
            return cnet_best_n_paths(confusion_network[1:],n,paths)
        else:
            paths=confusion_network[0][:n] #[['<s>', 0.9999000049998333], ['!null', 0.0]]
            paths=[[[l[0]],l[1]] for l in paths]        #[[['<s>'], 0.9999000049998333], [['!null'], 0.0]]
            return cnet_best_n_paths(confusion_network[1:],n,paths)
    else:
        return paths

def annotate(sent):
    global client
    if client is None:
        client = CoreNLPClient(default_annotators='ssplit,tokenize'.split(','))
    words = []
    for sent in client.annotate(sent).sentences:
        for tok in sent:
            words.append(tok.word)
    return words


class Turn:

    def __init__(self, turn_id, transcript, turn_label, belief_state, system_acts, system_transcript, asr, cnet=None, cnet_asr=None, num=None):
        self.id = turn_id
        self.transcript = transcript
        self.turn_label = turn_label
        self.belief_state = belief_state
        self.system_acts = system_acts
        self.system_transcript = system_transcript
        self.cnet = cnet
        self.num = num or {}
        self.asr = asr
        self.cnet_asr = cnet_asr

    def to_dict(self):
        return {'turn_id': self.id, 'transcript': self.transcript, 'turn_label': self.turn_label, 'belief_state': self.belief_state, 'system_acts': self.system_acts, 'system_transcript': self.system_transcript, 'num': self.num, 'asr':self.asr, 'cnet': self.cnet, 'cnet_asr':self.cnet_asr}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    @classmethod
    def annotate_raw(cls, raw, split='train', transcript2cnet=False, trancriptAsAsr=False):
        system_acts = []
        for a in raw['system_acts']:
            if isinstance(a, list):
                s, v = a
                system_acts.append(['inform'] + s.split() + ['='] + v.split())
            else:
                system_acts.append(['request'] + a.split())
        # NOTE: fix inconsistencies in data label
        fix = {'centre': 'center', 'areas': 'area', 'post code': 'postcode', 'dontcare': 'dont care', 'addressess': 'address', 'addresses': 'address', 'gastropub' : 'gastro pub', 'addre': 'address', 'signapore':'singapore', 'vegitarian':'vegetarian', 'catalanian':'catalonian', 'europene':'european', 'portugeuse':'portuguese', 'euorpean':'european', 'asian ori':'asian oriental', 'malyasian':'malaysian', 'restuarant':'restaurant', 'chineese':'chinese', 'chinses':'chinese', 'itailian':'italian', 'signaporian':'singaporean', 'malyasian':'malaysian', 'earetree':'mediterranean', 'adddress':'address', 'nymber':'number', 'arotrian':'eritrean', 'thatll':'that', 'derately':'moderately', 'tailand':'thailand', 'moroccon':'moroccan', 'foo':'food', 'addrss':'address', 'moderat':'moderate', 'eartrain':'eritrean', 'bristish':'british', 'restauran':'restaurant', 'cantonates':'cantonese', 'spani':'spanish', 'scandanavian':'scandinavian', 'ori':'oriental', 'earatree':'mediterranean', 'gasper': 'gastro', 'earatrain':'eritrean', 'ran': 'range', 'restaraunt': 'restaurant', 'ffood':'food', 'pri':'priced', 'halo':'halal', 'canope':'canapes', 'modreately':'moderately', 'mediteranian':'mediterranean', 'endonesian': 'indonesian', 'europ':'european', 'ostro':'australian', 'rerestaurant':'restaurant', 'airitran': 'eritrean', 'turkiesh':'turkish', 'medetanian':'mediterranean', 'restaurnt':'restaurant', 'airatarin':'eritrean', 'vietna':'vietnamese', 'signaporean': 'singaporean', 'medterranean':'mediterranean', 'modereate': 'moderate', 'baskey':'basque', 'modertley':'moderately', 'jamcian': 'jamaican', 'carraibean': 'carribean', 'jamcian':'jamaican', 'fdod':'food', 'veitnamese':'vietnamese', 'addresseses':'address', 'venetian':'venesian', 'brazillian':'brazilian', 'europea':'european', 'fus':'fusion', 'unusal':'unusual', 'fre':'french', 'austral':'austria', 'canopus':'canapes', 'ye':'yes', 'yea':'yeah', 'enlish':'english', 'pricerange': 'price range', 'bask': 'basque', 'vinesha':'venetian','labenese': 'labanese'}
        cnet = []
        if transcript2cnet:
            cnet.append([['<s>', 0.0]])
            cnet += [[[fix.get(word, word), 0.0]] for word in raw['transcript'].strip().split()]
            cnet.append([['</s>', 0.0]])
        else:
            cnet += [[[arc['word'], arc['score']] for arc in par_arcs['arcs']] for par_arcs in raw['cnet']]

        # CNET n-best paths
        n_best_paths = cnet_best_n_paths(cnet,10,[])
        compressed_n_best_paths = []
        if trancriptAsAsr:
            ts = annotate(raw['transcript'])
            compressed_n_best_paths = [(ts,1.0) for i in range(10)]
        else:
            n_best_paths = cnet_best_n_paths(cnet,10,[])
            for path in n_best_paths:
                s = [word for word in path[0] if word != '<s>' and word != '!null' and word != '</s>']
                compressed_n_best_paths.append((annotate(' '.join(s)), np.exp(path[1])))

        return cls(
            turn_id=raw['turn_idx'],
            transcript=annotate(raw['transcript']),
            system_acts=system_acts,
            turn_label=[[fix.get(s.strip(), s.strip()), fix.get(v.strip(), v.strip())] for s, v in raw['turn_label'] if s != ''],
            belief_state=raw['belief_state'],
            system_transcript=raw['system_transcript'],
            asr=[(annotate(hyp[0]), hyp[1]) for hyp in raw['asr']],
            cnet=cnet,
            cnet_asr=compressed_n_best_paths
        )

    def transcript_to_cnf(cls, transcript):
        pass

    def numericalize_(self, vocab):
        self.num['transcript'] = vocab.word2index(['<sos>'] + [w.lower() for w in self.transcript + ['<eos>']], train=True)
        self.num['system_acts'] = [vocab.word2index(['<sos>'] + [w.lower() for w in a] + ['<eos>'], train=True) for a in self.system_acts + [['<sentinel>']]]
        self.num['asr'] = [(vocab.word2index(['<sos>'] + [w.lower() for w in hyp[0]] + ['<eos>'], train=True), hyp[1]) for hyp in self.asr]
        self.num['cnet'] = [[[vocab.word2index(arc[0], train=True), arc[1]] for arc in par_arcs] for par_arcs in self.cnet]
        self.num['cnet_asr'] = [(vocab.word2index(['<sos>'] + [w.lower() for w in hyp[0]] + ['<eos>'], train=True), hyp[1]) for hyp in self.cnet_asr]

class Dialogue:

    def __init__(self, dialogue_id, turns):
        self.id = dialogue_id
        self.turns = turns

    def __len__(self):
        return len(self.turns)

    def to_dict(self):
        return {'dialogue_id': self.id, 'turns': [t.to_dict() for t in self.turns]}

    @classmethod
    def from_dict(cls, d):
        return cls(d['dialogue_id'], [Turn.from_dict(t) for t in d['turns']])

    @classmethod
    def annotate_raw(cls, raw, split='train', transcript2cnet=False, trancriptAsAsr=False):
        print('dialogue idx', raw['dialogue_idx'])
        return cls(raw['dialogue_idx'], [Turn.annotate_raw(t, split, transcript2cnet, trancriptAsAsr) for t in raw['dialogue']])


class Dataset:

    def __init__(self, dialogues):
        self.dialogues = dialogues

    def __len__(self):
        return len(self.dialogues)

    def iter_turns(self, indices=None):
        for d in self.dialogues:
            for t in d.turns:
                yield t

    def to_dict(self):
        return {'dialogues': [d.to_dict() for d in self.dialogues]}

    @classmethod
    def from_dict(cls, d):
        return cls([Dialogue.from_dict(dd) for dd in d['dialogues']])

    @classmethod
    def annotate_raw(cls, fname, split='train', transcript2cnet=False, trancriptAsAsr=False):
        with open(fname) as f:
            data = json.load(f)
            return cls([Dialogue.annotate_raw(d, split, transcript2cnet, trancriptAsAsr) for d in tqdm(data)])

    def numericalize_(self, vocab):
        for t in self.iter_turns():
            t.numericalize_(vocab)

    def extract_ontology(self):
        slots = set()
        values = defaultdict(set)
        for t in self.iter_turns():
            for s, v in t.turn_label:
                slots.add(s.lower())
                values[s].add(v.lower())
        return Ontology(sorted(list(slots)), {k: sorted(list(v)) for k, v in values.items()})

    def batch(self, batch_size, shuffle=False, indices=None, vocab=None):
        turns = list(self.iter_turns())
        if shuffle:
            np.random.shuffle(turns)
        for i in tqdm(range(0, len(turns), batch_size)):
            yield turns[i:i+batch_size]

    def evaluate_preds(self, preds, vocab=None, attention_best_pass=None, most_attentive_arc_weights=None, all_attention_arcs=None, padded_confnet_words=None):
        request = []
        inform = []
        joint_goal = []
        fix = {'centre': 'center', 'areas': 'area', 'post code': 'postcode', 'dontcare': 'dont care', 'addressess': 'address', 'addresses': 'address', 'gastropub' : 'gastro pub', 'addre': 'address', 'signapore':'singapore', 'vegitarian':'vegetarian', 'catalanian':'catalonian', 'europene':'european', 'portugeuse':'portuguese', 'euorpean':'european', 'asian ori':'asian oriental', 'malyasian':'malaysian', 'restuarant':'restaurant', 'chineese':'chinese', 'chinses':'chinese', 'itailian':'italian', 'signaporian':'singaporean', 'malyasian':'malaysian', 'earetree':'mediterranean', 'adddress':'address', 'nymber':'number', 'arotrian':'eritrean', 'thatll':'that', 'derately':'moderately', 'tailand':'thailand', 'moroccon':'moroccan', 'foo':'food', 'addrss':'address', 'moderat':'moderate', 'eartrain':'eritrean', 'bristish':'british', 'restauran':'restaurant', 'cantonates':'cantonese', 'spani':'spanish', 'scandanavian':'scandinavian', 'ori':'oriental', 'earatree':'mediterranean', 'gasper': 'gastro', 'earatrain':'eritrean', 'ran': 'range', 'restaraunt': 'restaurant', 'ffood':'food', 'pri':'priced', 'halo':'halal', 'canope':'canapes', 'modreately':'moderately', 'mediteranian':'mediterranean', 'endonesian': 'indonesian', 'europ':'european', 'ostro':'australian', 'rerestaurant':'restaurant', 'airitran': 'eritrean', 'turkiesh':'turkish', 'medetanian':'mediterranean', 'restaurnt':'restaurant', 'airatarin':'eritrean', 'vietna':'vietnamese', 'signaporean': 'singaporean', 'medterranean':'mediterranean', 'modereate': 'moderate', 'baskey':'basque', 'modertley':'moderately', 'jamcian': 'jamaican', 'carraibean': 'carribean', 'jamcian':'jamaican', 'fdod':'food', 'veitnamese':'vietnamese', 'addresseses':'address', 'venetian':'venesian', 'brazillian':'brazilian', 'europea':'european', 'fus':'fusion', 'unusal':'unusual', 'fre':'french', 'austral':'austria', 'canopus':'canapes', 'ye':'yes', 'yea':'yeah', 'enlish':'english', 'pricerange': 'price range', 'bask': 'basque', 'vinesha':'venetian','labenese': 'labanese', 'goodbye':"good bye", "dont":"do n't", "addresses": "address", "center":"centre", "im":"i 'm", "whats": "what 's", "gastropub": "gastro pub", "id": "i 'd", "doesnt": "does n't", "seafood": "sea food", "postcode": "post code", "steakhouse":"steak house"}
        stopwords = ['erm', 'aha', 'uhm' , 'mmm' , 'uhh', 'umm',  'ahh', 'hmm', 'oh', 'em', 'er', 'eh', 'uh', 'mm', 'ah','um', 'oops', 'haha']
        i = 0
        j = 0
        #total_time = 0
        for d in self.dialogues:
            pred_state = {}
            gold_state = {}
            j+=1
            turn_time = 0.0
            for t in d.turns:
                #start = time.time()
                gold_request = set([(s, v) for s, v in t.turn_label if s == 'request'])
                gold_inform = set([(s, v) for s, v in t.turn_label if s != 'request' and v != ''])
                pred_request = set([(s, v) for s, v in preds[i] if s == 'request'])
                pred_inform = set([(s, v) for s, v in preds[i] if s != 'request'])
                request.append(gold_request == pred_request)
                inform.append(gold_inform == pred_inform)

                gold_recovered = set()
                pred_recovered = set()
                for s, v in pred_inform:
                    pred_state[s] = v

                for s, v in gold_inform:
                    gold_state[s] = v
                for s, v in pred_state.items():
                    pred_recovered.add(('inform', s, v))
                for s, v in gold_state.items():
                    gold_recovered.add(('inform', s, v))
                joint_goal.append(gold_recovered == pred_recovered)
                i += 1
                #end = time.time()
                #turn_time += (end-start)
                #total_time += turn_time
                #print("turn time", (end-start))
                
        #print('total time', total_time)
        return {'turn_inform': np.mean(inform), 'turn_request': np.mean(request), 'joint_goal': np.mean(joint_goal)}

    def record_preds(self, preds, to_file):
        data = self.to_dict()
        i = 0
        for d in data['dialogues']:
            for t in d['turns']:
                t['pred'] = sorted(list(preds[i]))
                i += 1
        with open(to_file, 'wt') as f:
            json.dump(data, f)


class Ontology:

    def __init__(self, slots=None, values=None, num=None):
        self.slots = slots or []
        self.values = values or {}
        self.num = num or {}

    def __add__(self, another):
        new_slots = sorted(list(set(self.slots + another.slots)))
        new_values = {s: sorted(list(set(self.values.get(s, []) + another.values.get(s, [])))) for s in new_slots}
        return Ontology(new_slots, new_values)

    def __radd__(self, another):
        return self if another == 0 else self.__add__(another)

    def to_dict(self):
        return {'slots': self.slots, 'values': self.values, 'num': self.num}

    def numericalize_(self, vocab):
        self.num = {}
        for s, vs in self.values.items():
            self.num[s] = [vocab.word2index(annotate('{} = {}'.format(s, v)) + ['<eos>'], train=True) for v in vs]

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
