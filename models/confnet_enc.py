import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import csv

class ConfnetEncoder(nn.Module):
    def __init__(self, hidden_size=50):
        super(ConfnetEncoder, self).__init__()
        self.word_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.word_bias = nn.Linear(1, 2*hidden_size)#nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))
        self.dropout = nn.Dropout(0.2)

        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.word_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)
        
    def forward(self, input, scores, logger, emb, lengths):
        """
        Based on the paper NEURAL CONFNET CLASSIFICATION (http://150.162.46.34:8080/icassp2018/ICASSP18_USB/pdfs/0006039.pdf)

        """
        # word embedding
        output = emb(input.to('cuda'))
        #output = self.dropout(output)       

        sc = scores.unsqueeze(-1).expand(output.size())

        # confnet score weighted word embedding
        q = output.float() * scores.unsqueeze(-1).expand(output.size()).float()
        #q = q.permute(1,0,2)
        batch_size, max_par_arcs, emb_sz = q.size()

        
        t0 = self.word_weight.unsqueeze(0).expand(batch_size, emb_sz, emb_sz) # weights
        v = torch.nn.functional.tanh(torch.bmm(q,t0)) # new embedding rep

        # align weights
        v_bar = self.context_weight.unsqueeze(0).expand(batch_size, emb_sz, 1) # vweights
        attention = torch.bmm(v, v_bar)
        attention = F.softmax(attention.squeeze())  # attention weights
        if len(attention.size()) == 1:
            attention = attention.unsqueeze(-1)
       
         
        #### masking: Mask the padding ####
        # sentence lens
        lens = torch.tensor(lengths).cuda()
        lens = lens.unsqueeze(dim=1).expand(len(lengths), max_par_arcs) 
        grid = torch.tensor(np.arange(1, max_par_arcs+1)).cuda()
        grid = grid.expand(len(lengths), max_par_arcs)
        mask = grid <= lens 
        masked_attention = attention * mask.float()

        # normalize masked attention
        _sums = torch.sum(masked_attention, dim=1)
        nonzero_index = (_sums != 0).nonzero()
        zero_index = (_sums == 0).nonzero()
        _sums = _sums.unsqueeze(-1).expand_as(masked_attention)
        if nonzero_index.nelement() != 0: # all 0s, no non-zero indices
            _sums[zero_index.squeeze(),:] = 1
        attention = masked_attention.div(_sums)

        # apply attention weights
        output = q*attention.unsqueeze(-1).expand(q.size())

        # most attented words
        most_attentive_arc = torch.argmax(attention, dim=1)
        # highest attention weights
        most_attentive_arc_weights, _ = torch.max(attention, dim=1)
        #a = output
        
        a = torch.sum(output, dim=1) 
        #a = self.dropout(a)

        return a, most_attentive_arc, attention, most_attentive_arc_weights#output, h_output


if __name__ == "__main__":
    abc = WordAttNet("../data/glove.6B.50d.txt")
