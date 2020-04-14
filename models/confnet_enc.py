import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import csv

class ConfnetEncoder(nn.Module):
    def __init__(self, hidden_size=50):
        super(ConfnetEncoder, self).__init__()
        self.thetav = nn.Linear(in_features=2*hidden_size, out_features=2*hidden_size)
        self.v_bar = nn.Linear(in_features=2 * hidden_size, out_features=1)

    def forward(self, input, scores, logger, emb, lengths, args):
        """
        Based on the paper NEURAL CONFNET CLASSIFICATION (http://150.162.46.34:8080/icassp2018/ICASSP18_USB/pdfs/0006039.pdf)

        """
        # word embedding
        output = emb(input.to('cuda'))
        #output = self.dropout(output)       

        sc = scores.unsqueeze(-1).expand(output.size()).float()

        if args.ver4:
            p = output.float()
        else:
            # confnet score weighted word embedding
            p = output.float() * sc
        #q = q.permute(1,0,2)
        batch_size, max_par_arcs, emb_sz = p.size()
        #print('q size', q.size())
        q = torch.tanh(self.thetav(p))
        #print('v size', v.size())
        alpha = self.v_bar(q).squeeze()
        #### masking: Mask the padding ####
        mask = torch.arange(max_par_arcs)[None, :].to("cuda").type(torch.float) \
               < lengths[:, None].to("cuda").type(torch.float)
        mask = mask.type(torch.float)
        masked_alpha = torch.where(mask == False, torch.tensor([float("-inf") - 1e-10], device=q.device), alpha)
        attention = torch.softmax(masked_alpha, dim=1)
        final_attention = attention.masked_fill(torch.isnan(attention), 0)

        if args.ver1:
            output = q*sc
            a = torch.sum(output, dim=1)
        elif args.ver2:
            a = torch.sum(p, dim=1)
        elif args.ver3:
            output = sc*torch.tanh(self.thetav(output))
            a = torch.sum(output, dim=1)
        else:
            output = q*final_attention.unsqueeze(-1).expand(q.size())
            #print('output size', output.size())

            # most attented words
            most_attentive_arc = torch.argmax(attention, dim=1)
            # highest attention weights
            most_attentive_arc_weights, _ = torch.max(attention, dim=1)

            a = torch.sum(output, dim=1)
        return a#, most_attentive_arc, attention, most_attentive_arc_weights#output, h_output


