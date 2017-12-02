import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop

import numpy as np

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nhidlast, nlayers,
                 dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0,
                 tie_weights=False, ldropout=0.5, n_classes=10, class_count=[]):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.encoder = nn.Embedding(ntoken, ninp)

        self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else nhidlast, 1, dropout=0) for l
                     in range(nlayers)]
        if wdrop:
            self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)

        self.word_class = nn.Linear(nhidlast, n_classes, bias=False)
        # self.latent = nn.Sequential(nn.Linear(nhidlast, n_experts * ninp), nn.Tanh())
        self.latent = nn.Sequential(nn.Linear(nhidlast, ninp), nn.Tanh())
        #self.decoder = nn.Linear(ninp, ntoken + n_classes)
        self.decoder = nn.Linear(ninp, ntoken)
        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        # if tie_weights:
        #     if nhid != ninp:
        #        raise ValueError('When using the tied flag, nhid must be equal to emsize')
        #     self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nhidlast = nhidlast
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.ldropout = ldropout
        self.dropoutl = ldropout
        self.n_classes = n_classes
        self.ntoken = ntoken
        self.class_count = class_count

        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('param size: {}'.format(size))

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False, return_prob=False):
        batch_size = input.size(1)

        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        # emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        # raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                # self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        latent = self.latent(output)
        latent = self.lockdrop(latent, self.dropoutl)
        latent = latent.view(-1, self.ninp)
        logit = self.decoder(latent)

        # (batch_size * seq_len, n_classes)
        prior_logit = self.word_class(output).contiguous().view(-1, self.n_classes)
        prior = nn.functional.softmax(prior_logit)

        probs = []
        true_probs = []
        self.class_count.append(self.ntoken)
        for i in range(self.n_classes):
            index = np.arange(self.class_count[i], self.class_count[i+1])
            if torch.cuda.is_available():
                index = torch.from_numpy(index).cuda()
            else:
                index = torch.from_numpy(index)
            logits = logit.index_select(-1, Variable(index))
            prob_c = nn.functional.softmax(logits)
            true_prob = prob_c * prior[:,i].unsqueeze(-1)
            if not return_prob:
                prob = torch.log(prob_c + 1e-8)
            #prob = prob * prior[:,i].unsqueeze(-1)
            probs.append(prob)
            true_probs.append(true_prob)
        prob = torch.cat(probs, -1)
        true_prob = torch.cat(true_probs, -1)
        # prob = nn.functional.softmax(logit.view(-1, self.ntoken)).view(-1, self.n_experts, self.ntoken)
        # prob = (prob * prior.unsqueeze(2).expand_as(prob)).sum(1)


        if return_prob:
#            model_output = prob
            class_output = prior
        else:
#            p1 = prob + 1e-8
#            log_prob = torch.log(p1)
#            model_output = log_prob
            p = prior + 1e-8
            log_c = torch.log(p)
            class_output = log_c

        #model_output = model_output.view(-1, batch_size, self.ntoken)
        class_output = class_output.view(-1, batch_size, self.n_classes)

        if return_h:
            return class_output, prob, true_prob, hidden, raw_outputs, outputs
        return class_output, prob, true_prob, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return [(Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.nhidlast).zero_()),
                 Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.nhidlast).zero_()))
                for l in range(self.nlayers)]


if __name__ == '__main__':
    model = RNNModel('LSTM', 10, 12, 12, 12, 2)
    input = Variable(torch.LongTensor(13, 9).random_(0, 10))
    hidden = model.init_hidden(9)
    model(input, hidden)

    # input = Variable(torch.LongTensor(13, 9).random_(0, 10))
    # hidden = model.init_hidden(9)
    # print(model.sample(input, hidden, 5, 6, 1, 2, sample_latent=True).size())

