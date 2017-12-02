import os
import torch

from collections import Counter



class Dictionary_softmax(object):
    def __init__(self, label_file, class_num):
        self.word2idx = {}
        self.word2cx = {}
        self.word2cidx = {}
        self.idx2word = {}
        self.class_start_index = []
        self.fake_start_index = []
        self.total = 0
        self.c_num = class_num
        self.c_count = [0] * self.c_num

        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                word, c = line.split()
                c = int(c)
                self.c_count[c] += 1

        sum = 0
        fake_sum = 0
        for i in range(self.c_num):
            self.class_start_index.append(sum)
            self.fake_start_index.append(fake_sum)
            sum += self.c_count[i]
            fake_sum += self.c_count[i] + 1

        self.c_count = [0] * self.c_num

        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                word, c = line.split()
                c = int(c)
                if word not in self.word2idx:
                    index = self.class_start_index[c] + self.c_count[c]
                    self.word2idx[word] = index
                    self.word2cx[word] = c
                    self.word2cidx[word] = self.c_count[c] + 1
                    self.c_count[c] += 1
    def __len__(self):
        return sum(self.c_count)

    def _get_class_num(self):
        return self.fake_start_index

class Corpus_softmax(object):
    def __init__(self, path, dic):
        self.dictionary = dic
        self.train, self.train_cl, self.train_cidx = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid, self.valid_cl, self.valid_cidx = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test, self.test_cl, self.test_cidx = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding='utf-8') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                # for word in words:
                #     self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding='utf-8') as f:
            ids = torch.LongTensor(tokens)
            cs = torch.LongTensor(tokens)
            cids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    cs[token] = self.dictionary.word2cx[word]
                    cids[token] = self.dictionary.word2cidx[word]
                    token += 1

        return ids, cs, cids


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
        print(self.dictionary.__len__())
    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding='utf-8') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding='utf-8') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

class SentCorpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding='utf-8') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        sents = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line:
                    continue
                words = line.split() + ['<eos>']
                sent = torch.LongTensor(len(words))
                for i, word in enumerate(words):
                    sent[i] = self.dictionary.word2idx[word]
                sents.append(sent)

        return sents

class BatchSentLoader(object):
    def __init__(self, sents, batch_size, pad_id=0, cuda=False, volatile=False):
        self.sents = sents
        self.batch_size = batch_size
        self.sort_sents = sorted(sents, key=lambda x: x.size(0))
        self.cuda = cuda
        self.volatile = volatile
        self.pad_id = pad_id

    def __next__(self):
        if self.idx >= len(self.sort_sents):
            raise StopIteration

        batch_size = min(self.batch_size, len(self.sort_sents)-self.idx)
        batch = self.sort_sents[self.idx:self.idx+batch_size]
        max_len = max([s.size(0) for s in batch])
        tensor = torch.LongTensor(max_len, batch_size).fill_(self.pad_id)
        for i in range(len(batch)):
            s = batch[i]
            tensor[:s.size(0),i].copy_(s)
        if self.cuda:
            tensor = tensor.cuda()

        self.idx += batch_size

        return tensor
    
    next = __next__

    def __iter__(self):
        self.idx = 0
        return self

if __name__ == '__main__':
    corpus = SentCorpus('../penn')
    loader = BatchSentLoader(corpus.test, 10)
    for i, d in enumerate(loader):
        print(i, d.size())
