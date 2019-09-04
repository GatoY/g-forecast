# coding: utf-8

'''
Created by JamesYi
Created on 2018/10/23
'''

import spacy
import torch
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
spacy_de = spacy.load('de')
spacy_en = spacy.load('en')


def load_dataset(batch_size):

    DE = Field(tokenize=tokenize_de, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    EN = Field(tokenize=tokenize_en, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    train, val, test = Multi30k.splits(exts=('.de', '.en'), fields=(DE, EN))
    DE.build_vocab(train.src, min_freq=2)
    EN.build_vocab(train.trg, max_size=10000)
    train_iter, val_iter, test_iter = BucketIterator.splits(
        (train, val, test), batch_size=batch_size, repeat=False)
    return train_iter, val_iter, test_iter, DE, EN


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]
