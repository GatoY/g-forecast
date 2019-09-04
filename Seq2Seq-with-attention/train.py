# coding: utf-8

'''
Created by JamesYi
Created on 2018/10/23
'''

import argparse
import os
import math
import time
import torch.nn.functional as F
from torch import optim
import torch
from torch.nn.utils import clip_grad_norm
from model import Encoder, AttentionDecoder, Seq2Seq
from utils import load_dataset, device, tokenize_de, tokenize_en

start_time = time.time()


def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=100, help='Number of epochs for training')
    p.add_argument('-batch_size', type=int, default=32, help='Number of batch size for training')
    p.add_argument('-lr', type=float, default=0.0001, help='Initial learning rate for training')
    p.add_argument('-grad_clip', type=float, default=10.0, help='In case of gradient explosion')
    return p.parse_args()


def train(model, optimizer, train_iter, vocab_size, grad_clip, DE, EN):
    model.train()
    total_loss = 0
    pad = EN.vocab.stoi['<pad>']
    for index, batch in enumerate(train_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        src = src.to(device)
        trg = trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        loss = F.nll_loss(output[1:].view(-1, vocab_size), trg[1:].contiguous().view(-1), ignore_index=pad)
        loss.backward()
        clip_grad_norm(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()

        if index % 100 == 0 and index != 0:
            total_loss = total_loss / 100
            now_time = time.time()
            print("{} [Loss: {}] [Time: {}h{}m{}s]"
                  .format(index, total_loss, (now_time - start_time) // 3600,
                  (now_time - start_time) % 3600 // 60, (now_time - start_time) % 60))
            total_loss = 0


def evaluate(model, val_iter, vocab_size, DE, EN):
    model.eval()
    total_loss = 0
    pad = EN.vocab.stoi['<pad>']
    for index, batch in enumerate(val_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        src = src.to(device)
        trg = trg.to(device)
        output = model(src, trg, teacher_forcing_ratio=0)
        loss = F.nll_loss(output[1:].view(-1, vocab_size), trg[1:].contiguous().view(-1), ignore_index=pad)
        total_loss += loss.item()
        return total_loss / len(val_iter)


def online_translator(model, model_save_path, DE, EN):
    model.load_state_dict(torch.load(model_save_path))
    while True:
        s_de = input("Please input a german sentence:")
        if s_de == 'quit':
            break
        else:
            de_list = ['<sos>'] + tokenize_de(s_de) + ['<eos>']
            input_de = []
            for de_i in de_list:
                input_de.append(DE.vocab.stoi[de_i])
            input_de = torch.Tensor(input_de).unsqueeze(1).long().to(device)
            model.eval()
            output_en = model(input_de, input_de, teacher_forcing_ratio=0)
            output_en.squeeze_()
            s_en = ''
            pad = EN.vocab.stoi["<pad>"]
            for en_i in output_en:
                _, top1 = en_i.topk(1)
                if top1.item() == pad:
                    continue
                s_en += EN.vocab.itos[top1.item()] + ' '
            print("Translation result in English: {}".format(s_en))


if __name__ == "__main__":
    args = parse_arguments()
    hidden_size = 512
    embed_size = 256
    print(device)

    print('Loading dataset ......')
    train_iter, val_iter, test_iter, DE, EN = load_dataset(args.batch_size)
    de_size, en_size = len(DE.vocab), len(EN.vocab)
    print("[TRAIN]:%d (dataset:%d)\t[TEST]:%d (dataset:%d)\t[VALUATE]:%d (dataset:%d)"
          % (len(train_iter), len(train_iter.dataset),
             len(test_iter), len(test_iter.dataset),
             len(val_iter), len(val_iter.dataset)))
    print("[DE_vocab]:%d [en_vocab]:%d" % (de_size, en_size))

    print("Initialize model ......")
    encoder = Encoder(de_size, embed_size, hidden_size)
    decoder = AttentionDecoder(en_size, embed_size, hidden_size)
    seq2seq = Seq2Seq(encoder, decoder).to(device)
    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    print(seq2seq)

    best_val_loss = None
    for epoch in range(0, args.epochs):
        train(seq2seq, optimizer,train_iter, en_size, args.grad_clip, DE, EN)
        val_loss = evaluate(seq2seq, val_iter, en_size, DE, EN)
        now_time = time.time()
        print("[Epoch:{}] val_loss:{} | val_pp:{} | Time: {}h{}m{}s".format(epoch, val_loss, math.exp(val_loss), (now_time - start_time) // 3600,
                  (now_time - start_time) % 3600 // 60, (now_time - start_time) % 60))

        if not best_val_loss or val_loss < best_val_loss:
            print("Saving model ......")
            if not os.path.isdir(".save"):
                os.makedirs(".save")
            torch.save(seq2seq.state_dict(), './.save/seq2seq_%d.pt' % (epoch))
            best_val_loss = val_loss
    test_loss = evaluate(seq2seq, test_iter, en_size, DE, EN)
    print("[TEST] loss:{}".format(test_loss))
