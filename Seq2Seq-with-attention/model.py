# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from utils import device


class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, input, hidden=None):
        # hidden: [n_layers * num_dir, B, H]
        # [L, B] -> [L, B, E]
        embedded = self.embedding(input)
        # [L, B, E] -> [L, B, num_dir * H]
        output, hidden = self.gru(embedded, hidden)
        # output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size]
        return output, hidden


class AttentionDecoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, dropout=0.2):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, embed_size)
        self.attention = nn.Linear(hidden_size * 2 + hidden_size, hidden_size)
        self.v = torch.randn(hidden_size, requires_grad=True).to(device)
        self.attention_combined = nn.Linear(hidden_size * 2 + embed_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, dropout=dropout)
        self.out_layer = nn.Linear(hidden_size * 3, output_size)

    def forward(self, input, hidden, encoder_outputs):
        time_steps = encoder_outputs.size(0)
        # [1, B] -> [1, B, E]
        embedded = self.embedding(input.unsqueeze(0))
        # [1, B, H] -> [L, B, H]
        h = hidden.repeat(time_steps, 1, 1)
        # [H] -> [L, H, 1]
        v = self.v.repeat(time_steps, 1).unsqueeze(2)
        # [L, B, 3H] -> [L, B, H] * [L, H, 1] = [L, B, 1] -> [L, B]
        energies = F.tanh(self.attention(torch.cat((h, encoder_outputs), 2))).bmm(v).squeeze(2)
        # [L, B]
        attn_weights = F.softmax(energies, dim=0)
        # [L, B] -> [B, 1, L] * [B, L, 2H] = [B, 1, 2H]
        context = attn_weights.transpose(0, 1).unsqueeze(1).bmm(encoder_outputs.transpose(0,1))
        # [1, B, 2H + E] -> [1, B, H]
        attn_combined = self.attention_combined(torch.cat((context.transpose(0, 1), embedded), 2))
        # [1, B, H], [1, B, H]
        output, hidden_i = self.gru(attn_combined, hidden)
        # [1, B, 3H] -> [1, B, O] -> [B, O]
        output = self.out_layer(torch.cat((output, context.transpose(0, 1)), 2)).squeeze(0)
        output = F.log_softmax(output, dim=1)
        return output, hidden_i


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input, output, teacher_forcing_ratio=0.5):
        batch_size = input.size(1)
        max_len = output.size(0)
        vocab_size = self.decoder.output_size
        outputs = torch.zeros(max_len, batch_size, vocab_size).to(device)

        encoder_outputs, hidden = self.encoder(input)
        hidden = (hidden[0, :, :] + hidden[1, :, :]).unsqueeze(0)
        decoder_input = output[0, :]  # SOS_TOKEN

        for t in range(1, max_len):
            decoder_output, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            outputs[t] = decoder_output
            is_teacher = random.random() < teacher_forcing_ratio
            _, top1 = decoder_output.topk(1)
            top1.squeeze_(1)
            decoder_input = output[t, :] if is_teacher else top1

        return outputs
