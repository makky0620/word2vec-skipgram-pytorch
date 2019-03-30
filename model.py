# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

dtype = torch.FloatTensor

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(Word2Vec, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(embedding_size, vocab_size)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, input):
        output = self.embedding(input).view(1, -1)
        output = self.linear(output[0])
        output = self.softmax(output)

        return output


