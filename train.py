# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from pprint import pprint
from copy import deepcopy

from model import Word2Vec

def random_batch(data, size, vocab_size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)

    for i in random_index:
        random_inputs.append(data[i][0])  # target
        random_labels.append(np.eye(vocab_size)[data[i][1]])  # context word

    return random_inputs, random_labels

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sentences = [ "i like dog", "i like cat", "i like animal",
              "dog cat animal", "apple cat dog like", "dog fish milk like",
              "dog cat eyes like", "i like apple", "apple i hate",
              "apple i movie book music like", "cat dog hate", "cat dog like"]
    
    word_sequence = " ".join(sentences).split()
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
        
    batch_size = 20
    embedding_size = 2
    vocab_size = len(word_list)

    # ある単語wの一つ前と一つ後の単語のIDを配列に格納
    skip_grams = []
    for i in range(1, len(word_sequence) - 1):
        target = word_dict[word_sequence[i]]
        context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]
        for w in context:
            skip_grams.append([target, w])


    model = Word2Vec(vocab_size, embedding_size).to(device)
    print(model)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(10000):
        input_batch, target_batch = random_batch(skip_grams, batch_size, vocab_size)
        
        input_batch = Variable(torch.tensor(input_batch)).to(device)
        target_batch = Variable(torch.tensor(target_batch)).type(torch.FloatTensor).to(device)

        optimizer.zero_grad()
        # output = model(input_batch)
        loss = 0
        for input, target in zip(input_batch, target_batch):
            output = model(input)
            loss += criterion(output, target)

        if (epoch + 1)%1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
            torch.save(deepcopy(model).cpu().state_dict(), 'model_data/model'+str(epoch)+'.pth')

        loss.backward()
        optimizer.step()





