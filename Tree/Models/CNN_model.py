import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNN(nn.Module):
    def __init__(self, len_word, embedding_size, num_classes, hidden_size, dropout):
        super(CNN, self).__init__()

        self.embedding_layer = nn.Embedding(len_word, embedding_size)
        self.conv1 = nn.Conv1d(embedding_size, hidden_size, 5, padding = 2)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, 5, padding = 2)
        self.relu = nn.ReLU()
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

     
        
    def forward(self, X):
        emb_X = self.embedding_layer(X).transpose(1,2)
        out = self.relu(self.conv1(emb_X))
        out = self.relu(self.conv2(out))
        out = out.transpose(1,2)
        out = self.dropout(self.relu(self.linear_1(out)))
        out = self.linear_2(out)
        return F.log_softmax(out, dim= -1)
