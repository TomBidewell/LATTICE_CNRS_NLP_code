import torch.nn as nn
import torch.nn.functional as F


class WORD_LSTM(nn.Module):  

        def __init__(self, vocab_size, embedding_size, num_classes, hidden_layer_size, num_layers, dropout, batch_first = True, bidirectional=False):
            super(WORD_LSTM, self).__init__()

            self.embedding_layer = nn.Embedding(vocab_size, embedding_size, padding_idx = 0)
            
            self.lstm = nn.LSTM( embedding_size, 
                                hidden_layer_size, 
                                num_layers = num_layers,
                                batch_first=True, 
                                dropout = dropout,
                                bidirectional=True)
            
            if bidirectional:
                self.linear_1 = nn.Linear(2*hidden_layer_size, hidden_layer_size) # 2*hidden as using bidirectional
            else:
                self.linear_1 = nn.Linear(hidden_layer_size, hidden_layer_size)

            self.relu = nn.ReLU()
            
            self.linear_2 = nn.Linear(hidden_layer_size, num_classes)
            

        def forward(self, X):
            emb = self.embedding_layer(X)

                    
            #nn.lstm initialises hiddenstate to 0 tensors         
            out, _ = self.lstm(emb)

            
            out = self.linear_1(out)
            out = self.relu(out)
            
            out = self.linear_2(out)
            
            out = F.log_softmax(out, dim = -1)
            return out