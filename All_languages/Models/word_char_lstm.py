import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_WORD_CHAR(nn.Module):  

    def __init__(self, vocab_size, char_size, embedding_size_char, embedding_size_word, num_classes, hidden_layer_size_char, hidden_layer_size_word, num_layers, device, dropout):
        super(LSTM_WORD_CHAR, self).__init__()
        
        self.num_layers = num_layers

        self.hidden_layer_size_char = hidden_layer_size_char
        

        self.char_embedding_layer = nn.Embedding(char_size, embedding_size_char, padding_idx = 0)
        
        self.word_embedding_layer = nn.Embedding(vocab_size, embedding_size_word, padding_idx = 0)
        
        self.lstm_char_forward = nn.LSTM(embedding_size_char, 
                            hidden_layer_size_char, 
                            num_layers = num_layers,
                            batch_first=True, 
                            dropout = dropout,
                            bidirectional=True)
        
        self.device = device
        
        
        self.lstm_char_backward = nn.LSTM(embedding_size_char, 
                            hidden_layer_size_char, 
                            num_layers = num_layers,
                            batch_first=True, 
                            dropout = dropout,
                            bidirectional=False)
        
        
        self.lstm_word_n_char = nn.LSTM( 2 * hidden_layer_size_char + embedding_size_word, 
                                            hidden_layer_size_word, 
                                            num_layers = num_layers,
                                            batch_first=True, 
                                            dropout = dropout,
                                            bidirectional=True)
        

        
        self.linear_1 = nn.Linear(2*hidden_layer_size_word, hidden_layer_size_word) # 2*hidden as using bidirectional

        self.relu = nn.ReLU()
        
        self.linear_2 = nn.Linear(hidden_layer_size_word, num_classes)
        

    def forward(self, X_word, X_char):
        
        emb_char = self.char_embedding_layer(X_char)

        batch_size, max_sent_len, max_char, char_emb_size = emb_char.shape
        emb_char = torch.reshape(emb_char, (batch_size * max_sent_len, max_char, char_emb_size))
        char_last_hidden = self.lstm_char_forward(emb_char)[0][:,-1,:]
        char_last_hidden = torch.reshape(char_last_hidden, (batch_size, max_sent_len, 2 * self.hidden_layer_size_char))

        emb_word = self.word_embedding_layer(X_word)
        emb_both = torch.cat((emb_word, char_last_hidden), -1)
                
        out, _ = self.lstm_word_n_char(emb_both)
                
        out = self.linear_1(out)
        out = self.relu(out)
        
        out = self.linear_2(out)
        
        out = F.log_softmax(out, dim = -1)
        return out
      