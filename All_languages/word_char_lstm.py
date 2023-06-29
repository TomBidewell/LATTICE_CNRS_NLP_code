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
                            bidirectional=False)
        
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
        new_size = list(X_char.shape)
        new_size.pop()
        new_size.append(2  * self.hidden_layer_size_char)  # 2 as forward and backward over characters, num_layers as concatenating these layers, layers have hidden_layer_size
        
        all_sentences = torch.zeros(new_size)
        
        #emb_char = emb_char.view(emb_char.shape[0], emb_char.shape[1], (emb_char.shape[2]*emb_char.shape[3])) #concat embeddings
        
        
        for i, sentence in enumerate(X_char):

            emb_char = self.char_embedding_layer(sentence)
            out_char_forward, _ = self.lstm_char_forward(emb_char)
            out_char_backward, _ = self.lstm_char_backward(emb_char)
            out_char_forward = out_char_forward[:, -1, :]
            out_char_backward = out_char_backward[:, -1, :]
            out_char = torch.cat((out_char_forward, out_char_backward), dim = -1)
            all_sentences[i, :, :] = out_char

        all_sentences = all_sentences.to(self.device)
                  
        emb_word = self.word_embedding_layer(X_word)
        
        
        emb_both = torch.cat((emb_word, all_sentences), -1)
                
        out, _ = self.lstm_word_n_char(emb_both)
                
        out = self.linear_1(out)
        out = self.relu(out)
        
        out = self.linear_2(out)
        
        out = F.log_softmax(out, dim = -1)
        return out
      