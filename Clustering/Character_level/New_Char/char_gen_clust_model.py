import torch 
import torch.nn as nn
import torch.nn.functional as F

class NEXTCHAR(nn.Module):

    def __init__(self, num_chars, char_embedding_size, num_langs, lang_embedding_size, hidden_size, num_layers):
        super(NEXTCHAR, self).__init__()

        self.char_emb_layer = nn.Embedding(num_chars, char_embedding_size)
        self.lang_emb_layer = nn.Embedding(num_langs, lang_embedding_size)

        self.lstm = nn.LSTM(char_embedding_size + lang_embedding_size, 
                            hidden_size, 
                            num_layers,
                            batch_first = True, 
                            )
        
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_size, num_chars)

    def forward(self, X_chars, X_langs):
        char_embs = self.char_emb_layer(X_chars)
        lang_embs = self.lang_emb_layer(X_langs)
        n = char_embs.shape[1]
        lang_embs = lang_embs.unsqueeze(1).repeat((1, n, 1))
        out = torch.cat((char_embs, lang_embs), dim = -1)
        out, _ = self.lstm(out)
        out = self.relu(self.linear_1(out))
        out = self.linear_2(out)
        return F.log_softmax(out, dim = -1)