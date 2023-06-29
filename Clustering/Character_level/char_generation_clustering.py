import torch
import torch.nn as nn
import torch.nn.functional as F

class LANG_LSTM(nn.Module):
      def __init__(self, window_size, num_chars, num_langs, char_emb_size, lang_emb_size, hidden_size, num_layers, bidirectional, dropout):
        super(LANG_LSTM, self).__init__()
        self.char_embedding = nn.Embedding(num_chars, char_emb_size)
        self.lang_embedding = nn.Embedding(num_langs, lang_emb_size)
        self.lstm = nn.LSTM(char_emb_size + lang_emb_size, hidden_size, num_layers, batch_first = True, dropout = dropout, bidirectional = bidirectional)
        self.linear_1 = nn.Linear(2 * num_layers * window_size * hidden_size, hidden_size)       
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_size, num_chars)

      def forward(self, X_in, X_lang):
        char_embed = self.char_embedding(X_in)
        lang_embed = self.lang_embedding(X_lang)
        n_size = char_embed.shape[1]
        lang_embed = lang_embed.unsqueeze(1).repeat(1, n_size, 1)
        X = torch.cat((char_embed, lang_embed), dim = -1)
        out, _ = self.lstm(X)
        out = torch.flatten(out, 1,2)
        out = self.linear_1(out)
        out = self.relu(out)
        out = self.linear_2(out)
        out = F.log_softmax(out, dim = 1)
        return out