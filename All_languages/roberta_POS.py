import torch
import torch.nn as nn
import torch.nn.functional as F

class ROBERTA(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super(ROBERTA, self).__init__()

        #self.gru = nn.GRU(768, hidden_size//2, bidirectional=True, batch_first=True)
        
        self.linear_1 = nn.Linear(768, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, num_classes, bias = False)
        self.relu = nn.ReLU()
        self.roberta = None
        
    def forward(self, X_in, X_att):
        out = self.roberta(input_ids = X_in, attention_mask = X_att)
        out = out.last_hidden_state
        out = self.linear_1(out)
        
        out = self.relu(out)
        out = self.linear_2(out)
        out = F.log_softmax(out, dim=-1)
        return out