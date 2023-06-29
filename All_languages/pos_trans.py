import pandas as pd
from ast import literal_eval
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, XLMRobertaConfig,  XLMRobertaModel
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)



#load data and take a small portion for setting up architecture
df_train = pd.read_csv("en_lines-ud-train.csv", header = None)
df_train.columns = ['Sentence', 'PoS']
df_train.Sentence = df_train.Sentence.apply(literal_eval)
df_train.PoS = df_train.PoS.apply(literal_eval)
df_train.Sentence = df_train.Sentence.apply(lambda x: " ".join(x))
#df_train = df_train.head(500)

df_dev = pd.read_csv("en_lines-ud-dev.csv", header = None)
df_dev.columns = ['Sentence', 'PoS']
df_dev.Sentence = df_dev.Sentence.apply(literal_eval)
df_dev.PoS = df_dev.PoS.apply(literal_eval)
df_dev.Sentence = df_dev.Sentence.apply(lambda x: " ".join(x))
#df_dev = df_dev.head(100)

df_test = pd.read_csv("en_lines-ud-test.csv", header = None)
df_test.columns = ['Sentence', 'PoS']
df_test.Sentence = df_test.Sentence.apply(literal_eval)
df_test.PoS = df_test.PoS.apply(literal_eval)
df_test.Sentence = df_test.Sentence.apply(lambda x: " ".join(x))
#df_test = df_test.head(100)


tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')


#tokenize the PoS

label2id = {
    'BOS' : 0,
    'EOS': 1,
}

def create_POS_ids(x):
    for label in x:
        if label not in label2id:
            label2id[label] = len(label2id)
            
df_train['PoS'].apply(lambda x: create_POS_ids(x))
label2id



tokenized_feature_raw = tokenizer.batch_encode_plus(df_train.Sentence.values.tolist(), 
                                                    add_special_tokens = True      
                                                   ) 

token_sentence_length = [len(x) for x in tokenized_feature_raw['input_ids']]
print('max: ', max(token_sentence_length))
print('min: ', min(token_sentence_length))

"""
plt.hist(token_sentence_length, rwidth = 0.9)
plt.xlabel('Sequence Length', fontsize = 18)
plt.ylabel('# of Samples', fontsize = 18)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.show()
"""

list_len = []
for i in df_train['Sentence'].values.tolist():
    l = i.split(" ")
    list_len.append(len(l))


MAX_LEN = max(token_sentence_length)

#pad to max len

sentences_tr = df_train.Sentence.values.tolist()

tokenized_feature_train = tokenizer.batch_encode_plus(sentences_tr, add_special_tokens = True, 
                                                padding = 'max_length',
                                                max_length = MAX_LEN, 
                                                return_attention_mask = True,
                                                return_tensors = 'pt'       
                                               )

sentences_dv = df_dev.Sentence.values.tolist()


tokenized_feature_dev = tokenizer.batch_encode_plus(sentences_dv, add_special_tokens = True, 
                                                padding = 'max_length',
                                                max_length = MAX_LEN, 
                                                return_attention_mask = True,
                                                return_tensors = 'pt'       
                                               )


sentences_tst = df_test.Sentence.values.tolist()


tokenized_feature_test = tokenizer.batch_encode_plus(sentences_tst, add_special_tokens = True, 
                                                padding = 'max_length',
                                                max_length = MAX_LEN, 
                                                return_attention_mask = True,
                                                return_tensors = 'pt'       
                                               )


#encode POS

def encode_POS(x):
    encoded_pos = [label2id['BOS']]
    for label in x:
        if label in label2id:
            encoded_pos.append(label2id[label])
        else:
            return np.nan
    encoded_pos.append(label2id['EOS'])
    while len(encoded_pos) < MAX_LEN:
        encoded_pos.append(-100)
    return torch.LongTensor(encoded_pos)

df_train['PoS'] = df_train['PoS'].apply(lambda x: encode_POS(x))
df_dev['PoS'] = df_dev['PoS'].apply(lambda x: encode_POS(x))
df_test['PoS'] = df_test['PoS'].apply(lambda x: encode_POS(x))


df_train = df_train[df_train['PoS'].notna()]
df_dev = df_dev[df_dev['PoS'].notna()]
df_test = df_test[df_test['PoS'].notna()]


gold_class_train_tensor = torch.stack(df_train.PoS.values.tolist())
gold_class_dev_tensor = torch.stack(df_dev.PoS.values.tolist())
gold_class_test_tensor = torch.stack(df_test.PoS.values.tolist())



#split training into batches
BATCH_SIZE = 24


def split_into_batches(features, golds):
    feature_input = features['input_ids']
    feature_att = features['attention_mask']
    
    
    
    #shuffle data
    permutation = torch.randperm(feature_input.size()[0])
    feature_input = feature_input[permutation]
    feature_att = feature_att[permutation]
    golds = golds[permutation]
    
    batches_features_input = []
    batches_features_att = []
    batches_gold = []
    n_batches = feature_input.size()[0] // BATCH_SIZE

    
    for i in range(0, feature_input.size()[0], BATCH_SIZE):
               
        batches_features_input.append(feature_input[i : i + BATCH_SIZE])
        batches_features_att.append(feature_att[i : i + BATCH_SIZE])
        batches_gold.append(golds[i : i + BATCH_SIZE])


    
    return batches_features_input, batches_features_att, batches_gold


batches_features_input_train, batches_features_att_train, batches_gold_train = split_into_batches(tokenized_feature_train, gold_class_train_tensor)
batches_features_input_dev, batches_features_att_dev, batches_gold_dev = split_into_batches(tokenized_feature_dev, gold_class_dev_tensor)
batches_features_input_test, batches_features_att_test, batches_gold_test = split_into_batches(tokenized_feature_test, gold_class_test_tensor)



for sent, labels in zip(batches_features_input_train, batches_gold_train) :
    for i, row in enumerate(sent):
        
        for j, ids in enumerate(tokenizer.convert_ids_to_tokens(row)):
        
            if "▁" != ids[0]:
                if labels[i,j] not in [label2id['BOS'], label2id['EOS']]:
                    labels[i, j+1:] = torch.clone(labels[i, j:-1])
                    labels[i, j] = -100

                    
for sent, labels in zip(batches_features_input_dev, batches_gold_dev) :
    for i, row in enumerate(sent):
        
        for j, ids in enumerate(tokenizer.convert_ids_to_tokens(row)):
        
            if "▁" != ids[0]:
                if labels[i,j] not in [label2id['BOS'], label2id['EOS']]:
                    labels[i, j+1:] = torch.clone(labels[i, j:-1])
                    labels[i, j] = -100


for sent, labels in zip(batches_features_input_test, batches_gold_test) :
    for i, row in enumerate(sent):
        
        for j, ids in enumerate(tokenizer.convert_ids_to_tokens(row)):
        
            if "▁" != ids[0]:
                if labels[i,j] not in [label2id['BOS'], label2id['EOS']]:
                    labels[i, j+1:] = torch.clone(labels[i, j:-1])
                    labels[i, j] = -100


configuration = XLMRobertaConfig.from_pretrained('xlm-roberta-base')
configuration.hidden_dropout_prob = 0.5
configuration.attention_probs_dropout_prob = 0.5


class ROBERTA(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout):
        super(ROBERTA, self).__init__()

        #self.gru = nn.GRU(768, hidden_size//2, bidirectional=True, batch_first=True)
        
        self.linear_1 = nn.Linear(768, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, num_classes, bias = False)
        self.relu = nn.ReLU()
        self.roberta = None
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X_in, X_att):
        out = self.roberta(input_ids = X_in, attention_mask = X_att)
        out = out.last_hidden_state
        out = self.linear_1(out)

        #out = self.gru(out)[0]
        
        out = self.relu(out)
        #out = self.dropout(out)
        out = self.linear_2(out)
        out = F.log_softmax(out, dim=1)
        return out



hidden_layer_size = 500
num_classes = len(label2id)
num_epochs = 50

model =  XLMRobertaModel.from_pretrained('xlm-roberta-base', config = configuration)
model.to(device)

roberta_pos = ROBERTA(len(label2id), hidden_layer_size, 0.25)

loss_function = nn.NLLLoss()

optimizer_1 = optim.Adadelta(roberta_pos.parameters())

roberta_pos.roberta = model

roberta_pos.to(device)


optimizer_2 = optim.Adadelta(roberta_pos.roberta.parameters(), lr = 0.001)


# to record the total loss at each epoch
epoch_losses_train = []
epoch_losses_dev = []
epoch_accuracy_train = []
epoch_accuracy_dev = []

# loop on epochs
for epoch in range(num_epochs):
    print()
    print("Epoch", epoch)
    epoch_loss = 0
    
    accuracy_all = []
    
    #roberta_pos.eval()
    
    ln = len(batches_features_input_train)
    xxx = 0
    for X_in, X_att, y in tqdm(zip(batches_features_input_train, batches_features_att_train, batches_gold_train), total=ln):
        xxx += 1
        X_in = X_in.to(device)
        X_att = X_att.to(device)
        y = y.to(device)
    
        roberta_pos.zero_grad()
        logs_probs = roberta_pos(X_in, X_att)
        logs_probs = logs_probs.transpose(1,2)
        loss = loss_function(logs_probs, y)
        epoch_loss += loss.item()
        loss.backward()

        optimizer_1.step()
        #optimizer_2.step()
        
        # prediction and accuracy on the dev set
        pred_labels = torch.argmax(logs_probs, dim=1)
       
        
        mask = y != -100
        y = y[mask]
        pred_labels = pred_labels[mask]
        
        accuracy = (torch.sum((pred_labels.to(device) == y).int()).item()) / (mask.int().sum().cpu().numpy())
        
        accuracy_all.append(accuracy * 100)


        #if xxx == 33 :
        #    break

    print("Average Loss on training set at epoch %d : %f" %(epoch, epoch_loss))
    epoch_losses_train.append(epoch_loss)
    
    print("Accuracy on training set at epoch %d : %f" %(epoch, np.mean(accuracy_all)))
    epoch_accuracy_train.append(np.mean(accuracy_all))
    
    
    with torch.no_grad():
          dev_loss_all = 0
          dev_accuracy_all = []

          for X_in_dev, X_att_dev, y_dev  in tqdm(zip(batches_features_input_dev, batches_features_att_dev, batches_gold_dev), total=len(batches_gold_dev)):
            
            X_in_dev = X_in_dev.to(device)
            X_att_dev = X_att_dev.to(device)
            y_dev = y_dev.to(device)
            # forward propagation
            log_dev_probs = roberta_pos(X_in_dev,X_att_dev)  
 

            log_dev_probs = log_dev_probs.transpose(1, 2)
    
        
            # total loss on the dev set
            dev_loss = loss_function(log_dev_probs, y_dev)
           
            dev_loss_all += dev_loss.item()  
            
            pred_dev_labels = torch.argmax(log_dev_probs, dim=1)


            dev_mask = y_dev != -100
            
            y_dev = y_dev[dev_mask]
            pred_dev_labels = pred_dev_labels[dev_mask]
            

            dev_accuracy = (pred_dev_labels.to(device) == y_dev).int().sum().item() / (dev_mask.int().sum().cpu().numpy())
    
            dev_accuracy_all.append(dev_accuracy * 100)

          print("Loss on dev set at epoch %d : %f" %(epoch, dev_loss_all))
          epoch_losses_dev.append(dev_loss_all)

     
          print("Accuracy on dev set at epoch %d : %f" %(epoch, np.mean(dev_accuracy_all)))
          epoch_accuracy_dev.append(np.mean(dev_accuracy_all))
    if epoch > 50: 
        if epoch == 0:
            dev_loss_counter = 0
            previous_dev_loss = dev_loss_all
        elif dev_loss_all/previous_dev_loss  >= 0.95:
            dev_loss_counter += 1
            previous_dev_loss = dev_loss_all
            if dev_loss_counter > 5:
                break
        else:
            previous_dev_loss = dev_loss_all
            dev_loss_counter = 0
    

      
