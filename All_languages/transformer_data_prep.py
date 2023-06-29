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


def transformer_data_prep(train, dev, test):

    #load data and take a small portion for setting up architecture
    df_train = pd.read_csv(train, header = None)
    df_train.columns = ['Sentence', 'PoS']
    df_train.Sentence = df_train.Sentence.apply(literal_eval)
    df_train.PoS = df_train.PoS.apply(literal_eval)
    df_train.Sentence = df_train.Sentence.apply(lambda x: " ".join(x))
    #df_train = df_train.head(500)

    df_dev = pd.read_csv(dev, header = None)
    df_dev.columns = ['Sentence', 'PoS']
    df_dev.Sentence = df_dev.Sentence.apply(literal_eval)
    df_dev.PoS = df_dev.PoS.apply(literal_eval)
    df_dev.Sentence = df_dev.Sentence.apply(lambda x: " ".join(x))
    #df_dev = df_dev.head(100)

    df_test = pd.read_csv(test, header = None)
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

    #padding to max of train dev and test


    tokenized_feature_raw_tr = tokenizer.batch_encode_plus( df_train.Sentence.values.tolist(),
                                                           truncation = True, 
                                                           max_length = 512,
                                                        add_special_tokens = True      
                                                    ) 

    token_sentence_length_tr = [len(x) for x in tokenized_feature_raw_tr['input_ids']]

    MAX_LEN_tr = max(token_sentence_length_tr)

    tokenized_feature_raw_dev = tokenizer.batch_encode_plus( df_dev.Sentence.values.tolist(),
                                                            truncation = True, 
                                                           max_length = 512,
                                                        add_special_tokens = True      
                                                    ) 

    token_sentence_length_dev = [len(x) for x in tokenized_feature_raw_dev['input_ids']]

    MAX_LEN_dv = max(token_sentence_length_dev)

    tokenized_feature_raw_test = tokenizer.batch_encode_plus( df_test.Sentence.values.tolist(),
                                                             truncation = True, 
                                                           max_length = 512,
                                                        add_special_tokens = True      
                                                    ) 

    token_sentence_length_test = [len(x) for x in tokenized_feature_raw_test['input_ids']]


    MAX_LEN_tst = max(token_sentence_length_test)

    MAX_LEN = max(MAX_LEN_tr, MAX_LEN_dv, MAX_LEN_tst)

    if MAX_LEN > 512:
        MAX_LEN = 512



    #pad to max len

    sentences_tr = df_train.Sentence.values.tolist()

    tokenized_feature_train = tokenizer.batch_encode_plus(sentences_tr, add_special_tokens = True, 
                                                    padding = 'max_length',
                                                    truncation = True,
                                                    max_length = MAX_LEN, 
                                                    return_attention_mask = True,
                                                    return_tensors = 'pt'       
                                                )

    sentences_dv = df_dev.Sentence.values.tolist()


    tokenized_feature_dev = tokenizer.batch_encode_plus(sentences_dv, add_special_tokens = True, 
                                                    padding = 'max_length',
                                                    truncation = True,
                                                    max_length = MAX_LEN, 
                                                    return_attention_mask = True,
                                                    return_tensors = 'pt'       
                                                )


    sentences_tst = df_test.Sentence.values.tolist()


    tokenized_feature_test = tokenizer.batch_encode_plus(sentences_tst, add_special_tokens = True, 
                                                    padding = 'max_length',
                                                    truncation = True,
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

    tensor_dict = {
        'train' :  [batches_features_input_train, batches_features_att_train, batches_gold_train],
        'dev' : [batches_features_input_dev, batches_features_att_dev, batches_gold_dev],
        'test' : [batches_features_input_test, batches_features_att_test, batches_gold_test ]
    }

    return tensor_dict, len(label2id)

