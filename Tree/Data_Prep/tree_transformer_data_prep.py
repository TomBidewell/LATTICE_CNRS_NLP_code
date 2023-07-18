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


def transformer_data_prep(train, dev, test, label2id):

    #load data and take a small portion for setting up architecture
    df_train = pd.read_csv(train, header = None)
    df_train.columns = ['Sentence', 'PoS']
    df_train.Sentence = df_train.Sentence.apply(literal_eval)
    df_train.PoS = df_train.PoS.apply(literal_eval)

    df_dev = pd.read_csv(dev, header = None)
    df_dev.columns = ['Sentence', 'PoS']
    df_dev.Sentence = df_dev.Sentence.apply(literal_eval)
    df_dev.PoS = df_dev.PoS.apply(literal_eval)

    df_test = pd.read_csv(test, header = None)
    df_test.columns = ['Sentence', 'PoS']
    df_test.Sentence = df_test.Sentence.apply(literal_eval)
    df_test.PoS = df_test.PoS.apply(literal_eval)


    def convert_dataframe(df):
        df_new = pd.DataFrame(columns=['Sentence', 'PoS'])

        for idx in df.index:
            for i, j in zip(df['Sentence'][idx], df['PoS'][idx]):
                new_row = {'Sentence': [i], 'PoS': [j]}
                df_new = pd.concat([df_new, pd.DataFrame.from_dict(new_row)])
        
        return df_new
    
    df_train = convert_dataframe(df_train)
    df_dev = convert_dataframe(df_dev)
    df_test = convert_dataframe(df_test)


    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')


    #padding to max of train dev and test

    MAX_LEN = 512

    #pad to max len


    sentences_tr = [i[0] for i in [[" ".join(sentence)] for sentence in df_train.Sentence.values.tolist()]]


    tokenized_feature_train = tokenizer.batch_encode_plus(sentences_tr, add_special_tokens = True, 
                                                    padding = 'max_length',
                                                    truncation = True,
                                                    max_length = MAX_LEN, 
                                                    return_attention_mask = True,
                                                    return_tensors = 'pt'       
                                                )



    sentences_dv = [i[0] for i in [[" ".join(sentence)] for sentence in df_dev.Sentence.values.tolist()]]


    tokenized_feature_dev = tokenizer.batch_encode_plus(sentences_dv, add_special_tokens = True, 
                                                    padding = 'max_length',
                                                    truncation = True,
                                                    max_length = MAX_LEN, 
                                                    return_attention_mask = True,
                                                    return_tensors = 'pt'       
                                                )


    sentences_tst = [i[0] for i in [[" ".join(sentence)] for sentence in df_test.Sentence.values.tolist()]]


    tokenized_feature_test = tokenizer.batch_encode_plus(sentences_tst, add_special_tokens = True, 
                                                    padding = 'max_length',
                                                    truncation = True,
                                                    max_length = MAX_LEN, 
                                                    return_attention_mask = True,
                                                    return_tensors = 'pt'       
                                                )
    

    #encode POS

    
    def encode_POS(x):
        all_encoded_pos = []
        for sentence in x:
            encoded_pos = []
            for label in sentence:
                if label in label2id:
                    encoded_pos.append(label2id[label])
                else:
                    encoded_pos.append(label2id['UNK']) 
            while len(encoded_pos) < MAX_LEN:
                encoded_pos.append(-100)
            all_encoded_pos.append(encoded_pos)

        return torch.LongTensor(all_encoded_pos)

    df_train['PoS'] = df_train['PoS'].apply(lambda x: encode_POS(x))
    df_dev['PoS'] = df_dev['PoS'].apply(lambda x: encode_POS(x))
    df_test['PoS'] = df_test['PoS'].apply(lambda x: encode_POS(x))

  

    #df_train = df_train[df_train['PoS'].notna()]
    #df_dev = df_dev[df_dev['PoS'].notna()]
    #df_test = df_test[df_test['PoS'].notna()]
    

    gold_class_train_tensor = torch.cat(df_train.PoS.values.tolist(), dim = 0)
    gold_class_dev_tensor = torch.cat(df_dev.PoS.values.tolist(), dim = 0)
    gold_class_test_tensor = torch.cat(df_test.PoS.values.tolist(), dim = 0)

  


    #split training into batches
    BATCH_SIZE = 50


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

 


    for sent, labels in tqdm(zip(batches_features_input_train, batches_gold_train), total = len(batches_features_input_train), desc = 'Preparing Training Data: '):
        for i, row in enumerate(sent):
            
            for j, ids in enumerate(tokenizer.convert_ids_to_tokens(row)):
            
                if "▁" != ids[0]:
                    labels[i, j+1:] = torch.clone(labels[i, j:-1])
                    labels[i, j] = -100

                        
    for sent, labels in tqdm(zip(batches_features_input_dev, batches_gold_dev), total = len(batches_features_input_dev), desc = 'Preparing Dev Data: '):
        for i, row in enumerate(sent):
            
            for j, ids in enumerate(tokenizer.convert_ids_to_tokens(row)):
            
                if "▁" != ids[0]:
                    labels[i, j+1:] = torch.clone(labels[i, j:-1])
                    labels[i, j] = -100


    for sent, labels in tqdm(zip(batches_features_input_test, batches_gold_test), total = len(batches_features_input_test), desc= "Preparing Test Data: "):
        for i, row in enumerate(sent):
            
            for j, ids in enumerate(tokenizer.convert_ids_to_tokens(row)):
            
                if "▁" != ids[0]:
                    labels[i, j+1:] = torch.clone(labels[i, j:-1])
                    labels[i, j] = -100

    tensor_dict = {
        'train' :  [batches_features_input_train, batches_features_att_train, batches_gold_train],
        'dev' : [batches_features_input_dev, batches_features_att_dev, batches_gold_dev],
        'test' : [batches_features_input_test, batches_features_att_test, batches_gold_test ]
    }



    return tensor_dict, len(label2id)

