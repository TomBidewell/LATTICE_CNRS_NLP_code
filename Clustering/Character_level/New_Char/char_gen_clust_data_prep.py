import pandas as pd
from ast import literal_eval
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random



def prepared_data(language, train, dev, test):
    #load data and take a small portion for setting up architecture
    df_train = pd.read_csv(train, header = None)
    df_train.columns = ['Sentence', 'PoS']
    df_train.Sentence = df_train.Sentence.apply(literal_eval)
    df_train = df_train.Sentence
    df_train = pd.DataFrame(df_train)
    df_train['Language'] = language
    df_train = df_train.head(10)

    df_dev = pd.read_csv(dev, header = None)
    df_dev.columns = ['Sentence', 'PoS']
    df_dev.Sentence = df_dev.Sentence.apply(literal_eval)
    df_dev = df_dev.Sentence
    df_dev = pd.DataFrame(df_dev)
    df_dev['Language'] = language
    df_dev = df_dev.head(5)

    df_test = pd.read_csv(test, header = None)
    df_test.columns = ['Sentence', 'PoS']
    df_test.Sentence = df_test.Sentence.apply(literal_eval)
    df_test = df_test.Sentence
    df_test = pd.DataFrame(df_test)
    df_test['Language'] = language
    df_test = df_test.head(5)

    def combine(x):
        return " ".join(x).strip()
    
    df_train['Sentence'] = df_train['Sentence'].apply(lambda x: combine(x))
    df_dev['Sentence'] = df_dev['Sentence'].apply(lambda x: combine(x))
    df_test['Sentence'] = df_test['Sentence'].apply(lambda x: combine(x))

    return df_train, df_dev, df_test












def prepared_all_data(df):
    #creating indices for the vocab
    char2id = { 'BOS': 0,
            'UNK': 1,
            'PAD': 2,
            }

    def create_char_ids(x):
        for token in x:
            token = token
            for char in token:
                if char not in char2id:
                    char2id[char] = len(char2id)

    df['Sentence'].apply(lambda x: create_char_ids(x))

    #encoding sentences and PoS tags

    def encoding(x):
        encoding_sent_char = []
                
        encoding_sent_char.append(char2id['BOS']) 
        
        for word in x:
            for char in word:
                if char in char2id:
                    encoding_sent_char.append(char2id[char])
                else:
                    encoding_sent_char.append(char2id['UNK'])

        return encoding_sent_char

    df['Encoded_Sentence_Char'] = df['Sentence'].apply(lambda x: encoding(x))

  
    seq_len = []

    def find_len(x):
        seq_len.append(len(x))

    df['Encoded_Sentence_Char'].apply(lambda x: find_len(x))

        
    max_len = max(seq_len)

    #pad word and tags    
        
    def padding_sent_n_pos(x):
        if len(x) < 50:
            upper_bound = (np.floor(len(x)/10) + 1)*10
            while len(x) < upper_bound:
                x.append(0)           
            return x
        else: 
            while len(x) < max_len:
                x.append(0)
            return x 
            
            
    df['Encoded_Sentence_Char'] = df['Encoded_Sentence_Char'].apply(lambda x: padding_sent_n_pos(x))


    lang2id = {'UNK': 0,}

    def create_lang_ids(x):
        if x not in lang2id:
            lang2id[x] = len(lang2id)

    def encode_langs(x):
        if x in lang2id:
            return lang2id[x]
        else:
            return lang2id['UNK']

    df['Language'] = df['Language'].apply(lambda x: create_lang_ids(x))
    df['Language'] = df['Language'].apply(lambda x: encode_langs(x))

    #get inputs and gold classes
    def convert2tensors(df):
        input_data_char = []
        input_data_lang = []
        
        len_dic = {}


        for idx, row in df.iterrows():

            if str(len(row['Encoded_Sentence_Char'])) not in len_dic:
                len_dic[str(len(row['Encoded_Sentence_Char']))] = [(row['Encoded_Sentence_Char'], row['Language'])]
            else:
                len_dic[str(len(row['Encoded_Sentence_Char']))].append((row['Encoded_Sentence_Char'], row['Language']))
        
        
        for key in len_dic.keys():
            batch_input_data_char = []
            batch_input_lang = []
            
            list_of_batches = len_dic[key]

            for item in list_of_batches: 
                
                batch_input_data_char.append(torch.tensor(item[0]))
                batch_input_lang.append(torch.tensor(item[1]).long())
                
            tensor_batch_input_data_char = torch.stack(batch_input_data_char)  
            tensor_lang_batch = torch.stack(batch_input_lang)
            
            input_data_char.append(tensor_batch_input_data_char)
            input_data_lang.append(tensor_lang_batch)
            #put to device here
        return input_data_char, input_data_lang
                        
    train_input_char, input_data_lang = convert2tensors(df)

    return train_input_char, input_data_lang, char2id, lang2id











def prepared_data_other(df, char2id, lang2id):

    #encoding sentences and PoS tags

    def encoding(x):
        encoding_sent_char = []
                
        encoding_sent_char.append(char2id['BOS']) 
        
        for word in x:
            for char in word:
                if char in char2id:
                    encoding_sent_char.append(char2id[char])
                else:
                    encoding_sent_char.append(char2id['UNK'])

        return encoding_sent_char

    df['Encoded_Sentence_Char'] = df['Sentence'].apply(lambda x: encoding(x))

  
    seq_len = []

    def find_len(x):
        seq_len.append(len(x))

    df['Encoded_Sentence_Char'].apply(lambda x: find_len(x))

        
    max_len = max(seq_len)

    #pad word and tags    
        
    def padding_sent_n_pos(x):
        if len(x) < 50:
            upper_bound = (np.floor(len(x)/10) + 1)*10
            while len(x) < upper_bound:
                x.append(0)           
            return x
        else: 
            while len(x) < max_len:
                x.append(0)
            return x 
            
            
    df['Encoded_Sentence_Char'] = df['Encoded_Sentence_Char'].apply(lambda x: padding_sent_n_pos(x))


    def encode_langs(x):
        if x in lang2id:
            return lang2id[x]
        else:
            return lang2id['UNK']

    df['Language'] = df['Language'].apply(lambda x: encode_langs(x))

    #get inputs and gold classes
    def convert2tensors(df):
        input_data_char = []
        input_data_lang = []
        
        len_dic = {}


        for idx, row in df.iterrows():

            if str(len(row['Encoded_Sentence_Char'])) not in len_dic:
                len_dic[str(len(row['Encoded_Sentence_Char']))] = [(row['Encoded_Sentence_Char'], row['Language'])]
            else:
                len_dic[str(len(row['Encoded_Sentence_Char']))].append((row['Encoded_Sentence_Char'], row['Language']))
        
        
        for key in len_dic.keys():
            batch_input_data_char = []
            batch_input_lang = []
            
            list_of_batches = len_dic[key]

            for item in list_of_batches: 
                
                batch_input_data_char.append(torch.tensor(item[0]))
                batch_input_lang.append(torch.tensor(item[1]).long())
                
            tensor_batch_input_data_char = torch.stack(batch_input_data_char)  
            tensor_lang_batch = torch.stack(batch_input_lang)
            
            input_data_char.append(tensor_batch_input_data_char)
            input_data_lang.append(tensor_lang_batch)
            #put to device here
        return input_data_char, input_data_lang
                
                
        
    train_input_char, input_data_lang = convert2tensors(df)

    return train_input_char, input_data_lang


