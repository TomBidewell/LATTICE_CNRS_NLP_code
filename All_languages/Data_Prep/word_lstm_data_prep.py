
import pandas as pd
from ast import literal_eval
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np




def word_prepared_data(train, dev, test):
        
    df_train = pd.read_csv(train, header = None)
    df_train.columns = ['Sentence', 'PoS']
    df_train.Sentence = df_train.Sentence.apply(literal_eval)
    df_train.PoS = df_train.PoS.apply(literal_eval)
    df_train = df_train.head(2500)

    df_dev = pd.read_csv(dev, header = None)
    df_dev.columns = ['Sentence', 'PoS']
    df_dev.Sentence = df_dev.Sentence.apply(literal_eval)
    df_dev.PoS = df_dev.PoS.apply(literal_eval)
    df_dev = df_dev.head(300)

    df_test = pd.read_csv(test, header = None)
    df_test.columns = ['Sentence', 'PoS']
    df_test.Sentence = df_test.Sentence.apply(literal_eval)
    df_test.PoS = df_test.PoS.apply(literal_eval)
    df_test = df_test.head(300)


    counts = {}
    def get_counts(x):
        for w in x:
            try: 
                counts[w] += 1
            except:
                counts[w] = 1

    df_train['Sentence'].apply(lambda x: get_counts(x))


    #creating indices for the vocab
    word2id = {'PAD': 0,
            'UNK' : 1,
            }

    label2id = {'PAD': 0,
            'UNK' : 1,
            }

    def create_word_ids(x):
        for token in x:
            token = token
            if token not in word2id:
                if counts[token] == 1:
                    word2id[token] = word2id['UNK']
                else:
                    word2id[token] = len(word2id)

    def create_label_ids(x):
        for label in x:
            if label not in label2id:
                label2id[label] = len(label2id)

    df_train['Sentence'].apply(lambda x: create_word_ids(x))
    df_train['PoS'].apply(lambda x: create_label_ids(x))

    

    #encoding sentences and PoS tags

    def encoding(x, y):
        encoding_sent = []

        encoding_tags = []

        #encoding_sent.append(word2id['BOS']) #as its beginning of sentence
        #encoding_tags.append(word2id['BOS']) #as its beginning of sentence

        for word, tag in zip(x,y):
                
            word = word
            if word in word2id:
                encoding_sent.append(word2id[word])
                if tag in label2id:
                    encoding_tags.append(label2id[tag])
                else:
                    encoding_tags.append(label2id['UNK'])
            else: 
                encoding_sent.append(word2id['UNK'])
                if tag in label2id:
                    encoding_tags.append(label2id[tag])
                else:
                    encoding_tags.append(label2id['UNK'])

            
        #encoding_sent.append(word2id['EOS']) #as end of sentence
        #encoding_tags.append(label2id['EOS'])

        return encoding_sent, encoding_tags




    df_train[['Encoded_Sentence', 'Encoded_PoS']] = df_train.apply(lambda x: encoding(x.Sentence, x.PoS), axis = 1, result_type="expand")

    df_dev[['Encoded_Sentence', 'Encoded_PoS']] = df_dev.apply(lambda x: encoding(x.Sentence, x.PoS), axis = 1, result_type="expand")

    df_test[['Encoded_Sentence', 'Encoded_PoS']] = df_test.apply(lambda x: encoding(x.Sentence, x.PoS), axis = 1, result_type="expand")


    seq_len = []

    def find_len(x):
        seq_len.append(len(x))

    df_train['Encoded_Sentence'].apply(lambda x: find_len(x))
    df_dev['Encoded_Sentence'].apply(lambda x: find_len(x))
    df_test['Encoded_Sentence'].apply(lambda x: find_len(x))

    df_train['Encoded_PoS'].apply(lambda x: find_len(x))
    df_dev['Encoded_PoS'].apply(lambda x: find_len(x))
    df_test['Encoded_PoS'].apply(lambda x: find_len(x))

        
    max_len = max(seq_len)


    def padding_sent(x):
        if len(x) < 50:
            upper_bound = (np.floor(len(x)/10) + 1)*10
            while len(x) < upper_bound:
                x.append(0)
            return x
        else: 
            
            while len(x) < max_len:
                x.append(0)
            return x
        
    def padding_pos(x):
        if len(x) < 50:
            upper_bound = (np.floor(len(x)/10) + 1)*10
            while len(x) < upper_bound:
                x.append(-100)
            return x
        else: 
            
            while len(x) < max_len:
                x.append(-100)
            return x
            
        

    df_train['Encoded_Sentence'] = df_train['Encoded_Sentence'].apply(lambda x: padding_sent(x))
    df_train['Encoded_PoS'] = df_train['Encoded_PoS'].apply(lambda x: padding_pos(x))

    df_dev['Encoded_Sentence'] = df_dev['Encoded_Sentence'].apply(lambda x: padding_sent(x))
    df_dev['Encoded_PoS'] = df_dev['Encoded_PoS'].apply(lambda x: padding_pos(x))

    df_test['Encoded_Sentence'] = df_test['Encoded_Sentence'].apply(lambda x: padding_sent(x))
    df_test['Encoded_PoS'] = df_test['Encoded_PoS'].apply(lambda x: padding_pos(x))


    #sanity check train
    seq_len_sent = []

    def find_len_sent(x):
        seq_len_sent.append(len(x))

    seq_len_tag = []

    def find_len_tag(x):
        seq_len_tag.append(len(x))

    df_train['Encoded_Sentence'].apply(lambda x: find_len_sent(x))
    df_train['Encoded_PoS'].apply(lambda x: find_len_tag(x))

    #get inputs and gold classes

    def convert2tensors(df):
        input_data = []
        gold_class_data = []
        
        len_dic = {}
        for row in df.itertuples():

            #row[3] = sentences, row[4] = tags
            if str(len(row[3])) not in len_dic:
                len_dic[str(len(row[3]))] = [(row[3], row[4])]
            else:
                len_dic[str(len(row[3]))].append((row[3], row[4]))
        
        
        for key in len_dic.keys():
            batch_input_data = []
            batch_gold_class_data = []
            
            list_of_batches = len_dic[key]

            for item in list_of_batches: 

                batch_input_data.append(torch.tensor(item[0]))
                batch_gold_class_data.append(torch.tensor(item[1]))
            
            k = 33

            for i in range(0, len(batch_input_data), 33):
            
                tensor_batch_input = torch.stack(batch_input_data[i: i + k])  
                tensor_batch_gold = torch.stack(batch_gold_class_data[i: i + k])
            
                input_data.append(tensor_batch_input)
                gold_class_data.append(tensor_batch_gold)
        
        return input_data, gold_class_data
                
        
    train_input, train_gold = convert2tensors(df_train)
    dev_input, dev_gold = convert2tensors(df_dev)
    test_input, test_gold = convert2tensors(df_test)

    tensor_dict = {
        'train'  : [train_input, train_gold],
        'dev': [dev_input, dev_gold],
        'test': [test_input, test_gold]
    }
    return tensor_dict, len(word2id), len(label2id)
