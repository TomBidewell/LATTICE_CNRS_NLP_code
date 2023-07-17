import pandas as pd
from ast import literal_eval
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random



def word_char_prepared_data(train, dev, test, word2id, char2id, label2id):
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


    #encoding sentences and PoS tags

    def encoding(x, y):
        encoding_sent_char = []
        
        encoding_sent_word = []

        encoding_tags = []
        
        #encoding_sent_word.append(word2id['BOS'])
        #encoding_sent_char.append([char2id['BOS']]) #as its beginning of sentence
        #encoding_tags.append(label2id['BOS'])
        
        for word, tag in zip(x,y):
                
            word = word
            if word in word2id:
                word_char = []
                
                #word_char.append(char2id['BOW'])

                encoding_sent_word.append(word2id[word])
                
                if tag in label2id:
                    encoding_tags.append(label2id[tag])
                else:
                    encoding_tags.append(label2id['UNK'])

                for char in word:
                    if char in char2id:
                        if random.random() < 0.999:
                           word_char.append(char2id[char])
                        else:
                            word_char.append(char2id['UNK'])
                            
                    else:
                        word_char.append(char2id['UNK'])
                
                #word_char.append(char2id['EOW'])
                encoding_sent_char.append(word_char)
            else:
                word_char = []
                
                #word_char.append(char2id['BOW'])

                encoding_sent_word.append(word2id['UNK'])
                
                if tag in label2id:
                    encoding_tags.append(label2id[tag])
                else:
                    encoding_tags.append(label2id['UNK'])

                for char in word:
                    if char in char2id:
                        if random.random() < 0.999:
                           word_char.append(char2id[char])
                        else:
                            word_char.append(char2id['UNK'])
                    else:
                        word_char.append(char2id['UNK'])
                
                #word_char.append(char2id['EOW'])
                encoding_sent_char.append(word_char)
                
        
        return encoding_sent_word, encoding_sent_char, encoding_tags

    df_train[['Encoded_Sentence_Word', 'Encoded_Sentence_Char','Encoded_PoS']] = df_train.apply(lambda x: encoding(x.Sentence, x.PoS), axis = 1, result_type="expand")

    df_dev[['Encoded_Sentence_Word', 'Encoded_Sentence_Char','Encoded_PoS']] = df_dev.apply(lambda x: encoding(x.Sentence, x.PoS), axis = 1, result_type="expand")

    df_test[['Encoded_Sentence_Word', 'Encoded_Sentence_Char','Encoded_PoS']] = df_test.apply(lambda x: encoding(x.Sentence, x.PoS), axis = 1, result_type="expand")


    seq_len = []

    def find_len(x):
        seq_len.append(len(x))

    df_train['Encoded_Sentence_Word'].apply(lambda x: find_len(x))
    df_dev['Encoded_Sentence_Word'].apply(lambda x: find_len(x))
    df_test['Encoded_Sentence_Word'].apply(lambda x: find_len(x))
    df_train['Encoded_Sentence_Char'].apply(lambda x: find_len(x))
    df_dev['Encoded_Sentence_Char'].apply(lambda x: find_len(x))
    df_test['Encoded_Sentence_Char'].apply(lambda x: find_len(x))
    df_train['Encoded_PoS'].apply(lambda x: find_len(x))
    df_dev['Encoded_PoS'].apply(lambda x: find_len(x))
    df_test['Encoded_PoS'].apply(lambda x: find_len(x))

        
    max_len = max(seq_len)

    #pad word and tags    
        
    def padding_sent_n_pos(x,y,z):
        if len(x) < 50:
            upper_bound = (np.floor(len(x)/10) + 1)*10
            while len(x) < upper_bound:
                x.append(0)
                
            while len(y) < upper_bound:
                y.append([0])
            
            while len(z) < upper_bound:
                z.append(-100)
                
            return x, y, z
        else: 
            while len(x) < max_len:
                x.append(0)
            
            while len(y) < max_len:
                y.append([0])
                

            while len(z) < max_len:
                z.append(-100)
                
            return x, y, z 
            
            
    df_train[['Encoded_Sentence_Word', 'Encoded_Sentence_Char', 'Encoded_PoS']] = df_train.apply(lambda x: padding_sent_n_pos(x.Encoded_Sentence_Word, x.Encoded_Sentence_Char,x.Encoded_PoS), axis =1, result_type="expand")
    df_dev[['Encoded_Sentence_Word', 'Encoded_Sentence_Char','Encoded_PoS']] = df_dev.apply(lambda x: padding_sent_n_pos(x.Encoded_Sentence_Word, x.Encoded_Sentence_Char,x.Encoded_PoS), axis =1, result_type="expand")
    df_test[['Encoded_Sentence_Word', 'Encoded_Sentence_Char','Encoded_PoS']] = df_test.apply(lambda x: padding_sent_n_pos(x.Encoded_Sentence_Word, x.Encoded_Sentence_Char,x.Encoded_PoS), axis =1, result_type="expand")

    max_len_char = []

    def find_char_len(x):
        for word in x: 
            max_len_char.append(len(word))
            
    df_train['Encoded_Sentence_Char'].apply(lambda x: find_char_len(x))
    df_dev['Encoded_Sentence_Char'].apply(lambda x: find_char_len(x))
    df_test['Encoded_Sentence_Char'].apply(lambda x: find_char_len(x))


    max_char_len = max(max_len_char)

    def padding_words(x):
        for word in x:
            while len(word) < max_char_len:
                word.append(char2id['PAD'])

    df_train['Encoded_Sentence_Char'].apply(lambda x: padding_words(x))
    df_dev['Encoded_Sentence_Char'].apply(lambda x: padding_words(x))
    df_test['Encoded_Sentence_Char'].apply(lambda x: padding_words(x))


    #get inputs and gold classes
    def convert2tensors(df):
        input_data_word = []
        input_data_char = []
        gold_class_data = []
        
        len_dic = {}
        for row in df.itertuples():

            #row[3] = word sentences, row[4] = char sent, row[5] = tags
            if str(len(row[3])) not in len_dic:
                len_dic[str(len(row[3]))] = [(row[3], row[4], row[5])]
            else:
                len_dic[str(len(row[3]))].append((row[3], row[4], row[5]))
        
        
        for key in len_dic.keys():
            batch_input_data_word = []
            batch_input_data_char = []
            batch_gold_class_data = []
            
            list_of_batches = len_dic[key]

            for item in list_of_batches: 
                
                batch_input_data_word.append(torch.tensor(item[0]))
                batch_input_data_char.append(torch.tensor(item[1]))
                batch_gold_class_data.append(torch.tensor(item[2]))
                
            k = 1

            for i in range(0, len(batch_input_data_word), k):
                tensor_batch_input_data_word = torch.stack(batch_input_data_word[i : i+k])  
                tensor_batch_input_data_char = torch.stack(batch_input_data_char[i : i+k])  
                tensor_batch_gold = torch.stack(batch_gold_class_data[i : i+k])
            
                input_data_word.append(tensor_batch_input_data_word)
                input_data_char.append(tensor_batch_input_data_char)
                gold_class_data.append(tensor_batch_gold)
            #put to device here
        return input_data_word, input_data_char, gold_class_data
                
                
        
    train_input_word, train_input_char, train_gold = convert2tensors(df_train)
    dev_input_word, dev_input_char, dev_gold = convert2tensors(df_dev)
    test_input_word, test_input_char, test_gold = convert2tensors(df_test)

    tensor_dict = {
            'train' : [train_input_word, train_input_char, train_gold],
            'dev' : [dev_input_word, dev_input_char, dev_gold],
            'test': [test_input_word, test_input_char, test_gold],
        }
 

    return tensor_dict, len(word2id), len(char2id), len(label2id)

