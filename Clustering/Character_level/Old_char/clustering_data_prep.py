import pandas as pd
from ast import literal_eval
import torch
import numpy as np
import torch



def clustering_data_prep(train, dev, test):
    print("Starting data_prep")
    df_train = pd.read_csv(train) #1006450
    df_dev = pd.read_csv(dev)   #126020
    df_test = pd.read_csv(test) #179386
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_dev = df_dev.sample(frac=1).reset_index(drop=True)
    df_test = df_test.sample(frac=1).reset_index(drop=True)

    df_train = df_train.head(200)
    df_dev = df_dev.head(50)
    df_test = df_test.head(10)

    df_train['Sentence'] = df_train.Sentence.apply(lambda x: literal_eval(str(x))[0])
    df_train = df_train.drop(df_train.columns[0], axis =1)
    df_dev['Sentence'] = df_dev.Sentence.apply(lambda x: literal_eval(str(x))[0])
    df_dev = df_dev.drop(df_dev.columns[0], axis =1)

    df_test['Sentence'] = df_test.Sentence.apply(lambda x: literal_eval(str(x))[0])
    df_test = df_test.drop(df_test.columns[0], axis =1)

    #clean - change upper cases to lower cases
    df_train.Sentence = df_train.Sentence.apply(lambda x: x.strip().lower().replace("\"", ""))
    df_dev.Sentence = df_dev.Sentence.apply(lambda x: x.strip().lower().replace("\"", ""))
    df_test.Sentence = df_test.Sentence.apply(lambda x: x.strip().lower().replace("\"", ""))

    def remove_empties(x):
        if len(x) == 0:
            result = np.nan
        else:
            result = 1
        return result


    df_train['Catch_Empties'] = df_train.Sentence.apply(lambda x: remove_empties(x))
    df_dev['Catch_Empties'] = df_dev.Sentence.apply(lambda x: remove_empties(x))
    df_test['Catch_Empties'] = df_test.Sentence.apply(lambda x: remove_empties(x))

    df_train = df_train.dropna()
    df_dev = df_dev.dropna()
    df_test = df_test.dropna()

    df_train = df_train[['Sentence', 'Language']]
    df_dev = df_dev[['Sentence', 'Language']]
    df_test = df_test[['Sentence', 'Language']]


    #get character ids
    char2id = {'PAD': 0,
            'BOS': 1,
            'EOS': 2,
            'UNK': 3}

    def get_char_ids(x):
        for i in x:
            if i not in char2id:
                char2id[i] = len(char2id)

    df_train.Sentence.apply(lambda x: get_char_ids(x))

    #get lang ids
    lang2id = {'UNK': 0,}
    id2lang = []

    def get_lang_ids(x):
        if x not in lang2id:
            lang2id[x] = len(lang2id)
            id2lang.append(x)

    df_train.Language.apply(lambda x: get_lang_ids(x))


    #encode characters

    def encode_sent(x):
        encoded_sent = [char2id['BOS']]
        for i in x:
            if i in char2id:
                encoded_sent.append(char2id[i])
            else:
                encoded_sent.append(char2id['UNK'])
        encoded_sent.append(char2id['EOS'])
        return encoded_sent


    df_train['Encoded_Sent'] = df_train.Sentence.apply(lambda x: encode_sent(x))
    df_dev['Encoded_Sent'] = df_dev.Sentence.apply(lambda x: encode_sent(x))
    df_test['Encoded_Sent'] = df_test.Sentence.apply(lambda x: encode_sent(x))


    #encode lang

    def encode_langs(x):
        if x in lang2id:
            lang_encoding = lang2id[x]
        else:
            lang_encoding = lang2id['UNK']
        return lang_encoding

    df_train['Encoded_Lang'] = df_train.Language.apply(lambda x: encode_langs(x))
    df_dev['Encoded_Lang'] = df_dev.Language.apply(lambda x: encode_langs(x))
    df_test['Encoded_Lang'] = df_test.Language.apply(lambda x: encode_langs(x))


    #padding
    window_size = 2

    def padding(x):
        for i in range(window_size):
            x.append(0)
            x.insert(0, 0)
        return x

    df_train.Encoded_Sent = df_train.Encoded_Sent.apply(lambda x: padding(x))
    df_dev.Encoded_Sent = df_dev.Encoded_Sent.apply(lambda x: padding(x))
    df_test.Encoded_Sent = df_test.Encoded_Sent.apply(lambda x: padding(x))

    def convert2tensors(df):
        data = []
        for row in df.itertuples():
            for id in row[3]:   #row[3] is the encoded sent
                if id != 0:
                    index_of_word = row[3].index(id)
                    window_data = row[3][index_of_word - window_size : index_of_word + window_size + 1]
                    window_data.pop(window_size)
                    gold_class = row[3][index_of_word]
                    data.append(( torch.tensor(window_data) , torch.tensor(row[4]), torch.tensor(gold_class)))
        input_tensor = torch.stack(list(zip(*data))[0])
        language = torch.LongTensor(torch.stack(list(zip(*data))[1]))
        gold_class_tensor = torch.LongTensor(torch.stack(list(zip(*data))[2]))
        return input_tensor, language, gold_class_tensor

    train_input, train_language, train_gold = convert2tensors(df_train)
    dev_input, dev_language, dev_gold = convert2tensors(df_dev)
    test_input, test_language, test_gold = convert2tensors(df_test)

    tensor_dict = {
    'train': [train_input, train_language, train_gold],
    'dev': [dev_input, dev_language, dev_gold],
    'test': [test_input, test_language, test_gold]
    }

    return tensor_dict, len(char2id), lang2id, id2lang