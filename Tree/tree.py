import os
import sys
import conllu
import csv
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from os import execlp, fork, wait
from scipy.cluster.hierarchy import leaves_list, to_tree
import pickle
from ast import literal_eval



os.path.join(os.path.dirname(__file__), '../')
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from Data_Prep.word_lstm_data_prep import word_prepared_data


with open("/home/tbidewell/home/POS_tagging/code/scripts/Tree/data_per_lang_trial", "rb") as fp_tree:
    data_per_lang = pickle.load(fp_tree)


with open("/home/tbidewell/home/POS_tagging/code/scripts/Tree/linkage_matrix_trial", "rb") as fp_tree:
    link_mat = pickle.load(fp_tree)

with open("/home/tbidewell/home/POS_tagging/code/scripts/Tree/id2lang_trial", "rb") as fp_id2lang:
    id2lang = pickle.load(fp_id2lang)


df = pd.DataFrame(data = None, columns = ['Sentence', 'POS'])

for lang, data in data_per_lang.items():
    data = data['train']
    df_data = pd.DataFrame(list(zip(data[0], data[1])), columns = ['Sentence', 'POS'])
    df = pd.concat([df,df_data])


counts = {}
def get_counts(x):
    for w in x:
        try: 
            counts[w.lower()] += 1
        except:
            counts[w.lower()] = 1

df['Sentence'].apply(lambda x: get_counts(x))


#creating indices for the vocab
word2id = {'PAD': 0,
        'UNK' : 1,
        }

label2id = {'PAD': 0,
        'UNK' : 1,
        }

def create_word_ids(x):
    for token in x:
        token = token.lower()
        if token not in word2id:
            if counts[token] == 1:
                word2id[token] = word2id['UNK']
            else:
                word2id[token] = len(word2id)

def create_label_ids(x):
    for label in x:
        if label not in label2id:
            label2id[label] = len(label2id)

df['Sentence'].apply(lambda x: create_word_ids(x))
df['POS'].apply(lambda x: create_label_ids(x))


root, _ = to_tree(link_mat, rd=True)

#get id2word etc

directory = '/data/tbidewell/Tree'

root_id = '132'

#get the model for the root

todo = []

for f in os.listdir(directory):
    fam, parent, id = f.split("_")

    if f not in todo:
        todo.append(f)

    for child_search in os.listdir(directory):
        ch_fam, ch_parent, ch_id = child_search.split("_")
        
        if str(ch_parent) == str(id) and str(ch_id) != str(id): #i.e found a child
            todo.append(child_search)

print(todo)
        


        
        





























'''
for f in os.listdir(directory):
    fam, parent, id = f.split("_")

    for search_parent in os.listdir(directory):
        par_fam, par_parent, par_id = search_parent.split("_")

        if str(parent) == str(par_id):
            #get model
            pass

    train = []
    dev = []
    test = []

    for file in os.listdir(directory + "/" + f):

        if "train" in file:
            train.append(directory + "/" + f + "/" + file)
        elif "test" in file:
            test.append(directory + "/" + f + "/" + file)
        elif "dev" in file:
            dev.append(directory + "/" + f + "/" + file)
    
    #train on train, dev, test
    print(train[0])
    print(dev[0])
    print(test[0])

'''
    

