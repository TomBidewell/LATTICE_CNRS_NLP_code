import pickle
import os
import sys
from scipy.cluster.hierarchy import leaves_list, to_tree
import conllu
import pandas as pd
from pathlib import Path
import torch
import random
import csv
from tqdm import tqdm

os.path.join(os.path.dirname(__file__), '../')
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from All_languages.roberta_POS import ROBERTA



with open("/home/tbidewell/home/POS_tagging/code/scripts/Tree/linkage_matrix", "rb") as fp_tree:
    link_mat = pickle.load(fp_tree)

with open("/home/tbidewell/home/POS_tagging/code/scripts/Tree/id2lang", "rb") as fp_id2lang:
    id2lang = pickle.load(fp_id2lang)

with open("/home/tbidewell/home/POS_tagging/code/scripts/Tree/lang_dict", "rb") as fp_lang_dict:
    lang_dict = pickle.load(fp_lang_dict)



root, _ = to_tree(link_mat, rd=True)


directory = "/data/tbidewell/Tree"



def load_conllu(filename):
    with open(filename) as fp:
        data = conllu.parse(fp.read())

    sentences = [[token['form'] for token in sentence] for sentence in data]
    taggings = [[token['upos'] for token in sentence] for sentence in data]

    if len(sentences) > 1000:
        sentences = sentences[ : 1000]
        taggings = taggings[: 1000]
            
    return sentences, taggings



def get_list_leaves(x, parent_id):
    
    if x.is_leaf():

        child_id = x.get_id()

        folder_name = "alpha_" + str(parent_id) + "_" + str(child_id)

        parent_id = x.get_id()
            
        ids = x.pre_order(lambda x: x.id)

        train_sent = []
        train_tag = []
        dev_sent = []
        dev_tag = []
        test_sent = []
        test_tag = []
        
        for lang in ids:
            l = id2lang[lang]
            data = data_per_lang[l]
            train_sent.append(data['train'][0])
            train_tag.append(data['train'][1])
            dev_sent.append(data['dev'][0])
            dev_tag.append(data['dev'][1])
            test_sent.append(data['test'][0])
            test_tag.append(data['test'][1])

        random.Random(1).shuffle(train_sent)
        random.Random(1).shuffle(train_tag)
        random.Random(2).shuffle(dev_sent)
        random.Random(2).shuffle(dev_tag)
        random.Random(3).shuffle(test_sent)
        random.Random(3).shuffle(test_tag)

        path = Path(directory + "/" + folder_name)
        path.mkdir(parents=True)

        with open(directory + "/" + folder_name + "/train.csv", "w", newline='') as train_fp:
            writer = csv.writer(train_fp)
            for i, j in zip(train_sent, train_tag):
                writer.writerow((i, j))
        
        with open(directory + "/" + folder_name + "/dev.csv", "w", newline='') as dev_fp:
            writer = csv.writer(dev_fp)
            for i, j in zip(dev_sent, dev_tag):
                writer.writerow((i, j))
        
        with open(directory + "/" + folder_name + "/test.csv", "w", newline='') as test_fp:
            writer = csv.writer(test_fp)
            for i, j in zip(test_sent, test_tag):
                writer.writerow((i, j))


    else:

        child_id = x.get_id()

        folder_name = "alpha_" + str(parent_id) + "_" + str(child_id)

        parent_id = x.get_id()

        ids = x.pre_order(lambda x: x.id)
        
        train_sent = []
        train_tag = []
        dev_sent = []
        dev_tag = []
        test_sent = []
        test_tag = []
        
        for lang in ids:
            l = id2lang[lang]
            data = data_per_lang[l]
            train_sent.append(data['train'][0])
            train_tag.append(data['train'][1])
            dev_sent.append(data['dev'][0])
            dev_tag.append(data['dev'][1])
            test_sent.append(data['test'][0])
            test_tag.append(data['test'][1])

        random.Random(1).shuffle(train_sent)
        random.Random(1).shuffle(train_tag)
        random.Random(2).shuffle(dev_sent)
        random.Random(2).shuffle(dev_tag)
        random.Random(3).shuffle(test_sent)
        random.Random(3).shuffle(test_tag)


        path = Path(directory + "/" + folder_name)
        path.mkdir(parents=True)

        with open(directory + "/" + folder_name + "/train.csv", "w", newline='') as train_fp:
            writer = csv.writer(train_fp)
            for i, j in zip(train_sent, train_tag):
                writer.writerow((i, j))
        
        with open(directory + "/" + folder_name + "/dev.csv", "w", newline='') as dev_fp:
            writer = csv.writer(dev_fp)
            for i, j in zip(dev_sent, dev_tag):
                writer.writerow((i, j))
        
        with open(directory + "/" + folder_name + "/test.csv", "w", newline='') as test_fp:
            writer = csv.writer(test_fp)
            for i, j in zip(test_sent, test_tag):
                writer.writerow((i, j))

        
        get_list_leaves(x.get_right(), parent_id)
        get_list_leaves(x.get_left(), parent_id)
    

root_id = root.get_id()

langs = [id2lang[i] for i in root.pre_order(lambda x: x.id)]

data_per_lang = {}

for l in tqdm(langs, total=len(langs), desc='Per Lang'):

    links = lang_dict[l]
    data = {}

    for link in links:

        link = link[0]

        if 'train' in link:
            sentences, tags = load_conllu(link)
            data['train'] = (sentences, tags)

        elif 'dev' in link:
            sentences, tags = load_conllu(link)
            data['dev'] = (sentences, tags)

        elif 'test' in link:
            sentences, tags = load_conllu(link)
            data['test'] = (sentences, tags)

    data_per_lang[l] = data


get_list_leaves(root, root_id)



with open("/home/tbidewell/home/POS_tagging/code/scripts/Tree/data_per_lang", "wb") as d_per_lang:   #Pickling
    pickle.dump(data_per_lang, d_per_lang)