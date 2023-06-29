from io import open
import os
import sys
import conllu
import csv
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import pickle
import random


os.path.join(os.path.dirname(__file__), '../')
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


def load_conllu(filename):
    with open(filename) as fp:
        data = conllu.parse(fp.read())

    sentences = [[token['form'] for token in sentence] for sentence in data]
    taggings = [[token['upos'] for token in sentence] for sentence in data]
    return sentences, taggings


def count_conllu(filename):
  with open(filename) as fp:
    data = conllu.parse(fp.read())
  return len(data)


# assign directory
directory = '/home/tbidewell/home/POS_tagging/Data/Trial'
 
# iterate over files in
# that directory
train_dict = {}
dev_dict = {}
test_dict = {}


lang_dict = {}
amount_of_data = []

for lang in os.listdir(directory):
    lang_f = os.path.join(directory, lang)
  
    for file in os.listdir(lang_f):
        if ".conllu" in file:
            l = file.split("_")[0]
            n = count_conllu(lang_f + "/" + file)
            amount_of_data.append(n)
            if l not in lang_dict:
               lang_dict[l] = [(lang_f + "/" + file, n)]
            else:
               lang_dict[l].append((lang_f + "/" + file, n))


data_per_lang = {}
amount_of_data = []

for key in lang_dict:
    link_n_count = lang_dict[key]
    total_data = 0
    for item in link_n_count:
        total_data += item[1]
    data_per_lang[key] = total_data
    amount_of_data.append(total_data)



max_data_per_lang = max(amount_of_data) #round(np.quantile(amount_of_data, .75))


n = 3
n_gram_upos2id = {}


#go through and create the tri-gram upos tags

for key in lang_dict.keys():
    link_counts = lang_dict[key]

    all_tags_per_lang = []
    for link in link_counts:
        _, list_of_tags = load_conllu(link[0])
        for sent in list_of_tags:
            all_tags_per_lang.append(sent)
    
    random.Random(1).shuffle(all_tags_per_lang)

    if len(all_tags_per_lang) <= max_data_per_lang:
        data = all_tags_per_lang
    else:
        data = all_tags_per_lang[:max_data_per_lang]

    for sent_tags in data:

        for i in range(len(sent_tags)-n):
            combination = "_".join(sent_tags[i : i+n])
            if combination not in n_gram_upos2id:
                n_gram_upos2id[combination] = len(n_gram_upos2id)


lang_count = np.zeros((len(lang_dict), len(n_gram_upos2id)))

lang2id = {}
id2lang = []

for enumerate_lang, key in enumerate(lang_dict.keys()):

    if key not in lang2id:
       lang2id[key] = len(lang2id)
       id2lang.append(key)
       
    link_counts = lang_dict[key]

    all_tags_per_lang = []
    for link in link_counts:
        _, list_of_tags = load_conllu(link[0])
        for sent in list_of_tags:
            all_tags_per_lang.append(sent)

    random.Random(1).shuffle(all_tags_per_lang)

    if len(all_tags_per_lang) <= max_data_per_lang:
        data = all_tags_per_lang
    else:
        data = all_tags_per_lang[:max_data_per_lang]

    for sent_tags in data:

        for i in range(len(sent_tags)-n):
            combination = "_".join(sent_tags[i : i+n])
            tag_id = n_gram_upos2id[combination]
            lang_count[enumerate_lang, tag_id] += 1




X = csr_matrix(lang_count)

svd = TruncatedSVD(n_components = 100)
X_new = svd.fit_transform(X)

with open("lang_counts_trial", "wb") as fp:   #Pickling
    pickle.dump(X_new, fp)

with open("id2lang_trial", "wb") as fid2lang_p:   #Pickling
    pickle.dump(id2lang, fid2lang_p)
    
with open("lang_dict_trial", "wb") as lang_dict_fp:   #Pickling
    pickle.dump(lang_dict, lang_dict_fp)