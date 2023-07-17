from io import open
import os
import sys
import conllu
import csv
import pandas as pd

os.path.join(os.path.dirname(__file__), '../')
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


def load_conllu(filename):
  with open(filename) as fp:
    data = conllu.parse(fp.read())

  sentences = [[token['form'] for token in sentence] for sentence in data]
  taggings = [[token['upos'] for token in sentence] for sentence in data]
  return sentences, taggings



def extract_text(file, dict):
    lang = file.split("_")[0]
    text_file = open(lang_f + "/" + file, 'r')
    all_text = ''
    for line in text_file.readlines():
        if "_" not in line:
            all_text = all_text + line.replace("\n", "")

    if lang not in dict:
        dict[lang] = all_text
    else:
        dict[lang] = dict[lang] + ". " + all_text




# assign directory
directory = '/home/tbidewell/home/POS_tagging/Data/ud-treebanks-v2.12'
 
# iterate over files in
# that directory
train_dict = {}
dev_dict = {}
test_dict = {}

for lang in os.listdir(directory):
    lang_f = os.path.join(directory, lang)
  
    for file in os.listdir(lang_f):
        if "train.txt" in file:
            extract_text(file, train_dict)

        elif "dev.txt" in file:
            extract_text(file, dev_dict)

        elif "test.txt" in file:
            extract_text(file, test_dict)



def get_data(dict):
    tmp_dict = [[([j], i) for j in dict[i].split(".")] for i in dict.keys()]
    all_data = []
    for i in tmp_dict:
        for j in i:
            all_data.append(j)
    return all_data


train_data = get_data(train_dict)
dev_data = get_data(dev_dict)
test_data = get_data(test_dict)


df_train = pd.DataFrame(train_data, columns =['Sentence', 'Language'])
df_dev = pd.DataFrame(dev_data, columns =['Sentence', 'Language'])
df_test = pd.DataFrame(test_data, columns =['Sentence', 'Language'])    

          
df_train.to_csv("~/home/POS_tagging/Data/Clustering_Data/per_lang_train.csv")      
df_dev.to_csv("~/home/POS_tagging/Data/Clustering_Data/per_lang_dev.csv")
df_test.to_csv("~/home/POS_tagging/Data/Clustering_Data/per_lang_test.csv")

        
    

        

      
            
