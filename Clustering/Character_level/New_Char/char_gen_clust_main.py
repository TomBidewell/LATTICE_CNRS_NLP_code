import os
import sys
import conllu
import csv
from pathlib import Path
from tqdm import tqdm
from os import execlp, fork, wait

os.path.join(os.path.dirname(__file__), '../')
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from Clustering.Character_level.New_Char.char_gen_clust_run import run_model


def load_conllu(filename):
  with open(filename) as fp:
    data = conllu.parse(fp.read())

  sentences = [[token['form'].replace(" ", "") for token in sentence] for sentence in data]
  taggings = [[token['upos'] for token in sentence] for sentence in data]
  return sentences, taggings


todo = []


# assign directory
directory = '/home/tbidewell/home/POS_tagging/Data/Clean'

# iterate over files in
# that directory



for lang in tqdm(os.listdir(directory), desc = 'Loading Each Language'):


    lang_f = os.path.join(directory, lang)

    
    for file in os.listdir(lang_f):
        if ".csv" in file:
            os.remove(lang_f + "/" + file)



      
    for file in os.listdir(lang_f):

        #writing the sentences to csv files 
        if "train.conllu" in file:
            
            train_sentences, train_taggings = load_conllu(lang_f + "/" + file)
            file = file.replace("conllu", "csv")
            with open(lang_f + "/" + file, "w") as train_file_out:
                writer = csv.writer(train_file_out)
                for i, j in zip(train_sentences, train_taggings):
                    writer.writerow((i, j))
        
        
        if "test.conllu" in file:

            test_sentences, test_taggings = load_conllu(lang_f + "/" + file)
            file = file.replace("conllu", "csv")
            
            with open(lang_f + "/" + file, "w") as test_file_out:
                writer = csv.writer(test_file_out)
                for i, j in zip(test_sentences, test_taggings):
                    writer.writerow((i, j))
            
        if "dev.conllu" in file:

            dev_sentences, dev_taggings = load_conllu(lang_f + "/" + file)
            file = file.replace("conllu", "csv")
            
            with open(lang_f + "/" + file, "w") as dev_file_out:
                writer = csv.writer(dev_file_out)
                for i, j in zip(dev_sentences, dev_taggings):
                    writer.writerow((i, j))
                
            
        
    for file in os.listdir(lang_f):
        
        if "train.csv" in file:

            language = file.split("_")[0]
            train = lang_f + "/" + file

        elif "dev.csv" in file:
            dev = lang_f + "/" + file
            
        elif "test.csv" in file:
            test = lang_f + "/" + file



    todo.append((language, train, dev, test))

        
print("Running Training: ")
num_repeats = str(1)

gpu = '0'


run_model(todo, gpu)
