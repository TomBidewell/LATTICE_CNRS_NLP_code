import os
import sys
import conllu
import csv
from pathlib import Path
from tqdm import tqdm
from os import execlp, fork, wait
from train_word_char_lstm import w_ch_lstm
from train_word_lstm import w_lstm
from train_transformer import transformer

os.path.join(os.path.dirname(__file__), '../')
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


def load_conllu(filename):
  with open(filename) as fp:
    data = conllu.parse(fp.read())

  sentences = [[token['form'].replace(" ", "") for token in sentence] for sentence in data]
  taggings = [[token['upos'] for token in sentence] for sentence in data]
  return sentences, taggings


todo = []
models =  [w_ch_lstm, w_lstm, transformer]


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
            train = lang_f + "/" + file

        elif "dev.csv" in file:
            dev = lang_f + "/" + file
            
        elif "test.csv" in file:
            test = lang_f + "/" + file



    for model in models:
        todo.append((lang_f, train, dev, test, model))

        
print("Running Training: ")


# then run it

running = {'0':-1, '1':-1}# these are the two gpus or atropos

while todo != []:
    if -1 in running.values(): # there at least one free GPU (for me)

        lang_f, train,dev,test, model = todo[0]

        dest = lang_f.split("/")[-1]
        destination = "/data/tbidewell/Metrics/" + dest
        #destination = "/home/mdehouck/" + dest

        todo = todo[1:]

        gpu = [x for x, y in running.items() if y == -1][0]

        pid = fork() # fork

        if pid == 0:

            execlp('python3.9', 'python3.9', '/home/tbidewell/home/POS_tagging/code/old/run_models.py', destination, train, dev, test, gpu, model.__name__)

            exit()

        else:

            # the parent remembers the pid, could also store the parameters todo[0]

            running[gpu] = pid

    else: # all the gpus are being used

        pid, status = wait()

          # status should be 0

        gpu = [gpu for gpu, process in running.items() if process == pid][0]

        running[gpu] = -1

        
wait()



            
       

