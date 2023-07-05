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

from Train_models.train_lstm import w_lstm
from Models.word_lstm import WORD_LSTM



model_dict = {
    'w_lstm': WORD_LSTM
}


with open("/home/tbidewell/home/POS_tagging/code/scripts/Tree/Pickled_Files/data_per_lang_trial", "rb") as fp_tree:
    data_per_lang = pickle.load(fp_tree)


with open("/home/tbidewell/home/POS_tagging/code/scripts/Tree/Pickled_Files/linkage_matrix_trial", "rb") as fp_tree:
    link_mat = pickle.load(fp_tree)

with open("/home/tbidewell/home/POS_tagging/code/scripts/Tree/Pickled_Files/id2lang_trial", "rb") as fp_id2lang:
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


with open("/home/tbidewell/home/POS_tagging/code/scripts/Tree/Pickled_Files/word2id", "wb") as word2id_fp:   #Pickling
    pickle.dump(word2id, word2id_fp)



with open("/home/tbidewell/home/POS_tagging/code/scripts/Tree/Pickled_Files/label2id", "wb") as label2id_fp:   #Pickling
    pickle.dump(label2id, label2id_fp)


root, _ = to_tree(link_mat, rd=True)

#get id2word etc

directory = '/home/tbidewell/home/POS_tagging/Data/Tree_Trial'

root_id = root.get_id()

#get the model for the root


trains = {}
devs = {}
tests = {}

for file in os.listdir(directory):
    fam, parent, id = file.split("_")

    for f in os.listdir(directory + "/" + file):
        if 'train' in f:
            fam, par, cur = file.split('_')
            trains[fam, int(par), int(cur)] = directory + "/" + file + "/" + f
        elif 'dev' in f:
            fam, par, cur = file.split('_')
            devs[fam, int(par), int(cur)] = directory + "/" + file + "/" + f
        elif 'test' in f:
            fam, par, cur = file.split('_')
            tests[fam, int(par), int(cur)] = directory + "/" + file + "/" + f
    
todo = []
children = {}

for k, v in sorted(trains.items(), reverse= True):

    if k not in devs:
        continue

    if str(k[1]) == str(root_id) and str(k[2]) == str(root_id):
        for mod in [w_lstm]:    #, 'transf']:
            todo.append((k, v, devs[k], tests[k], mod))

        
    elif str(k[1]) == str(root_id) and str(k[2]) != str(root_id):
        for mod in [w_lstm]:   
                children[k[0], k[2], mod] = []

    else:
        for mod in [w_lstm]:      #, 'transf']:
            children[k[0], k[1], mod].append((k, v, devs[k], tests[k], mod))
            children[k[0], k[2], mod] = []

 

for k, v in sorted(children.items()):
    if len(v) == 0:
        del(children[k])


# then run it
running = {'0':-1, '1':-1}# these are the two gpus on atropos
#running = {'1':-1}
procs = {}
done = set()




while todo != [] or len(children) != 0:

    if len(todo) != 0 and -1 in running.values(): # there at least one free GPU (for me)
            k, train, dev, test, mod = todo[0]

            #print(k, mod)#, done)
        
            todo = todo[1:]

            gpu = [x for x, y in running.items() if y == -1][0]

            pid = fork() # fork
        
            if pid == 0:
                # run you new code in the child process
                bits = train.split('/')[-2].split('_')
                this = bits[0] + '_' + bits[-1]
                parent = '_'.join(bits[:2])

                destination = directory + "/" + train.split('/')[-2] + "/" + mod.__name__

                for file in os.listdir(directory):
                    if str(k[1]) == str(file.split("_")[-1]):
                        parent_folder = file

                if str(bits[1]) == str(root_id) and str(bits[2]) == str(root_id):  #i.e we're at the root node
                    parent_model = 'NIL'
                else:
                    model_name = model_dict[mod.__name__]
                    parent_model = directory + "/" + parent_folder + "/" + mod.__name__ + "/" + model_name.__name__

                execlp('python3.9', 'python3.9', '/home/tbidewell/home/POS_tagging/code/scripts/Tree/Run/run_tree.py', destination, mod.__name__, parent_model, train, dev, test, gpu)
                exit()
            
            else:
                # the parent remembers the pid,    could also store the parameters todo[0]
                procs[pid] = k[0], k[1], k[2], mod
                running[gpu] = pid

    else: # if no available GPU, or no next task, wait
        pid, status = wait() # here we could catch the failing parameters and put them back in todo...


        if status != 0:
            with open('failed', 'a') as out:
                print(pid, status, procs[pid], file=out)

        else:
            task = procs[pid]
            done.add(task)
            #print(pid, task)

            key = tuple(task[x] for x in [0, 2, 3])
            if key in children:
                #print(children[tuple(task[x] for x in [0, 2, 3])])
                todo += children[tuple(task[x] for x in [0, 2, 3])]
                del(children[tuple(task[x] for x in [0, 2, 3])])
            
        if running['0'] == pid:
            running['0'] = -1
        elif running['1'] == pid:
            running['1'] = -1
