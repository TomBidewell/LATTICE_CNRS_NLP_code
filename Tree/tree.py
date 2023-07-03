import os
import sys
import conllu
import csv
from pathlib import Path
from tqdm import tqdm
from os import execlp, fork, wait
#from All_languages.train_word_char_lstm import w_ch_lstm
#from All_languages.train_word_lstm import w_lstm
#from All_languages.train_transformer import transformer

os.path.join(os.path.dirname(__file__), '../')
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


directory = '/data/tbidewell/Tree'




trains = {}
devs = {}
tests = {}

for f in os.listdir(directory):

    for file in os.listdir(directory + "/" + f):

        if 'train' in file:
            fam, par, cur = f.split('_')
            trains[fam, int(par), int(cur)] = directory + "/" + f + "/" + file

        elif 'dev' in file:
            fam, par, cur = f.split('_')
            devs[fam, int(par), int(cur)] = directory + "/" + f + "/" + file

        elif 'test' in file:
            fam, par, cur = f.split('_')
            devs[fam, int(par), int(cur)] = directory + "/" + f + "/" + file



todo = []
children = {}
for k, v in sorted(trains.items()):
    if k not in devs:
        continue

    if k[1] == 0:
        for mod in ['lstm', 'char']:#, 'transf']:
            todo.append((k, v, devs[k], mod))
            children[k[0], k[2], mod] = []

    else:
        for mod in ['lstm', 'char']:#, 'transf']:
            children[k[0], k[1], mod].append((k, v, devs[k], mod))
            children[k[0], k[2], mod] = []

print(trains)