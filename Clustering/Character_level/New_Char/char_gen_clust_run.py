import csv
import torch
from pathlib import Path
import sys
import os
import torch

os.path.join(os.path.dirname(__file__), '../')
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from torch import manual_seed
from random import seed
from Clustering.Character_level.New_Char.char_gen_clust_train import train_model
from tqdm import tqdm
from argparse import ArgumentParser
import pickle

seed(5)
manual_seed(5)


def run_model(todo, device):

    device = torch.device(int(device))

    lang_embedding, lang2id = train_model(todo, device)

    torch.save(lang_embedding, "/home/tbidewell/home/POS_tagging/code/scripts/Clustering/Character_level/Pickled_Files/lang_embeds.pt")

    with open("/home/tbidewell/home/POS_tagging/code/scripts/Clustering/Character_level/Pickled_Files/lang2id", "wb") as lang2id_fp:   #Pickling
        pickle.dump(lang2id, lang2id_fp)



