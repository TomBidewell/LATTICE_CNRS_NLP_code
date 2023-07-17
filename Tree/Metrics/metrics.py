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
import numpy as np

directory = '/home/tbidewell/home/POS_tagging/Data/Tree_Trial'


with open("/home/tbidewell/home/POS_tagging/code/scripts/Tree/Pickled_Files/linkage_matrix_trial", "rb") as fp_tree:
    link_mat = pickle.load(fp_tree)

with open("/home/tbidewell/home/POS_tagging/code/scripts/Tree/Pickled_Files/id2lang_trial", "rb") as fp_id2lang:
    id2lang = pickle.load(fp_id2lang)


tree = to_tree(link_mat)

#get the model for the root

ids = tree.pre_order(lambda x: x.id)

per_node_data = {}



metrics_w_lstm = {'acc_dev': 0,
           'loss_dev': 0, 
           'acc_test': 0,
           'loss_test': 0, 
           'acc_train': 0,
           'loss_train': 0}

metrics_w_ch_lstm = {'acc_dev': 0,
           'loss_dev': 0, 
           'acc_test': 0,
           'loss_test': 0, 
           'acc_train': 0,
           'loss_train': 0}

metrics_transformer = {'acc_dev': 0,
           'loss_dev': 0, 
           'acc_test': 0,
           'loss_test': 0, 
           'acc_train': 0,
           'loss_train': 0}

metrics_cnn = {'acc_dev': 0,
           'loss_dev': 0, 
           'acc_test': 0,
           'loss_test': 0, 
           'acc_train': 0,
           'loss_train': 0}


model_metrics = {
    'w_ch_lstm': metrics_w_ch_lstm, 
    'w_lstm': metrics_w_lstm, 
    'transformer': metrics_transformer,
    'cnn': metrics_cnn
}


def extract_data(file, f , child, model, csv_path, type_of_metric, metric_dict, metrics):
    if model in f:
        for metric in os.listdir(directory + "/" + file + "/" + f):
            if csv_path in metric:
                with open(directory + "/" + file + "/" + f + '/' + csv_path, newline='') as csv_f:
                    reader = csv.reader(csv_f)
                    data = list(reader)
                    total = []
                    for i in data:
                        if type_of_metric not in i[0]:
                            total.append(float(i[0][1:-1]))
                    metric_dict[metrics] += np.mean(total)  



def extract_all(csv_file, type, name_in_dict):
    extract_data(file, f , child, "w_lstm", csv_file, type, metrics_w_lstm, name_in_dict)
    extract_data(file, f , child, "w_ch_lstm", csv_file, type, metrics_w_ch_lstm, name_in_dict)
    extract_data(file, f , child, "transformer", csv_file, type, metrics_transformer, name_in_dict)
    extract_data(file, f , child, "cnn", csv_file, type, metrics_cnn, name_in_dict)



for file in os.listdir(directory):
    fam, parent, child = file.split("_")
    if int(child) in ids:
        for f in os.listdir(directory + "/" + file):
            extract_all("epoch_accuracy_dev.csv", "Accuracy", 'acc_dev')
            extract_all("epoch_accuracy_train.csv", "Accuracy", 'acc_train')
            extract_all("epoch_losses_dev.csv", "Loss", 'loss_dev')
            extract_all("epoch_losses_train.csv", "Loss", 'loss_train')
            extract_all("test_accuracy_all.csv", "Accuracy", 'acc_test')
            extract_all("test_loss_all.csv", "Loss", 'loss_test')
  
  
for key, value in model_metrics.items():
    for i, j in value.items():
        value[i] = j / len(ids)
    model_metrics[key] = value

for k, v in model_metrics.items():
    print(k, v)



