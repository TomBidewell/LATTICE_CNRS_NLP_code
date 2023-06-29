import pickle
import os
import sys
from scipy.cluster.hierarchy import leaves_list, to_tree
import conllu
import pandas as pd
from word_lstm_data_prep import word_prepared_data
from train_word_lstm import w_lstm
from pathlib import Path
import torch
from word_lstm import WORD_LSTM


os.path.join(os.path.dirname(__file__), '../')
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


def load_conllu(filename):
    with open(filename) as fp:
        data = conllu.parse(fp.read())

    sentences = [[token['form'] for token in sentence] for sentence in data]
    taggings = [[token['upos'] for token in sentence] for sentence in data]
    return sentences, taggings


with open("linkage_matrix_trial", "rb") as fp_tree:
    link_mat = pickle.load(fp_tree)

with open("/home/tbidewell/home/POS_tagging/code/old/id2lang_trial", "rb") as fp_id2lang:
    id2lang = pickle.load(fp_id2lang)

with open("/home/tbidewell/home/POS_tagging/code/old/lang_dict_trial", "rb") as fp_lang_dict:
    lang_dict = pickle.load(fp_lang_dict)




root, _ = to_tree(link_mat, rd=True)

directory = "/data/tbidewell/Tree_trial"


def all_leaves_per_node(x):
    if x.is_leaf():
        links = [(i[0], id2lang[x.get_id()]) for i in lang_dict[id2lang[x.get_id()]]]
        return (links, x.get_id())
    else:
        all_leaves_per_node(x.get_left())
        all_leaves_per_node(x.get_right())



l = []

def get_list_leaves(x):
    if x.is_leaf():
        links = [[(i[0], id2lang[x.get_id()]) for i in lang_dict[id2lang[x.get_id()]]]]
        l.append([links])
    else:
        languages = [i[0] for i in x.pre_order(lambda y: all_leaves_per_node(y))]

        langs_per_node = []

        for language in languages: 
            for file in language:
                langs_per_node.append(file[1])
        node_id = "_".join(list(set(langs_per_node)))

        all_nodes_sent_train = []
        all_nodes_tag_train = []
        all_nodes_sent_dev = []
        all_nodes_tag_dev = []
        all_nodes_sent_test = []
        all_nodes_tag_test = []

        for lang in languages:
            for file in lang:

                if "train" in file:
                    sent, tags = load_conllu(file)
                    all_nodes_sent_train.append(sent)
                    all_nodes_tag_train.append(tags)

                if "dev" in file:
                    sent, tags = load_conllu(file)
                    all_nodes_sent_dev.append(sent)
                    all_nodes_tag_dev.append(tags)

                if "train" in file:
                    sent, tags = load_conllu(file)
                    all_nodes_sent_test.append(sent)
                    all_nodes_tag_test.append(tags)

        df_train = pd.DataFrame(list(zip(all_nodes_sent_train, all_nodes_tag_train)), columns= ['Sentence', 'POS'])
        df_dev = pd.DataFrame(list(zip(all_nodes_sent_dev, all_nodes_tag_dev)), columns= ['Sentence', 'POS'])
        df_test = pd.DataFrame(list(zip(all_nodes_sent_test, all_nodes_tag_test)), columns= ['Sentence', 'POS'])

        path = Path(directory + "/" + node_id + "/" + w_lstm.__name__ )
        path.mkdir(parents=True)




        save_path, epoch_losses_train, epoch_accuracy_train, epoch_losses_dev, epoch_accuracy_dev, test_loss_all, test_accuracy_all = w_lstm(path, df_train, df_dev, df_test, '0', tree = False)
        
        WORD_LSTM.load_state_dict(torch.load(save_path))
        WORD_LSTM.eval()
        
        get_list_leaves(x.get_right(), WORD_LSTM)
        get_list_leaves(x.get_left(), WORD_LSTM)















get_list_leaves(root)



