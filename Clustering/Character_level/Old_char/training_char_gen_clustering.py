import pandas as pd
from ast import literal_eval
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim
from tqdm import tqdm
import numpy as np
from clustering_data_prep import clustering_data_prep
from char_generation_clustering import LANG_LSTM
import pickle

device = torch.device('cuda:1')

def train_gen_clustering(train, dev, test):

    tensor_dict, len_char2id, lang2id, id2lang = clustering_data_prep(train, dev, test)


    train_input, train_language, train_gold = tensor_dict['train']
    dev_input, dev_language, dev_gold = tensor_dict['dev']
    test_input, test_language, test_gold = tensor_dict['test']


    #make batches

    BATCH_SIZE = 24

    def split_into_batches(features, lang, gold):
        return torch.split(features, BATCH_SIZE), torch.split(lang, BATCH_SIZE), torch.split(gold, BATCH_SIZE)


    #initialise hyperparameters
    num_chars = len_char2id
    num_langs = len(lang2id)
    char_emb_size = 300
    lang_emb_size = 64
    hidden_size = 500
    num_layers = 2
    bidirectional = True
    dropout = 0
    window_size = 2


    #training
    lang_lstm = LANG_LSTM(window_size, num_chars, num_langs, char_emb_size, lang_emb_size, hidden_size, num_layers, bidirectional, dropout)
    lang_lstm.to(device)
    num_epochs = 1000

    # Define the loss function and optimizer
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(lang_lstm.parameters())


    training_loss = []
    training_accuracy = []
    dev_loss = []
    dev_accuracy = []



    # Training loop
    for epoch in range(num_epochs):
        lang_lstm.train()  # Set the model to training mode

        running_loss = 0.0

        # Training the model
        #shuffle data
        permutation = torch.randperm(len(train_input))
        train_input = train_input[permutation]
        train_language = train_language[permutation]
        train_gold = train_gold[permutation]
        
        batches_train_input, batches_train_language, batches_train_gold = split_into_batches(train_input, train_language, train_gold)
        
        num_preds = 0
        good_preds = 0
        
        for X_in, X_lang, y_true in zip(batches_train_input, batches_train_language, batches_train_gold):
            X_in = X_in.to(device)
            X_lang = X_lang.to(device)
            y_true = y_true.to(device)
            
            optimizer.zero_grad()

            log_probs = lang_lstm(X_in, X_lang)
            
            loss = loss_function(log_probs, y_true)

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            pred_labels = torch.argmax(log_probs, dim=1)
            
            good_preds += (pred_labels.to(device) == y_true).int().sum().item()
            
            num_preds += y_true.size()[0]

        # Print the average loss for the epoch
        average_loss = running_loss / len(batches_train_input)
        epoch_accuracy = good_preds/ num_preds * 100
        print(f"Epoch {epoch + 1}/{num_epochs}, Training:  Loss: {average_loss}, Accuracy: {epoch_accuracy}")
        training_loss.append(average_loss)
        training_accuracy.append(epoch_accuracy)
        
        
        
        batches_dev_input, batches_dev_language, batches_dev_gold = split_into_batches(dev_input, dev_language, dev_gold)
        
        dev_num_preds = 0
        dev_good_preds = 0
        
        for X_in_dev, X_lang_dev, y_true_dev in zip(batches_dev_input, batches_dev_language, batches_dev_gold):
            X_in_dev = X_in_dev.to(device)
            X_lang_dev = X_lang_dev.to(device)
            y_true_dev = y_true_dev.to(device)
            
            optimizer.zero_grad()

            log_probs_dev = lang_lstm(X_in_dev, X_lang_dev)
            
            loss = loss_function(log_probs_dev, y_true_dev)

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            pred_labels_dev = torch.argmax(log_probs_dev, dim=1)
            
            dev_good_preds += (pred_labels_dev.to(device) == y_true_dev).int().sum().item()
            
            dev_num_preds += y_true_dev.size()[0]

        # Print the average loss for the epoch
        average_loss_dev = running_loss / len(batches_dev_input)
        epoch_accuracy_dev = dev_good_preds/ dev_num_preds * 100
        print(f"             Dev: Loss: {average_loss_dev}, Accuracy: {epoch_accuracy_dev}")
        print()
        dev_loss.append(average_loss_dev)
        dev_accuracy.append(epoch_accuracy_dev)
        
        if epoch == 0:
            highest_accuracy = epoch_accuracy_dev
            decrease_counter = 0
        else:
            if highest_accuracy > epoch_accuracy_dev:
                decrease_counter += 1
                if decrease_counter > 10:
                    break
            else:
                highest_accuracy = epoch_accuracy_dev
                decrease_counter = 0

    return lang_lstm.state_dict()['lang_embedding.weight'], lang2id, id2lang


lang_embeds, lang2id, id2lang = train_gen_clustering("~/home/POS_tagging/Data/Clustering_Data/per_lang_train.csv", "~/home/POS_tagging/Data/Clustering_Data/per_lang_dev.csv", "~/home/POS_tagging/Data/Clustering_Data/per_lang_test.csv")


torch.save(lang_embeds, "lang_embeds.pt")

with open("lang2id", "wb") as lang2id_fp:   #Pickling
    pickle.dump(lang2id, lang2id_fp)

with open("id2lang", "wb") as id2lang_fp:   #Pickling
    pickle.dump(id2lang, id2lang_fp)