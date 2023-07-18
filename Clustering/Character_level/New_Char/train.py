import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from tqdm import tqdm
from pathlib import Path
from transformers import XLMRobertaModel
import pandas as pd



os.path.join(os.path.dirname(__file__), '../')
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from data_prep import prepared_data, prepared_all_data, prepared_data_other
from model import NEXTCHAR



def train_model(todo, device):


    num_layers = 2
    dropout = 0.5
    num_epochs = 1000

    optimizers = []

    df_train = pd.DataFrame([])
    df_dev = pd.DataFrame([])
    df_test = pd.DataFrame([])

    for i in todo:
        train, dev, test = prepared_data(i[0], i[1], i[2], i[3])
        df_train = pd.concat([df_train, train])
        df_dev = pd.concat([df_dev, dev])
        df_test = pd.concat([df_test, test])

    train_data_char, train_data_lang, char2id, lang2id = prepared_all_data(df_train)

    dev_data_char, dev_data_lang = prepared_data_other(df_dev, char2id, lang2id)

    test_data_char, test_data_lang = prepared_data_other(df_test, char2id, lang2id)
    

    num_chars = len(char2id)
    num_langs = len(lang2id)
    char_embedding_size = 100
    lang_embedding_size = 64
    hidden_size = 300
    num_layers= 2

    
    next_char = NEXTCHAR(num_chars, char_embedding_size, num_langs, lang_embedding_size, hidden_size, num_layers)

    optimizer = optim.Adadelta(next_char.parameters())

    next_char.to(device)

    loss_function = nn.NLLLoss()

    # to record the total loss at each epoch
    epoch_losses_train = []
    epoch_losses_dev = []
    epoch_accuracy_train = []
    epoch_accuracy_dev = []

    # loop on epochs
    for epoch in tqdm(range(num_epochs), total = num_epochs):
        #print("Epoch", epoch)
        epoch_loss = 0

        num_pred_train = 0
        good_pred_train = 0

        data = zip(train_data_char, train_data_lang)
       
        for data_1, data_2 in data:

            data_1 = data_1.to(device)

            data_2 = data_2.to(device)

            next_char.zero_grad()

            log_probs = next_char(data_1, data_2)

            log_probs = log_probs.transpose(1, 2)

            y = data_1[:,1:]

            log_probs = log_probs[:, :, :-1]

            loss = loss_function(log_probs, y) 
            
            epoch_loss += loss.item()
            

            loss.backward() 

            optimizer.step()
            

            mask_train = (y != -100)

            num_pred_train += mask_train.int().sum().item()

            pred_labels = torch.argmax(log_probs, dim=1)

            good_pred_train += (pred_labels.to(device) == y).mul(mask_train).int().sum().item()

        epoch_accuracy_train = [good_pred_train / num_pred_train * 100]

        epoch_losses_train.append([epoch_loss])

        with torch.no_grad():
            dev_loss_all = 0
            dev_accuracy_all = []

            good_pred_dev = 0
            num_pred_dev = 0

            dev_data = zip(dev_data_char, dev_data_lang)
       
            for dev_data_1, dev_data_2 in dev_data:

                dev_data_1 = dev_data_1.to(device)

                dev_data_2 = dev_data_2.to(device)

                next_char.zero_grad()

                dev_log_probs = next_char(dev_data_1, dev_data_2)

                dev_log_probs = dev_log_probs.transpose(1, 2)

                y_dev = dev_data_1[:,1:]

                dev_log_probs = dev_log_probs[:, :, :-1]

                dev_loss = loss_function(dev_log_probs, y_dev) 

                dev_loss_all += dev_loss.item()

                mask_dev = (y_dev != -100)

                num_pred_dev += mask_dev.int().sum().item()

                pred_dev_labels = torch.argmax(dev_log_probs, dim=1)

                good_pred_dev += (pred_dev_labels.to(device) == y_dev).mul(mask_dev).int().sum().item()

        epoch_accuracy_dev = [good_pred_dev / num_pred_dev * 100]

        epoch_losses_dev.append([dev_loss_all])

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
            

    with torch.no_grad():
        test_loss_all = 0
        test_accuracy_all = []

        num_pred_test = 0
        good_pred_test = 0

        test_data = zip(test_data_char, test_data_lang)
    
        for test_data_1, test_data_2 in test_data:

            test_data_1 = test_data_1.to(device)

            test_data_2 = test_data_2.to(device)

            next_char.zero_grad()

            test_log_probs = next_char(test_data_1, test_data_2)

            test_log_probs = test_log_probs.transpose(1, 2)

            y_test = test_data_1[:,1:]

            test_log_probs = test_log_probs[:, :, :-1]

            test_loss = loss_function(test_log_probs, y_test) 

            test_loss_all += test_loss.item()

            mask_test = (y_test != -100)

            num_pred_test += mask_test.int().sum().item()

            pred_test_labels = torch.argmax(test_log_probs, dim=1)

            good_pred_test += (pred_test_labels.to(device) == y_test).mul(mask_test).int().sum().item()

    test_accuracy_all = [good_pred_test / num_pred_test * 100]

    
    return next_char.state_dict()['lang_emb_layer.weight'], lang2id
 













