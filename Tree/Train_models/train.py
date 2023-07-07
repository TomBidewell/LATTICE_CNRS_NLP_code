import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from tqdm import tqdm
from pathlib import Path
from transformers import XLMRobertaModel



os.path.join(os.path.dirname(__file__), '../')
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from Data_Prep.tree_word_lstm_data_prep import word_prepared_data
from Data_Prep.tree_transformer_data_prep import word_char_prepared_data
#from Data_Prep.tree_transformer_data_prep import tree_transformer_data_prep

from Models.word_char_lstm import LSTM_WORD_CHAR
from Models.roberta_POS import ROBERTA
from Models.word_lstm import WORD_LSTM


models = {
    "w_lstm": WORD_LSTM, 
    "w_ch_lstm": LSTM_WORD_CHAR, 
    "transformer": ROBERTA
}


def train_model(path, transformer, model_name, train, dev, test, device):

    model_class = models[model_name]

    num_layers = 2
    dropout = 0.5
    num_epochs = 1000

    optimizers = []

    if transformer:

        tensor_dict, num_classes = tree_transformer_data_prep(train, dev, test)

        batches_features_input_train, batches_features_att_train, batches_gold_train = tensor_dict['train']
        batches_features_input_dev, batches_features_att_dev, batches_gold_dev = tensor_dict['dev']
        batches_features_input_test, batches_features_att_test, batches_gold_test = tensor_dict['test']

        hidden_layer_size = 500

        pre_trained_model =  XLMRobertaModel.from_pretrained('xlm-roberta-base')
        pre_trained_model.to(device)

        model = ROBERTA(num_classes, hidden_layer_size) 

        optimizers.append(optim.Adadelta(model.parameters()))

        model.roberta = pre_trained_model

        optimizers.append(optim.SGD(model.roberta.parameters(), lr = 0.001))

    else:
        if model_name == "w_lstm":
            tensor_dict, len_word2id, len_label2id = word_prepared_data(train, dev, test)

            train_input, train_gold = tensor_dict['train']
            dev_input, dev_gold = tensor_dict['dev']
            test_input_word, test_gold = tensor_dict['test']

            vocab_size = len_word2id
            num_classes = len_label2id
            embedding_size = 300
            hidden_layer_size = 300
            bidirectional = True
            batch_first = True
            
            model = model_class(vocab_size, embedding_size, num_classes, hidden_layer_size, num_layers, dropout, batch_first, bidirectional)

        elif model_name == "w_ch_lstm":

            tensor_dict, len_w2id, len_char2id, len_lab2id = word_char_prepared_data(train, dev, test)

            train_input_word, train_input_char, train_gold = tensor_dict['train']
            dev_input_word, dev_input_char, dev_gold = tensor_dict['dev']
            test_input_word, test_input_char, test_gold = tensor_dict['test']


            vocab_size = len_w2id
            char_size = len_char2id
            num_classes = len_lab2id
            embedding_size_char = 100
            embedding_size_word = 300
            hidden_layer_size_char = 100
            hidden_layer_size_word = 300
        
            model = model_class(vocab_size, char_size, embedding_size_char, embedding_size_word, num_classes, hidden_layer_size_char, hidden_layer_size_word, num_layers, device, dropout)

        optimizers.append(optim.Adadelta(model.parameters()))
    
    model.to(device)

    save_path = path / model.__class__.__name__ 

    loss_function = nn.NLLLoss()

    # to record the total loss at each epoch
    epoch_losses_train = []
    epoch_losses_dev = []
    epoch_accuracy_train = []
    epoch_accuracy_dev = []

    # loop on epochs
    for epoch in range(num_epochs):
        #print("Epoch", epoch)
        epoch_loss = 0

        num_pred_train = 0
        good_pred_train = 0

        if transformer:
            length = len(batches_features_input_train)
            data = zip(batches_features_input_train, batches_features_att_train, batches_gold_train)
        else:
            if model_name == "w_ch_lstm":
                length = len(train_input_word)
                data = zip(train_input_word, train_input_char, train_gold)

            elif model_name == "w_lstm":
                length = len(train_input)
                data = zip(train_input, [0]*len(train_input), train_gold)

        for data_1, data_2, y in tqdm(data, total = length, desc = 'Train: '):

            data_1 = data_1.to(device)

            if isinstance(data_2, int) == False: #i.e word and char
                data_2 = data_2.to(device)

            y = y.to(device)

            model.zero_grad()

            if isinstance(data_2, int): # i.e. just word
                log_probs = model(data_1)

            else:
                log_probs = model(data_1, data_2)

            log_probs = log_probs.transpose(1, 2)

            loss = loss_function(log_probs, y) 
            
            epoch_loss += loss

            loss.backward() 

            for optimizer in optimizers:
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

            if transformer:
                dev_length = len(batches_features_input_dev)
                dev_data = zip(batches_features_input_dev, batches_features_att_dev, batches_gold_dev)
            else:
                if model_name == "w_ch_lstm":
                    dev_length = len(dev_input_word)
                    dev_data = zip(dev_input_word, dev_input_char, dev_gold)

                elif model_name == "w_lstm":
                    dev_length = len(dev_input)
                    dev_data = zip(dev_input, [0]*len(dev_input), dev_gold)

            for dev_data_1, dev_data_2, y_dev in tqdm(dev_data, total = dev_length, desc='Dev: '):

                dev_data_1 = dev_data_1.to(device)

                if isinstance(dev_data_2, int) == False:
                    dev_data_2 = dev_data_2.to(device)

                y_dev = y_dev.to(device)
                
                if isinstance(dev_data_2, int):
                    log_dev_probs = model(dev_data_1) 
                else:
                    log_dev_probs = model(dev_data_1, dev_data_2)

                log_dev_probs = log_dev_probs.transpose(1, 2)

                # total loss on the dev set
                dev_loss = loss_function(log_dev_probs, y_dev)

                dev_loss_all += dev_loss

                mask_dev = (y_dev != -100)

                num_pred_dev += mask_dev.int().sum().item()

                pred_dev_labels = torch.argmax(log_dev_probs, dim=1)

                good_pred_dev += (pred_dev_labels.to(device) == y_dev).mul(mask_dev).int().sum().item()

        epoch_accuracy_dev = [good_pred_dev / num_pred_dev * 100]

        epoch_losses_dev.append([dev_loss_all])

        if epoch == 0:
            highest_accuracy = epoch_accuracy_dev
            decrease_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            if highest_accuracy > epoch_accuracy_dev:
                decrease_counter += 1
                if decrease_counter > 10:
                    break
            else:
                highest_accuracy = epoch_accuracy_dev
                decrease_counter = 0
                torch.save(model.state_dict(), save_path)
            
    model.load_state_dict(torch.load(save_path))

    with torch.no_grad():
        test_loss_all = 0
        test_accuracy_all = []

        num_pred_test = 0
        good_pred_test = 0

        if transformer:
            test_data = zip(batches_features_input_test, batches_features_att_test, batches_gold_test)
        else:
            if model_name == "w_ch_lstm":
                test_data = zip(test_input_word, test_input_char, test_gold)

            elif model_name == "w_lstm":
                test_data = zip(test_input_word, [0]*len(test_input_word), test_gold)

        for test_data_1, test_data_2, y_test in test_data:

            test_data_1 = test_data_1.to(device)

            if isinstance(test_data_2, int) == False:
                test_data_2 = test_data_2.to(device)

            y_test = y_test.to(device)


            if isinstance(test_data_2, int):
                log_test_probs = model(test_data_1)
            else:
                log_test_probs = model(test_data_1, test_data_2)


            log_test_probs = log_test_probs.transpose(1, 2)


            # total loss on the dev set
            test_loss = loss_function(log_test_probs, y_test)

            test_loss_all += test_loss

            mask_test = (y_test != -100)

            num_pred_test += mask_test.int().sum().item()

            pred_test_labels = torch.argmax(log_test_probs, dim=1)

            good_pred_test += (pred_test_labels.to(device) == y_test).mul(mask_test).int().sum().item()

    test_accuracy_all = [good_pred_test / num_pred_test * 100]
    
    return epoch_losses_train, [epoch_accuracy_train], epoch_losses_dev, [highest_accuracy], [[test_loss_all]], [test_accuracy_all]













