import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from All_languages.Data_Prep.word_lstm_data_prep import word_prepared_data
from All_languages.Models.word_lstm import WORD_LSTM
from tqdm import tqdm
from pathlib import Path

def w_lstm(path, train, dev, test, device):
    
    tensor_dict, len_word2id, len_label2id = word_prepared_data(train, dev, test)

    train_input, train_gold = tensor_dict['train']
    dev_input, dev_gold = tensor_dict['dev']
    test_input, test_gold = tensor_dict['test']

    
    vocab_size = len_word2id
    num_classes = len_label2id
    embedding_size = 300
    hidden_layer_size = 300
    num_layers = 2
    bidirectional = True
    batch_first = True
    dropout = 0.5
    num_epochs = 1000

    lstm = WORD_LSTM(vocab_size, embedding_size, num_classes, hidden_layer_size, num_layers, dropout, batch_first, bidirectional)


    lstm.to(device)

    save_path = path / lstm.__class__.__name__ 

    loss_function = nn.NLLLoss()

    optimizer = optim.Adadelta(lstm.parameters())
    


    # to record the total loss at each epoch
    epoch_losses_train = []
    epoch_losses_dev = []
    epoch_accuracy_train = []
    epoch_accuracy_dev = []

    # loop on epochs
    for epoch in tqdm(range(num_epochs), total = num_epochs, desc = 'Word LSTM: ') :
        #print("Epoch", epoch)
        epoch_loss = 0

        num_pred_train = 0
        good_pred_train = 0
        
    
        for X, y in zip(train_input, train_gold):
            
            X = X.to(device)
            y = y.to(device)

            #shuffle data
            #permutation = torch.randperm(X.size()[0])
            #X = X[permutation]
            #y = y[permutation]
                
            lstm.zero_grad()

            log_probs = lstm(X)
                
            """
            my y.shape = batch_size x seq_len
            
            my log_probs.shape = batch_size x seq_len x num_classes
            
            looking it up: NLLLoss takes input batch_size x num_classes x seq_len 
            
            so transpose the last two dimensions
            """
            
            log_probs = log_probs.transpose(1, 2)

            loss = loss_function(log_probs, y) 
            
            epoch_loss += loss.item()

            loss.backward() 

            optimizer.step()

            mask_train = (y != -100)

            num_pred_train += mask_train.int().sum().item()

            pred_labels = torch.argmax(log_probs, dim=1)

            good_pred_train += (pred_labels.to(device) == y).mul(mask_train).int().sum().item()
                    
            # prediction and accuracy on the dev set
            #pred_labels = torch.argmax(log_probs, dim=1)
            
            #accuracy = (torch.sum((pred_labels.to(device) == y).int()).item()) / (pred_labels.shape[0]*pred_labels.shape[1])
            
            #accuracy_all.append(accuracy * 100)

        #print("Average Loss on training set at epoch %d : %f" %(epoch, epoch_loss))
        epoch_losses_train.append([epoch_loss])
        
        #print("Accuracy on training set at epoch %d : %f" %(epoch, np.mean(accuracy_all)))
        epoch_accuracy_train = [good_pred_train / num_pred_train * 100]

        #print(epoch_accuracy_train)
        
        with torch.no_grad():
            dev_loss_all = 0
            dev_accuracy_all = []

            good_pred_dev = 0
            num_pred_dev = 0

            for X_dev, y_dev in zip(dev_input, dev_gold):
                X_dev = X_dev.to(device)
                y_dev = y_dev.to(device)
                
                # forward propagation
                log_dev_probs = lstm(X_dev)  
                
                log_dev_probs = log_dev_probs.transpose(1, 2)

                # total loss on the dev set
                dev_loss = loss_function(log_dev_probs, y_dev)

                dev_loss_all += dev_loss.item()

                mask_dev = (y_dev != -100)

                num_pred_dev += mask_dev.int().sum().item()

                pred_dev_labels = torch.argmax(log_dev_probs, dim=1)

                good_pred_dev += (pred_dev_labels.to(device) == y_dev).mul(mask_dev).int().sum().item()

                # prediction and accuracy on the dev set
                #pred_dev_labels = torch.argmax(log_dev_probs, dim=1)
                
                #dev_accuracy = (torch.sum((pred_dev_labels.to(device) == y_dev).int()).item()) / (pred_dev_labels.shape[0]*pred_dev_labels.shape[1])

                #dev_accuracy_all.append(dev_accuracy * 100)

        #print("Loss on dev set at epoch %d : %f" %(epoch, dev_loss_all))
        epoch_losses_dev.append([dev_loss_all])

        
        #print("Accuracy on dev set at epoch %d : %f" %(epoch, np.mean(dev_accuracy_all)))
        epoch_accuracy_dev = [good_pred_dev / num_pred_dev * 100]     
        #print(epoch_accuracy_dev)

        #print(epoch_accuracy_dev)

        if epoch == 0:
            highest_accuracy = epoch_accuracy_dev
            decrease_counter = 0
            torch.save(lstm.state_dict(), save_path)
        else:
            if highest_accuracy > epoch_accuracy_dev:
                decrease_counter += 1
                if decrease_counter > 10:
                    break
            else:
                highest_accuracy = epoch_accuracy_dev
                decrease_counter = 0
                torch.save(lstm.state_dict(), save_path)

    lstm.load_state_dict(torch.load(save_path))
        
    with torch.no_grad():
        test_loss_all = 0
        test_accuracy_all = []

        num_pred_test = 0
        good_pred_test = 0

        for X_test, y_test in zip(test_input, test_gold):
            X_test = X_test.to(device)
            y_test = y_test.to(device)

            # forward propagation
            log_test_probs = lstm(X_test)  
            
            log_test_probs = log_test_probs.transpose(1, 2)


            # total loss on the dev set
            test_loss = loss_function(log_test_probs, y_test)

            test_loss_all += test_loss.item()

            mask_test = (y_test != -100)

            num_pred_test += mask_test.int().sum().item()

            pred_test_labels = torch.argmax(log_test_probs, dim=1)

            good_pred_test += (pred_test_labels.to(device) == y_test).mul(mask_test).int().sum().item()

            # prediction and accuracy on the dev set
            #pred_test_labels = torch.argmax(log_test_probs, dim=1)
            
            #test_accuracy = (torch.sum((pred_test_labels.to(device) == y_test).int()).item()) / (pred_test_labels.shape[0]*pred_test_labels.shape[1])

            #test_accuracy_all.append(test_accuracy * 100)

    test_accuracy_all = [good_pred_test / num_pred_test * 100]

    return epoch_losses_train, [epoch_accuracy_train], epoch_losses_dev, [highest_accuracy], [[test_loss_all]], [test_accuracy_all]




