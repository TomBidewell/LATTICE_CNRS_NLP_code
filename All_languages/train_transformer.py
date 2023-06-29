from transformer_data_prep import transformer_data_prep
from roberta_POS import ROBERTA
from transformers import XLMRobertaConfig,  XLMRobertaModel
from torch import nn, optim
import torch
import numpy as np
from tqdm import tqdm


def transformer(path, train, dev, test, device):

    tensor_dict, num_classes = transformer_data_prep(train, dev, test)

    batches_features_input_train, batches_features_att_train, batches_gold_train = tensor_dict['train']
    batches_features_input_dev, batches_features_att_dev, batches_gold_dev = tensor_dict['dev']
    batches_features_input_test, batches_features_att_test, batches_gold_test = tensor_dict['test']

    hidden_layer_size = 500
    num_epochs = 1000

    print("Model Loading")

    model =  XLMRobertaModel.from_pretrained('xlm-roberta-base')
    model.to(device)

    print("Model Ready")

    roberta_pos = ROBERTA(num_classes, hidden_layer_size) 

    loss_function = nn.NLLLoss()

    optimizer_1 = optim.Adadelta(roberta_pos.parameters())

    roberta_pos.roberta = model

    roberta_pos.to(device)

    optimizer_2 = optim.SGD(roberta_pos.roberta.parameters(), lr = 0.001)

    save_path = path / roberta_pos.__class__.__name__ 


    # to record the total loss at each epoch
    epoch_losses_train = []
    epoch_losses_dev = []
    epoch_accuracy_train = []
    epoch_accuracy_dev = []

    # loop on epochs
    for epoch in tqdm(range(num_epochs), total = num_epochs, desc = 'Transformer: '):
        #print()
        #print("Epoch", epoch)
        epoch_loss = 0
                
        #roberta_pos.eval()

        num_pred_train = 0
        good_pred_train = 0

        for X_in, X_att, y in zip(batches_features_input_train, batches_features_att_train, batches_gold_train):
            X_in = X_in.to(device)
            X_att = X_att.to(device)
            y = y.to(device)
        
            roberta_pos.zero_grad()
            logs_probs = roberta_pos(X_in, X_att)

            logs_probs = logs_probs.transpose(1,2)
            loss = loss_function(logs_probs, y)
            epoch_loss += loss.item()
            loss.backward()

            optimizer_1.step()
            optimizer_2.step()

            mask_train = (y != -100)

            num_pred_train += mask_train.int().sum().item()

            pred_labels = torch.argmax(logs_probs, dim=1)

            good_pred_train += (pred_labels.to(device) == y).mul(mask_train).int().sum().item()


        #print("Average Loss on training set at epoch %d : %f" %(epoch, epoch_loss))
        epoch_losses_train.append([epoch_loss])
        
        #print("Accuracy on training set at epoch %d : %f" %(epoch, np.mean(accuracy_all)))
        epoch_accuracy_train = [good_pred_train / num_pred_train * 100]

        #print(epoch_accuracy_train)
        
        
        with torch.no_grad():
            dev_loss_all = 0

            good_pred_dev = 0
            num_pred_dev = 0

            for X_in_dev, X_att_dev, y_dev  in zip(batches_features_input_dev, batches_features_att_dev, batches_gold_dev):
                
                X_in_dev = X_in_dev.to(device)
                X_att_dev = X_att_dev.to(device)
                y_dev = y_dev.to(device)

                # forward propagation
                log_dev_probs = roberta_pos(X_in_dev,X_att_dev)  


                log_dev_probs = log_dev_probs.transpose(1, 2)
        
            
                # total loss on the dev set
                dev_loss = loss_function(log_dev_probs, y_dev)
            
                dev_loss_all += dev_loss.item()  

                mask_dev = (y_dev != -100)

                num_pred_dev += mask_dev.int().sum().item()

                pred_dev_labels = torch.argmax(log_dev_probs, dim=1)

                good_pred_dev += (pred_dev_labels.to(device) == y_dev).mul(mask_dev).int().sum().item()


        #print("Loss on dev set at epoch %d : %f" %(epoch, dev_loss_all))
        epoch_losses_dev.append([dev_loss_all])

    
        #print("Accuracy on dev set at epoch %d : %f" %(epoch, np.mean(dev_accuracy_all)))
        epoch_accuracy_dev = [good_pred_dev / num_pred_dev * 100]
        #print(epoch_accuracy_dev)
    
        if epoch == 0:
            highest_accuracy = epoch_accuracy_dev
            decrease_counter = 0
            torch.save(roberta_pos.state_dict(), save_path)
        else:
            if highest_accuracy > epoch_accuracy_dev:
                decrease_counter += 1
                if decrease_counter > 10:
                    break
            else:
                highest_accuracy = epoch_accuracy_dev
                decrease_counter = 0
                torch.save(roberta_pos.state_dict(), save_path)


    with torch.no_grad():
      test_loss_all = 0
      test_accuracy_all = []

      num_pred_test = 0
      good_pred_test = 0

      for X_test_word, X_test_char, y_test in zip(batches_features_input_test, batches_features_att_test, batches_gold_test):
        
        X_test_word = X_test_word.to(device)
        X_test_char = X_test_char.to(device)
        y_test = y_test.to(device)
        
        # forward propagation
        log_test_probs = roberta_pos(X_test_word, X_test_char)
        
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

        #test_mask = y_test != -100
                
        #y_test = y_test[test_mask]
        #pred_test_labels = pred_test_labels[test_mask]

        
        #test_accuracy = (torch.sum((pred_test_labels.to(device) == y_test).int()).item()) / (test_mask.int().sum().cpu().numpy())

        #test_accuracy_all.append(test_accuracy * 100)
    test_accuracy_all = [good_pred_test / num_pred_test * 100]
    
    return epoch_losses_train, [epoch_accuracy_train], epoch_losses_dev, [epoch_accuracy_dev], [[test_loss_all]], [test_accuracy_all]



