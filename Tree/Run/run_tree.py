import csv
import torch
from pathlib import Path
import sys
import torch
from tqdm import tqdm
from torch import manual_seed
from random import seed
import os


os.path.join(os.path.dirname(__file__), '../')
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from Train_models.train_lstm import w_lstm



seed(5)
manual_seed(5)

#execlp('python3.9', 'python3.9', '/home/tbidewell/home/POS_tagging/code/scripts/Tree/run_tree.py', destination, mod, parent_model, train, dev, test, gpu)


destination = sys.argv[1]
model = sys.argv[2]
parent_model = sys.argv[3]
train = sys.argv[4]
dev = sys.argv[5]
test = sys.argv[6]
gpu = int(sys.argv[7])
device = torch.device(gpu)


models = {
    #'w_ch_lstm': w_ch_lstm,
    'w_lstm': w_lstm, 
    #'transformer': transformer
}

model = models[model]


path = Path(destination)
path.mkdir(parents=True)

epoch_losses_train, epoch_accuracy_train, epoch_losses_dev, epoch_accuracy_dev, test_loss_all, test_accuracy_all = model(path, parent_model, train, dev, test, device)


with open(destination + "/" + model.__name__ + "/" + "epoch_losses_train.csv", 'w', newline = '') as f:
    write = csv.writer(f)
    write.writerow(["Loss"])
    write.writerows(epoch_losses_train)

with open(destination + "/" + model.__name__ + "/" + "epoch_accuracy_train.csv", 'w', newline = '') as f:
    write = csv.writer(f)
    write.writerow(["Accuracy"])
    write.writerows(epoch_accuracy_train)

with open(destination + "/" + model.__name__ + "/" + "epoch_losses_dev.csv", 'w', newline = '') as f:
    write = csv.writer(f)
    write.writerow(["Loss"])
    write.writerows(epoch_losses_dev)

with open(destination + "/" + model.__name__ + "/" + "epoch_accuracy_dev.csv", 'w', newline = '') as f:
    write = csv.writer(f)
    write.writerow(["Accuracy"])
    write.writerows(epoch_accuracy_dev)

with open(destination + "/" + model.__name__ + "/" + "test_loss_all.csv", 'w', newline = '') as f:
    write = csv.writer(f)
    write.writerow(["Loss"])
    write.writerows(test_loss_all)


with open(destination + "/" + model.__name__ + "/" + "test_accuracy_all.csv", 'w', newline = '') as f:
    write = csv.writer(f)
    write.writerow(["Accuracy"])
    write.writerows(test_accuracy_all)

