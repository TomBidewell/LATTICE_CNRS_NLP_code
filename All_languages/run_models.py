import csv
import torch
from pathlib import Path
import sys
import torch
from train_word_char_lstm import w_ch_lstm
from train_word_lstm import w_lstm
from train_transformer import transformer
from tqdm import tqdm
from torch import manual_seed
from random import seed

seed(5)
manual_seed(5)


destination = sys.argv[1]
train = sys.argv[2]
dev = sys.argv[3]
test = sys.argv[4]
device = int(sys.argv[5])
device = torch.device(device)
model = sys.argv[6]


models = {
    'w_ch_lstm': w_ch_lstm,
    'w_lstm': w_lstm, 
    'transformer': transformer
}

model = models[model]

path = Path(destination + "/" + model.__name__ )
path.mkdir(parents=True)


epoch_losses_train, epoch_accuracy_train, epoch_losses_dev, epoch_accuracy_dev, test_loss_all, test_accuracy_all = model(path, train, dev, test, device)


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
