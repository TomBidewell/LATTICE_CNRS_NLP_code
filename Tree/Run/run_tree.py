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
from Train_models.train import train_model
from tqdm import tqdm
from argparse import ArgumentParser

seed(5)
manual_seed(5)

ap = ArgumentParser()
ap.add_argument('destination', help='Path to where metrics are stored file.')
ap.add_argument('current_model', help = 'Model path for this node')
ap.add_argument('parent_model', help = 'Model path for parent')
ap.add_argument('train', help= 'Path to train file')
ap.add_argument('dev', help = 'Path to dev file')
ap.add_argument('test', help = 'Path to test file')
ap.add_argument('device', help = 'Device Used')
ap.add_argument('number_of_repeats', help = 'Number of repetitions of each model')
args = ap.parse_args()

device = torch.device(int(args.device))

use_transformer = False

if args.current_model == "transformer":
    use_transformer = True


path = Path(args.destination + "/" + args.current_model)
try: 
    path.mkdir(parents=True)
except: 
    ()

all_epoch_losses_train = []
all_epoch_accuracy_train = [] 
all_epoch_losses_dev = [] 
all_epoch_accuracy_dev = [] 
all_test_loss_all = [] 
all_test_accuracy_all = []

for i in tqdm(range(int(args.number_of_repeats)), total = int(args.number_of_repeats), desc = 'Repetitions: '):
    epoch_losses_train, epoch_accuracy_train, epoch_losses_dev, epoch_accuracy_dev, test_loss_all, test_accuracy_all = train_model(path, use_transformer, args.parent_model, args.current_model, args.train, args.dev, args.test, device)
    all_epoch_losses_train.append(epoch_losses_train)
    all_epoch_accuracy_train.append(epoch_accuracy_train)
    all_epoch_losses_dev.append(epoch_losses_dev)
    all_epoch_accuracy_dev.append(epoch_accuracy_dev)
    all_test_loss_all.append(test_loss_all)
    all_test_accuracy_all.append(test_accuracy_all)


with open(args.destination + "/" + args.current_model + "/" + "epoch_losses_train.csv", 'w', newline = '') as f:
    write = csv.writer(f)
    write.writerow(["Loss"])
    write.writerows(all_epoch_losses_train)

with open(args.destination + "/" + args.current_model + "/" + "epoch_accuracy_train.csv", 'w', newline = '') as f:
    write = csv.writer(f)
    write.writerow(["Accuracy"])
    write.writerows(all_epoch_accuracy_train)

with open(args.destination + "/" + args.current_model + "/" + "epoch_losses_dev.csv", 'w', newline = '') as f:
    write = csv.writer(f)
    write.writerow(["Loss"])
    write.writerows(all_epoch_losses_dev)

with open(args.destination + "/" + args.current_model + "/" + "epoch_accuracy_dev.csv", 'w', newline = '') as f:
    write = csv.writer(f)
    write.writerow(["Accuracy"])
    write.writerows(all_epoch_accuracy_dev)

with open(args.destination + "/" + args.current_model + "/" + "test_loss_all.csv", 'w', newline = '') as f:
    write = csv.writer(f)
    write.writerow(["Loss"])
    write.writerows(all_test_loss_all)


with open(args.destination + "/" + args.current_model + "/" + "test_accuracy_all.csv", 'w', newline = '') as f:
    write = csv.writer(f)
    write.writerow(["Accuracy"])
    write.writerows(all_test_accuracy_all)
