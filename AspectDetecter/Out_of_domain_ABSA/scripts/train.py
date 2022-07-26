"""
Usage:
    main.py [options]

Options:
    -h --help                         show this screen
    --max-length=<int>                text length [default: 128]
    --output-dropout=<float>          prob of dropout applied to the output layer [default: 0.1]
    --seed=<int>                      fixed random seed number [default: 42]
    --train-batch-size=<int>          batch size [default: 32]
    --eval-batch-size=<int>           batch size [default: 32]
    --max-epoch=<int>                 max epoch [default: 20]
    --ffn-lr=<float>                  ffn learning rate [default: 0.001]
    --bert-lr=<float>                 bert learning rate [default: 2e-5]
    --bert-type=<str>                 language choice [default: base-bert]
    --dev-path=<str>                  file path of the dev set [default: '']
    --train-path=<str>                file path of the train set [default: '']
"""
import argparse
import os

from learner import Trainer
from model import OodModel
from data_loader import DataClass
from torch.utils.data import DataLoader
import torch
from docopt import docopt
import datetime
import json
import numpy as np


# args = docopt(__doc__)
device = torch.device('cpu')
parser = argparse.ArgumentParser(description='Train model on multiple cards')
parser.add_argument('--train_path', type=str)
parser.add_argument('--test_path', type=str)
parser.add_argument('--dev_path', type=str)

parser.add_argument('--model_path', type=str)
parser.add_argument('--bert_type', type=str)
parser.add_argument('--max_length', type=int)

parser.add_argument('--output_dropout')
parser.add_argument('--seed', type=str)
parser.add_argument('--train_batch_size', type=int)
parser.add_argument('--eval_batch_size', type=int)
parser.add_argument('--max_epoch', type=int)
parser.add_argument('--ffn_lr')
parser.add_argument('--bert_lr')
parser.add_argument('--checkpoint_path')


now = datetime.datetime.now()
filename = now.strftime("%Y-%m-%d-%H:%M:%S")
fw = open('/Users/jordanharris/SCAPT-ABSA/AspectDetecter/Out_of_domain_ABSA/configs/' + filename + '.json', 'a')
model_path = filename + '.pt'
# args['--checkpoint-path'] = model_path
os.environ["--checkpoint_path"] = model_path

args = parser.parse_args()



# vals(args)

if str(device) == 'mps':
    print("Currently using GPU: {}".format(device))
    np.random.seed(int(args['--seed']))
    torch.manual_seed_all(int(args['--seed']))
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
else:
    print("Currently using CPU")
#####################################################################
# Save hyper-parameter values ---> config.json
# Save models weights ---> filename.pt using current time
#####################################################################
# now = datetime.datetime.now()
# filename = now.strftime("%Y-%m-%d-%H:%M:%S")
# fw = open('/Users/jordanharris/SCAPT-ABSA/AspectDetecter/Out_of_domain_ABSA/configs/' + filename + '.json', 'a')
# model_path = filename + '.pt'
# args['--checkpoint-path'] = model_path

# json.dump(args, fw, sort_keys=True, indent=2)
#####################################################################
# Define Dataloaders
#####################################################################
train_dataset = DataClass(args, vars(args)['train_path'])
train_data_loader = DataLoader(train_dataset,
                               batch_size=int(vars(args)['train_batch_size']),
                               shuffle=True
                               )
print('The number of training batches: ', len(train_data_loader))
dev_dataset = DataClass(args, vars(args)['dev_path'])
dev_data_loader = DataLoader(dev_dataset,
                             batch_size=int(vars(args)['eval_batch_size']),
                             shuffle=False
                             )
print('The number of validation batches: ', len(dev_data_loader))
#############################################################################
# Define Model & Training Pipeline
#############################################################################
model = OodModel(output_dropout=float(vars(args)['output_dropout']),
                model_type=vars(args)['bert_type'])
#############################################################################
# Start Training
#############################################################################
learn = Trainer(model, train_data_loader, dev_data_loader, file_name=filename)
learn.fit(
    num_epochs=int(vars(args)['max_epoch']),
    args=args,
    device=device
)