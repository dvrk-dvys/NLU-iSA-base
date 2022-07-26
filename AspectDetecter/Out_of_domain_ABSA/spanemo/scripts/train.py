"""
Usage:
    main.py [options]

Options:
    -h --help                         show this screen
    --bert-type=<str>                 type of bert [default: BERT]
    --loss-type=<str>                 Which loss to use cross-ent|corr|joint. [default: cross-entropy]
    --max-length=<int>                text length [default: 128]
    --output-dropout=<float>          prob of dropout applied to the output layer [default: 0.1]
    --seed=<int>                      fixed random seed number [default: 42]
    --train-batch-size=<int>          batch size [default: 32]
    --eval-batch-size=<int>           batch size [default: 32]
    --max-epoch=<int>                 max epoch [default: 20]
    --ffn-lr=<float>                  ffn learning rate [default: 0.001]
    --bert-lr=<float>                 bert learning rate [default: 2e-5]
    --lang=<str>                      language choice [default: English]
    --dev-path=<str>                  file path of the dev set [default: '']
    --train-path=<str>                file path of the train set [default: '']
    --alpha-loss=<float>              weight used to balance the loss [default: 0.2]

"""
import argparse
import parser

from learner import Trainer
from model import SpanEmo
from data_loader import DataClass
from torch.utils.data import DataLoader
import torch
from docopt import docopt
import datetime
import json
import numpy as np


# args = docopt(__doc__)

parser = argparse.ArgumentParser(description='Train model on multiple cards')
parser.add_argument('--train-path', type=str)
parser.add_argument('--dev-path', type=str)
parser.add_argument('--loss-type')
parser.add_argument('--max-length', type=int)
parser.add_argument('--output-dropout')
parser.add_argument('--seed', type=int)
parser.add_argument('--train-batch-size', type=int)
parser.add_argument('--eval-batch-size', type=int)
parser.add_argument('--max-epoch', type=int)
parser.add_argument('--ffn-lr')
parser.add_argument('--bert-lr')
parser.add_argument('--lang', type=str)
parser.add_argument('--alpha-loss', type=str)
parser.add_argument('--bert-type', type=str)
parser.add_argument('--checkpoint-path', type=str)

args = parser.parse_args()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if str(device) == 'cuda:0':
    print("Currently using GPU: {}".format(device))
    np.random.seed(int(args['--seed']))
    torch.cuda.manual_seed_all(int(args['--seed']))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    print("Currently using CPU")
#####################################################################
# Save hyper-parameter values ---> config.json
# Save model weights ---> filename.pt using current time
#####################################################################
now = datetime.datetime.now()
filename = now.strftime("%Y-%m-%d-%H:%M:%S")
fw = open('configs/' + filename + '.json', 'a')
model_path = filename + '.pt'
# args['--checkpoint-path'] = model_path
args.checkpoint_path = model_path
json.dump(vars(args), fw, sort_keys=True, indent=2)
#####################################################################
# Define Dataloaders
#####################################################################
# train_dataset = DataClass(args, args['--train-path'])
# batch_size = int(args['--train-batch-size']),

train_dataset = DataClass(args, args.train_path)
train_data_loader = DataLoader(train_dataset,
                               batch_size=int(args.train_batch_size),
                               shuffle=True
                               )
print('The number of training batches: ', len(train_data_loader))
dev_dataset = DataClass(args, args.dev_path)
dev_data_loader = DataLoader(dev_dataset,
                             batch_size=int(args.eval_batch_size),
                             shuffle=False
                             )
print('The number of validation batches: ', len(dev_data_loader))
#############################################################################
# Define Model & Training Pipeline
#############################################################################
model = SpanEmo(output_dropout=float(args.output_dropout),
                lang=args.lang,
                joint_loss=args.loss_type,
                alpha=float(args.alpha_loss),
                bert_type=args.bert_type)
model_dict = torch.load("/Users/jordanharris/SCAPT-ABSA/AspectDetecter/Out_of_domain_ABSA/SpanEmo/models/SemEval_2022-07-26-15:40:26_checkpoint.pt")
model.load_state_dict(state_dict=model_dict)
#############################################################################
# Start Training
#############################################################################
learn = Trainer(model, train_data_loader, dev_data_loader, filename=filename)
learn.fit(
    num_epochs=int(args.max_epoch),
    args=args,
    device=device
)
