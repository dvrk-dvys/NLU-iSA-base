"""
Usage:
    main.py [options]

Options:
    -h --help                         show this screen
    --bert-type=<str>                 type of bert [default: BERT]
    --model-path=<str>                path of the trained model
    --max-length=<int>                text length [default: 128]
    --seed=<int>                      seed [default: 0]
    --test-batch-size=<int>           batch size [default: 32]
    --lang=<str>                      language choice [default: English]
    --test-path=<str>                 file path of the test set [default: ]
"""
from learner import EvaluateOnTest
from model import SpanEmo
from data_loader import DataClass
from torch.utils.data import DataLoader
import torch
from docopt import docopt
import numpy as np
import argparse

# args = docopt(__doc__)
device = torch.device('cpu')

parser = argparse.ArgumentParser(description='Train model on multiple cards')
parser.add_argument('--test_path', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--max_length', type=str)
parser.add_argument('--bert_type', type=str)
parser.add_argument('--test_batch_size', type=int)
parser.add_argument('--lang', type=str)
args = parser.parse_args()


if str(device) == 'mps:0':
    print("Currently using GPU: {}".format(device))
    np.random.seed(int(args['--seed']))
    torch.cuda.manual_seed_all(int(args['--seed']))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    print("Currently using CPU")
#####################################################################
# Define Dataloaders
#####################################################################
test_dataset = DataClass(args, vars(args)['test_path'])
test_data_loader = DataLoader(test_dataset,
                              batch_size=int(vars(args)['test_batch_size']),
                              shuffle=False)
print('The number of Test batches: ', len(test_data_loader))
#############################################################################
# Run the model on a Test set
#############################################################################
model = SpanEmo(lang=vars(args)['lang'], bert_type=vars(args)['bert_type'])
# learn = EvaluateOnTest(model, test_data_loader, model_path='models/' + args['--model-path'])
learn = EvaluateOnTest(model, test_data_loader, model_path=vars(args)['model_path'])
learn.predict(device=device)
