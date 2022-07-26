"""
Usage:
    main.py [options]

Options:
    -h --help                         show this screen
    --models-path=<str>                path of the trained models
    --max-length=<int>                text length [default: 128]
    --seed=<int>                      seed [default: 0]
    --test-batch-size=<int>           batch size [default: 32]
    --bert-type=<str>                      language choice [default: base-bert]
    --real-test-path=<str>                 file path of the real test set [default: ]
    --fake-test-path=<str>              file path of the fake test set [default: ]
"""
import argparse

from learner import PredictTest
from model import OodModel
from data_loader import DataClass
from torch.utils.data import DataLoader
import torch
from docopt import docopt
import numpy as np

# args = docopt(__doc__)

device = torch.device('cpu')

parser = argparse.ArgumentParser(description='Train model on multiple cards')
parser.add_argument('--real_test_path', type=str)
parser.add_argument('--fake_test_path', type=str)

parser.add_argument('--model_path', type=str)
parser.add_argument('--max_length', type=str)
parser.add_argument('--bert_type', type=str)
parser.add_argument('--test_batch_size', type=int)
parser.add_argument('--lang', type=str)
parser.add_argument('--num_classes', type=int)


args = parser.parse_args()


device = torch.device('cpu')
if str(device) == 'mps:0':
    print("Currently using GPU: {}".format(device))
    np.random.seed(int(vars(args)['--seed']))
    torch.manual_seed_all(int(vars(args)['--seed']))
    torch.backends.deterministic = True
    torch.backends.benchmark = False
else:
    print("Currently using CPU")
#####################################################################
# Define Dataloaders
#####################################################################
real_test_dataset = DataClass(args, vars(args)['real_test_path'])
fake_test_dataset = DataClass(args, vars(args)['fake_test_path'])
real_test_data_loader = DataLoader(real_test_dataset,
                                   batch_size=int(vars(args)['test_batch_size']),
                                   shuffle=False)
fake_test_data_loader = DataLoader(fake_test_dataset,
                                   batch_size=int(vars(args)['test_batch_size']),
                                   shuffle=False)

print('The number of Test batches: ', len(fake_test_data_loader))
#############################################################################
# Run the models on a Test set
#############################################################################
model = OodModel(model_type=vars(args)['bert_type'])
learn = PredictTest(model, fake_test_data_loader, real_test_data_loader, model_path='models/' + vars(args)['model_path'])
learn.predict(device=device)
