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
    --test-path=<str>                 file path of the test set [default: ]
"""
import argparse

from learner import EvaluateOnTest
from model import OodModel
from data_loader import DataClass
from torch.utils.data import DataLoader
import torch
from docopt import docopt
import numpy as np

# args = docopt(__doc__)

device = torch.device('cpu')
parser = argparse.ArgumentParser(description='Train model on multiple cards')
parser.add_argument('--test_path', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--max_length', type=str)
parser.add_argument('--bert_type', type=str)

args = parser.parse_args()


# args['--models-path'] = '2022-07-18-16:08:28_checkpoint.pt'
device = torch.device('cpu')
if str(device) == 'mps:0':
    print("Currently using GPU: {}".format(device))
    np.random.seed(int(args['--seed']))
    torch.manual_seed_all(int(args['--seed']))
    torch.backends.deterministic = True
    torch.backends.benchmark = False
else:
    print("Currently using CPU")
#####################################################################
# Define Dataloaders
#####################################################################

test_dataset = DataClass(args, vars(args)['test_path'])
test_data_loader = DataLoader(test_dataset,
                              batch_size=int(args['test_batch_size']),
                              shuffle=False)
print('The number of Test batches: ', len(test_data_loader))
#############################################################################
# Run the models on a Test set
#############################################################################
model = OodModel(model_type=args['--bert-type'])
learn = EvaluateOnTest(model, test_data_loader, model_path='/Users/jordanharris/SCAPT-ABSA/AspectDetecter/Out_of_domain_ABSA/models/' + args['--models-path'])
learn.predict(device=device)
