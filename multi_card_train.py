# python -m torch.distributed.launch --nproc_per_node=4 --use_env example_2.py


import argparse
import os

import torch
import yaml

from train.train import train
from pytorch_lightning.accelerators import CPUAccelerator, CPUAccelerator, GPUAccelerator
from pytorch_lightning.strategies import DDPStrategy



def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


if __name__ == "__main__":
    # this ensures that the current MacOS version is at least 12.3+
    print('MPS backend available?: ', torch.backends.mps.is_available())
    # this ensures that the current current PyTorch installation was built with MPS activated.
    print('MPS backend built?: ', torch.backends.mps.is_built())

    Xccelerate = CPUAccelerator
    ddp = DDPStrategy(accelerator=Xccelerate, process_group_backend='gloo')

    dtype = torch.float
    device = torch.device("mps")

    os.environ['GLOO_LL_THRESHOLD'] = '0'
    # os.environ['NCCL_LL_THRESHOLD'] = '0'

    MASTER_PORT = find_free_port()
    os.environ['MASTER_PORT'] = MASTER_PORT
    print(f"{MASTER_PORT}")
    os.environ['MASTER_ADDR'] = '192.168.1.131'
    print(f"192.168.1.131")

    os.environ["WORLD_SIZE"] = '1'
    os.environ["RANK"] = '0'
    os.environ["LOCAL_RANK"] = '0'


    # os.environ['config_batch'] = 'config/{amazon,yelp}_{TransEnc,BERT}_pretrain.yml'

    parser = argparse.ArgumentParser(description='Train model on multiple cards')
    parser.add_argument('--config', help='path to yaml config file')
    parser.add_argument('--local_rank', type=int, help='local gpu id')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    config['local_rank'] = args.local_rank
    # torch.cuda.set_device(args.local_rank)
    # torch.distributed.init_process_group(backend='gloo', init_method='env://')

    train(config)

    # torch.distributed.init_process_group(backend='nccl', init_method='env://')
    # uppose we run our training in 2 servers or nodes and each with 4 GPUs.
    # The world size is 4*2=8. The ranks for the processes will be [0, 1, 2, 3, 4, 5, 6, 7].
    # In each node, the local rank will be [0, 1, 2, 3].
    # https: // stackoverflow.com / questions / 58271635 / in -distributed - computing - what - are - world - size - and -rank

