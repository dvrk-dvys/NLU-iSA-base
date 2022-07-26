import os

from train.misc import set_seed, set_device, prepare_dataset
from train.trainer.aspect_finetune import aspect_finetune, finetune_Lite, collate_fn, ABSADataset as ABSADataset_FT
from train.trainer.pretrain import pretrain_Lite, create_dataloader #,fp16_multi_pretrain

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.accelerators import CPUAccelerator
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch

Xccelerate = CPUAccelerator


def train(config):
    set_seed(config['seed'])
    if config['mode'] == 'fp16_multi_pretrain':
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Callbacks
        save_ckpt_path = '/Users/jordanharris/SCAPT-ABSA/checkpoints/' + config['model_path'].split('/')[-1] + '_' + config['model'] + '/'
        checkpoint_callback = ModelCheckpoint(dirpath=save_ckpt_path,
                                              save_top_k=-1,
                                              monitor="pretrain_loss",
                                              every_n_train_steps=600,
                                              train_time_interval=None,
                                              auto_insert_metric_name=True,
                                              # save_on_train_epoch_end=True,
                                              save_last=True,
                                              verbose=True)
        early_stop_callback = EarlyStopping(monitor="pretrain_loss", patience=2, verbose=True, mode="max", min_delta=.01)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!!!!~~~~~~~~~~Path Of Model~~~~~~~~!!!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        model = pretrain_Lite(num_labels=3,
                              task_name=config['mode'],
                              learning_rate=config['learning_rate'],
                              adam_epsilon=1e-8,
                              warmup_steps=0,
                              weight_decay=config['weight_decay'],
                              train_batch_size=config['batch_size'],
                              eval_batch_size=config['batch_size'],
                              config=config).to(device=torch.device('cpu'))
            # .load_from_checkpoint(config['ckpt'])

        trainer = pl.Trainer(accelerator='cpu',
                            max_epochs=8,
                            min_epochs=6,
                            callbacks=[RichProgressBar(refresh_rate=1,), early_stop_callback, checkpoint_callback],
                            overfit_batches=0.015,
                            # overfit_batches=3,
                            auto_scale_batch_size="binsearch",
                            enable_progress_bar=True,
                            log_every_n_steps=10,
                            precision=32,
                            amp_backend="native"
        )

        dl = create_dataloader(config)
        ckpt = config['ckpt']
        if ckpt != '':
            ckpt = config['ckpt']
            # checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
            trainer.fit(model=model, train_dataloaders=dl, ckpt_path=ckpt)
        elif config['state_dict'] != '':
            state_dict = config['state_dict']
            # model = model.load_state_dict(torch.load(state_dict, map_location=torch.device('cpu')))
            trainer.fit(model=model, train_dataloaders=dl)
        else:
            trainer.fit(model=model, train_dataloaders=dl)


        model_path = 'models_weights/'
        model_file = config['model_path'].split('/')[-1] + '_' + config['model'] + '_state_dict.pth'
        print('Final Model State Dict Saved Path: ', os.path.join(model_path, model_file))
        torch.save(model.state_dict(), os.path.join(model_path, model_file))
        model_file = config['model_path'].split('/')[-1] + '_' + config['model'] + '_best_model.pth'
        print('Final (BEST) Model Saved Path: ', os.path.join(model_path, model_file))
        torch.save(checkpoint_callback.best_model_path, os.path.join(model_path, model_file))
        return model
    # return fp16_multi_pretrain(config)


    if config['mode'] == 'aspect_finetune':
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Callbacks
        save_ckpt_path = '/Users/jordanharris/SCAPT-ABSA/checkpoints/' + config['model_path'].split('/')[-1] + '_' + config['model'] + '/'
        checkpoint_callback = ModelCheckpoint(dirpath=save_ckpt_path,
                                              save_top_k=-1,
                                              monitor="total_loss",
                                              every_n_train_steps=600,
                                              train_time_interval=None,
                                              auto_insert_metric_name=True,
                                              # save_on_train_epoch_end=True,
                                              save_last=True,
                                              verbose=True)
        early_stop_callback = EarlyStopping(monitor="total_loss", patience=2, verbose=True, mode="max", min_delta=.01)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!!!!~~~~~~~~~~Path Of Model~~~~~~~~!!!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        train_loader, dev_loader, test_loader = prepare_dataset(config,
                                                                absa_dataset=ABSADataset_FT,
                                                                collate_fn=collate_fn)
        loaders = [train_loader, dev_loader, test_loader]


        model = finetune_Lite(num_labels=3,
                              task_name=config['mode'],
                              learning_rate=config['learning_rate'],
                              adam_epsilon=1e-8,
                              warmup_steps=0,
                              weight_decay=config['weight_decay'],
                              train_batch_size=config['batch_size'],
                              eval_batch_size=config['batch_size'],
                              loaders=loaders,
                              config=config).to(device=torch.device('cpu'))


        trainer = pl.Trainer(accelerator='cpu',
                             max_epochs=8,
                             min_epochs=6,
                             devices='auto',
                             callbacks=[RichProgressBar(refresh_rate=1, ), early_stop_callback, checkpoint_callback],
                             # overfit_batches=0.015,
                             # overfit_batches=100,
                             auto_scale_batch_size="binsearch",
                             enable_progress_bar=True,
                             log_every_n_steps=10,
                             precision=32,
                             amp_backend="native"
        )

        # dl = create_dataloader(config)  # (os.cpu_count()/2))
        ckpt = config['ckpt']

        if ckpt != '':
            trainer.fit(model=model, train_dataloaders=train_loader, ckpt_path=ckpt)
        else:
            trainer.fit(model=model, train_dataloaders=train_loader)

        model_path = 'models_weights/'
        model_file = config['model_path'].split('/')[-1] + '_' + config['model'] + '_state_dict.pth'
        print('Final FT Model State Dict Saved Path: ', os.path.join(model_path, model_file))
        torch.save(model.state_dict(), os.path.join(model_path, model_file))
        model_file = config['model_path'].split('/')[-1] + '_' + config['model'] + '_best_model.pth'
        print('Final FT (BEST) Model Saved Path: ', os.path.join(model_path, model_file))
        torch.save(checkpoint_callback.best_model_path, os.path.join(model_path, model_file))
        return model

        # return aspect_finetune(config)
    raise TypeError(f"Not supported train mode {config['mode']}")

# test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
# if config['mode'] == 'aspect_finetune':
#     loaded_dict = checkpoint['model_state_dict']
#     prefix = 'classifier.'
#     n_clip = len(prefix)
#     adapted_dict = {k[n_clip:]: v for k, v in loaded_dict.items()
#                     if k.startswith(prefix)}
#     model.load_state_dict(adapted_dict)
