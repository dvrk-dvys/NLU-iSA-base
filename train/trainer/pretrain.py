import copy
import os
import pickle
import time

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.init import xavier_uniform_
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

#______HuggingFace Transformers & Accelerators________________________
from transformers import (
    BertTokenizer, DistilBertTokenizerFast,
    EncoderDecoderModel,
    get_linear_schedule_with_warmup,
)
from accelerate import Accelerator as HF_Accelerator

#______Lightning________________________
from pytorch_lightning.accelerators import CPUAccelerator
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import LightningModule, Trainer, seed_everything

#___________Local________________________
from model.transformer_decoder import TransformerDecoder, Generator, TransformerDecoderState
from train.misc import get_model_path, get_masked_inputs_and_labels
from train.model import build_absa_model
from train.model import SupConLoss
from train.optimizer import build_optim, build_optim_bert

Xccelerate = CPUAccelerator
ddp = DDPStrategy(accelerator=Xccelerate)
print(Xccelerate.get_device_stats(Xccelerate, 0))

XLdevice = HF_Accelerator.device
dtype = torch.float
device = torch.device("cpu")

HF_Accelerator = HF_Accelerator()

class ABSADataset(Dataset):
    def __init__(self, path):
        super(ABSADataset, self).__init__()
        data = pickle.load(open(path, 'rb'))
        self.raw_texts = data['raw_texts']
        if 'finetune' not in path:
            self.raw_nested_aspect_terms = data['raw_nested_aspect_terms']
        self.len = len(data['labels'])
        self.bert_tokens = [torch.FloatTensor(bert_token) for bert_token in data['bert_tokens']]
        self.aspect_masks = [torch.FloatTensor(bert_mask) for bert_mask in data['aspect_masks']]
        self.labels = torch.FloatTensor(data['labels'])

    def __getitem__(self, index):
        return (self.bert_tokens[index],
                self.aspect_masks[index],
                self.labels[index],
                self.raw_texts[index],
                self.raw_nested_aspect_terms[index])
    def __len__(self):
        return self.len


def collate_fn(batch):
    bert_tokens, aspect_masks, labels, raw_texts, raw_nested_aspect_terms = zip(*batch)
    bert_masks = pad_sequence([torch.ones(tokens.shape, device=device, dtype=dtype) for tokens in bert_tokens], batch_first=True)
    bert_tokens = pad_sequence(bert_tokens, batch_first=True)
    aspect_masks = pad_sequence(aspect_masks, batch_first=True)
    labels = torch.stack(labels)

    bert_tokens_masked, masked_labels = get_masked_inputs_and_labels(bert_tokens.to(device='cpu'), aspect_masks.to(device='cpu'))
    bert_tokens_masked = torch.FloatTensor(bert_tokens_masked)
    masked_labels = torch.FloatTensor(masked_labels)

    return (bert_tokens_masked,
            masked_labels,
            bert_tokens,
            bert_masks,
            aspect_masks,
            labels,
            raw_texts,
            raw_nested_aspect_terms)


def has_opposite_labels(labels):
    return not (labels.sum().item() <= 1 or (1 - labels).sum().item() <= 1)


def set_parameter_linear(p):
    if p.dim() > 1:
        xavier_uniform_(p)
    else:
        p.data.zero_()


def set_parameter_tf(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

def create_dataloader(config):
    data_path = config['data_path']
    if config['mode'] == 'aspect_finetune':
        pretrain_path = os.path.join(data_path, config['train_file'])
    else:
        pretrain_path = os.path.join(data_path, config['pretrain_file'])

    pretrain_dataset = ABSADataset(pretrain_path)
    global_rank = ddp.global_rank
    # sampler = DistributedSampler(pretrain_dataset, num_replicas=ddp.world_size, rank=config['local_rank'])
    # sampler = DistributedSampler(pretrain_dataset, num_replicas=4, rank=config['local_rank'])

    print("Card {} start training".format(global_rank) + ' ' + pretrain_path)
    pretrain_loader = DataLoader(pretrain_dataset,
                                 batch_size=config['batch_size'],
                                 # sampler=sampler,
                                 collate_fn=collate_fn,
    )
    pretrain_loader = HF_Accelerator.prepare_data_loader(pretrain_loader)
    return pretrain_loader


class pretrain_Lite(LightningModule):
    def __init__(
            self,
            # model_name_or_path: str,
            num_labels: int,
            task_name: str,
            learning_rate: float = 2e-5,
            adam_epsilon: float = 1e-8,
            warmup_steps: int = 0,
            weight_decay: float = 0.0,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            config: dict = None,
            optim: list = None,
            # dec_gen_paths: list = None,
            # eval_splits: Optional[list] = None,

    ):
        super().__init__()
        self.optim = optim
        self.model = build_absa_model(config)
        self.config = config
        self.hparams[learning_rate] = config['learning_rate']
        self.hparams[warmup_steps] = config['warm_up']
        self.hparams[weight_decay] = config['weight_decay']
        self.save_hyperparameters()
        self.task_name = task_name
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.tgt_embeddings = nn.Embedding(len(self.tokenizer.vocab), self.model.config.hidden_size, padding_idx=0)
        self.decoder = TransformerDecoder(
            config['decoder_layers'],
            config['decoder_hidden'],
            heads=config['decoder_heads'],
            d_ff=config['decoder_ff'],
            dropout=config['decoder_dropout'],
            embeddings=self.tgt_embeddings
        )
        self.generator = Generator(len(self.tokenizer.vocab), config['decoder_hidden'], self.tokenizer.vocab['[PAD]'])
        self.generator.linear.weight = self.decoder.embeddings.weight

        if config['state_dict'] != '':
            model_dict = torch.load(config['state_dict'], map_location=torch.device('cpu'))
            self.model.load_state_dict(state_dict=model_dict)
            self.model = self.model.to(torch.device('cpu'))
        elif config['gen_state_dict'] != '':
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Decoder Load
            decoder_dict = torch.load(config['dec_state_dict'])
            self.decoder.load_state_dict(state_dict=decoder_dict)
            self.decoder = self.decoder.to(device=torch.device('cpu'))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Generator Load
            generator_dict = torch.load(config['gen_state_dict'])
            self.generator.load_state_dict(state_dict=generator_dict)
            self.generator = self.generator.to(device=torch.device('cpu'))


        self.similar_criterion = SupConLoss()
        self.reconstruction_criterion = nn.NLLLoss(ignore_index=self.tokenizer.vocab['[PAD]'], reduction='mean')
        self.avg_mlm_loss = 0.
        self.avg_similar_loss = 0.
        self.avg_reconstruction_loss = 0.
        self.avg_loss = 0.
        self.current_step = 0
        self.pretrain_loss = 0


        for p in self.decoder.modules():
            set_parameter_tf(p)
        for p in self.generator.parameters():
            set_parameter_tf(p)

        if "share_emb" in config:
            if 'BERT' in config['model']:
                self.tgt_embeddings = nn.Embedding(len(self.tokenizer.vocab), self.model.config.hidden_size, padding_idx=0)
                self.tgt_embeddings.weight = copy.deepcopy(self.model.bert.embeddings.word_embeddings.weight)
            self.decoder.embeddings = self.tgt_embeddings
            self.generator.linear.weight = self.decoder.embeddings.weight
        if 'cache' in config:
            ddp.barrier()
            # configure map_location properly
            map_location = {'cpu:%d' % 0: 'cpu:%d' % config['local_rank']}
            model_dict = torch.load(config['cache'], map_location=map_location)
            decoder_dict = torch.load(config['decoder_cache'], map_location=map_location)
            generator_dict = torch.load(config['generator_cache'], map_location=map_location)
            self.model.load_state_dict(model_dict)
            self.decoder.load_state_dict(decoder_dict)
            self.generator.load_state_dict(generator_dict)

        if ddp.global_rank == 0:
            self.model_path = get_model_path(config['model'], config['model_path'])

    def forward(self, x):
        return torch.relu(self.model(x.view(x.size(0), -1)))


    def training_step(self, batch, batch_idx, optimizer_idx):

        # _______________________HuggingFace Accelerator________________________
        self.model = HF_Accelerator.prepare_model(self.model)
        self.decoder = HF_Accelerator.prepare_model(self.decoder)
        self.generator = HF_Accelerator.prepare_model(self.generator)
        # _______________________HuggingFace Accelerators________________________

        bert_tokens_masked, masked_labels, bert_tokens, bert_masks, aspect_masks, labels, raw_texts, raw_nested_aspect_terms = batch

        self.model.train()
        self.decoder.train()
        self.generator.train()
        self.model.zero_grad()
        self.decoder.zero_grad()
        self.generator.zero_grad()
        self.current_step += 1

        bert_tokens_masked = bert_tokens_masked.to(dtype=torch.long)
        masked_labels = masked_labels.to( dtype=torch.long)
        bert_tokens = bert_tokens.to( dtype=torch.long)
        aspect_masks = aspect_masks.to( dtype=torch.long)
        bert_masks = bert_masks.to( dtype=torch.long)
        labels = labels.to( dtype=torch.long)

        if has_opposite_labels(labels):
            cls_hidden, hidden_state, masked_lm_loss = self.model(
                input_ids=bert_tokens_masked,
                attention_mask=bert_masks,
                aspect_mask=aspect_masks,
                labels=masked_labels,
                return_dict=True,
                multi_card=True,
                has_opposite_labels=True
            )

        else:
            hidden_state, masked_lm_loss = self.model(
                input_ids=bert_tokens_masked,
                attention_mask=bert_masks,
                aspect_mask=aspect_masks,
                labels=masked_labels,
                return_dict=True,
                multi_card=True,
                has_opposite_labels=False
            )


        decode_context = self.decoder(bert_tokens_masked[:, :-1], hidden_state,
                                 TransformerDecoderState(bert_tokens_masked))

        reconstruction_text = self.generator(decode_context.view(-1, decode_context.size(2)))

        reconstruction_loss = self.reconstruction_criterion(reconstruction_text, bert_tokens[:, 1:].reshape(-1))

        if not has_opposite_labels(labels):
            loss = self.config['lambda_map'] * masked_lm_loss + \
                   self.config['lambda_rr'] * reconstruction_loss
        else:
            normed_cls_hidden = F.normalize(cls_hidden, dim=-1)
            similar_loss = self.similar_criterion(normed_cls_hidden.unsqueeze(1), labels=labels.to(dtype=torch.float32))

            loss = self.config['lambda_map'] * masked_lm_loss + \
                   self.config['lambda_scl'] * similar_loss + \
                   self.config['lambda_rr'] * reconstruction_loss
            self.avg_similar_loss += self.config['lambda_scl'] * similar_loss.item()

        self.avg_mlm_loss += self.config['lambda_map'] * masked_lm_loss.item()
        self.avg_reconstruction_loss += self.config['lambda_rr'] * reconstruction_loss.item()
        self.avg_loss += loss.item()

        # Logging to TensorBoard by default
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'])
        self.pretrain_loss = self.avg_loss / self.current_step
        self.log("pretrain_loss", self.pretrain_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'])

        if batch_idx % self.config['report_frequency'] == 0 and batch_idx != 0 and ddp.global_rank == 0:
            mlm_loss = self.avg_mlm_loss / self.current_step
            similar_loss = self.avg_similar_loss / self.current_step
            rec_loss = self.avg_reconstruction_loss / self.current_step
            self.pretrain_loss = self.avg_loss / self.current_step
            self.avg_loss = self.avg_mlm_loss = self.avg_similar_loss = self.avg_reconstruction_loss = 0.
            self.current_step = 0
            self.log("pretrain_loss", self.pretrain_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'])
            print("PRETRAIN [Epoch {:2d}] [step {:3d}]".format(self.current_epoch, batch_idx),
                  "MAP loss: {:.4f}, SCL loss: {:.4f}, RR loss: {:.4f}, pretrain loss: {:.4f}"
                  .format(mlm_loss, similar_loss, rec_loss, self.pretrain_loss))
        if self.current_step % self.config['save_frequency'] == 0 and self.global_step != 0 and ddp.global_rank == 0:
            # self.avg_mlm_loss = 0.
            # self.avg_similar_loss = 0.
            # self.avg_reconstruction_loss = 0.
            # self.avg_loss = 0.
            # self.current_step = 0
            self.log("pretrain_loss", self.pretrain_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'])
            model_file = "epoch_{}_step_{}.pt".format(self.current_epoch, self.global_step)
            torch.save(self.model.state_dict(), os.path.join(self.model_path, model_file))
            decoder_file = "epoch_{}_step_{}_decoder.pt".format(self.current_epoch, self.global_step)
            torch.save(self.decoder.state_dict(), os.path.join(self.model_path, decoder_file))
            generator_file = "epoch_{}_step_{}_generator.pt".format(self.current_epoch, self.global_step)
            torch.save(self.generator.state_dict(), os.path.join(self.model_path, generator_file))
            print("Model saved: {}".format(model_file))
            return loss

        if (self.current_step % 250) == 0:
            self.log("pretrain_loss", self.pretrain_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'])
            model_file = "epoch_{}_step_{}.pt".format(self.current_epoch, self.global_step)
            torch.save(self.model.state_dict(), os.path.join(self.model_path, model_file))
            decoder_file = "epoch_{}_step_{}_decoder.pt".format(self.current_epoch, self.global_step)
            torch.save(self.decoder.state_dict(), os.path.join(self.model_path, decoder_file))
            generator_file = "epoch_{}_step_{}_generator.pt".format(self.current_epoch, self.global_step)
            torch.save(self.generator.state_dict(), os.path.join(self.model_path, generator_file))
            print("Model saved: {}".format(model_file))
            print(self.config['model'] + '_' + self.config['data_path'])
        return loss

    def configure_optimizers(self):
        optimizer_bert = build_optim_bert(self.config, self.model)
        optimizer_decoder = build_optim(self.config, self.decoder)
        optimizer_generator = build_optim(self.config, self.generator)

        # ______HuggingFace Accelerators________________________
        optimizer_bert.optimizer = HF_Accelerator.prepare_optimizer(optimizer_bert.optimizer)
        optimizer_decoder.optimizer = HF_Accelerator.prepare_optimizer(optimizer_decoder.optimizer)
        optimizer_generator.optimizer = HF_Accelerator.prepare_optimizer(optimizer_generator.optimizer)
        # ______HuggingFace Accelerators________________________

        self.optim = [optimizer_bert.optimizer, optimizer_decoder.optimizer, optimizer_generator.optimizer]
        return self.optim