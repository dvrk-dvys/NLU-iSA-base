



#
# class inference():
#     def __init__(self):
#         self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
#         # self.LABEL = data.LabelField()
#         self.itos = vocab.Vocab
#
#
#     def placeholder_model(self, config):
#
#         model = pretrain_Lite(model_name_or_path=config['model_name_or_path'],
#                               num_labels=3,
#                               task_name=config['mode'],
#                               learning_rate=config['learning_rate'],
#                               adam_epsilon=1e-8,
#                               warmup_steps=0,
#                               weight_decay=config['weight_decay'],
#                               train_batch_size=32,
#                               eval_batch_size=32,
#                               config=config)
#         yelp_bert_model = model.load_state_dict(torch.load('/Users/jordanharris/SCAPT-ABSA/models_weights/amzn_bert_torchmodel_weights.pth'))
#
#         return yelp_bert_model
#
#
#     def predict(self, model, sentence, device):
#         tokenized = [tok.text for tok in self.tokenizer.tokenize(sentence)]  # tokenize the sentence
#         indexed = [vocab.stoi[t] for t in tokenized]  # convert to integer sequence
#         length = [len(indexed)]  # compute no. of words
#         tensor = torch.LongTensor(indexed).to(device)  # convert to tensor
#         tensor = tensor.unsqueeze(1).T  # reshape in form of batch,no. of words
#         length_tensor = torch.LongTensor(length)  # convert to tensor
#         prediction = model(tensor, length_tensor)  # prediction
#         return prediction.item()
#
#
#     #!!!!!!!!!!!!!!!!!!! best   https://www.analyticsvidhya.com/blog/2021/05/bert-for-natural-language-inference-simplified-in-pytorch/
#     def tokenize_bert(self, sentence):
#         tokens = self.tokenizer.tokenize(sentence)
#         return tokens
#     def split_and_cut(self, sentence):
#         tokens = sentence.strip().split(" ")
#         tokens = tokens
#         return tokens
#
#     def trim_sentence(self, sent):
#         try:
#             sent = sent.split()
#             sent = sent[:128]
#             return " ".join(sent)
#         except:
#             return sent
#
#     def get_sent1_token_type(self, sent):
#         try:
#             return [0] * len(sent)
#         except:
#             return []
#
#     # Get list of 1s
#     def get_sent2_token_type(self, sent):
#         try:
#             return [1] * len(sent)
#         except:
#             return []
#
#     # combine from lists
#     def combine_seq(self, seq):
#         return " ".join(seq)
#
#     # combines from lists of int
#     def combine_mask(self, mask):
#         mask = [str(m) for m in mask]
#         return " ".join(mask)
#
#     def convert_to_int(self, tok_ids):
#         tok_ids = [int(x) for x in tok_ids]
#         return tok_ids
#
#     def predict_inference(self, premise, hypothesis, model, device):
#         model.eval()
#         premise = '[CLS] ' + premise + ' [SEP]'
#         hypothesis = hypothesis + ' [SEP]'
#         prem_t = self.tokenize_bert(premise)
#         hypo_t = self.tokenize_bert(hypothesis)
#         prem_type = self.get_sent1_token_type(prem_t)
#         hypo_type = self.get_sent2_token_type(hypo_t)
#         indexes = prem_t + hypo_t
#         indexes = self.tokenizer.convert_tokens_to_ids(indexes)
#         indexes_type = prem_type + hypo_type
#         attn_mask = self.get_sent2_token_type(indexes)
#         indexes = torch.LongTensor(indexes).unsqueeze(0).to(device)
#         indexes_type = torch.LongTensor(indexes_type).unsqueeze(0).to(device)
#         attn_mask = torch.LongTensor(attn_mask).unsqueeze(0).to(device)
#         prediction = model(indexes, attn_mask, indexes_type)
#         prediction = prediction.argmax(dim=-1).item()
#         return self.get_itos[prediction]
#         # return self.LABEL.vocab.itos[prediction]
#
#     def pad_indx(self, tokens):
#         indexes = DistilBertTokenizerFast.convert_tokens_to_ids(tokens)
#         print(indexes)
#         cls_token = DistilBertTokenizerFast.cls_token
#         sep_token = DistilBertTokenizerFast.sep_token
#         pad_token = DistilBertTokenizerFast.pad_token
#         unk_token = DistilBertTokenizerFast.unk_token
#         print(cls_token, sep_token, pad_token, unk_token)
#         cls_token_idx = DistilBertTokenizerFast.cls_token_id
#         sep_token_idx = DistilBertTokenizerFast.sep_token_id
#         pad_token_idx = DistilBertTokenizerFast.pad_token_id
#         unk_token_idx = DistilBertTokenizerFast.unk_token_id
#         print(cls_token_idx, sep_token_idx, pad_token_idx, unk_token_idx)
#
#         return cls_token, sep_token, pad_token, unk_token, cls_token_idx, sep_token_idx, pad_token_idx, unk_token_idx
#
#     # trainer = pl.Trainer(accelerator="cpu",
#     #                      devices=1,
#     #                      max_epochs=8,
    #                      callbacks=[RichProgressBar(refresh_rate=1)],
    #                      overfit_batches=0.010,
    #                      # overfit_batches=14,
    #                      auto_scale_batch_size="binsearch",
    #                      strategy='dp',
    #                      precision=32)
    # predictions = trainer.predict(yelp_bert_model, dataloaders=dl)
    # inference
