import os, pdb, sys
import numpy as np
import re

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from transformers import BertModel, BertConfig
from transformers import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
import transformers

class IntentModel(nn.Module):
  def __init__(self, args, tokenizer, target_size):
    super().__init__()
    self.tokenizer = tokenizer
    self.model_setup(args)
    self.target_size = target_size
    self.dropout = nn.Dropout(args.drop_rate) # task2: initilize the dropout and classify layers
    self.classify = Classifier(args,target_size)
    
  def model_setup(self, args):
    print(f"Setting up {args.model} model")

    # task1: get a pretrained model of 'bert-base-uncased'
    self.encoder = BertModel.from_pretrained('bert-base-uncased')
    
    self.encoder.resize_token_embeddings(len(self.tokenizer))  # transformer_check

    # for layer in self.encoder.parameters():
    #     layer.requires_grad = False
  def forward(self, inputs, targets):
    """
    task1: 
        feeding the input to the encoder, 
    task2: 
        take the last_hidden_state's <CLS> token as output of the
        encoder, feed it to a drop_out layer with the preset dropout rate in the argparse argument, 
    task3:
        feed the output of the dropout layer to the Classifier which is provided for you.
    """
    outputs = self.encoder(**inputs)

    # taking the the output of [CLS] token...

    pooled_output = outputs[1] #pooler output
    # pooled_output = outputs[0][:,0,:] #[CLS] output

    pooled_output = self.dropout(pooled_output)
    logits = self.classify(pooled_output)

    return logits
    
  
class Classifier(nn.Module):
  def __init__(self, args, target_size):
    super().__init__()
    input_dim = args.embed_dim
    self.top = nn.Linear(input_dim, args.hidden_dim)
    self.relu = nn.ReLU()
    self.bottom = nn.Linear(args.hidden_dim, target_size)

  def forward(self, hidden):
    middle = self.relu(self.top(hidden))
    logit = self.bottom(middle)
    return logit


class CustomModel(IntentModel):
  def __init__(self, args, tokenizer, target_size):
    super().__init__()
    self.tokenizer = tokenizer
    self.model_setup(args)
    self.target_size = target_size
    self.dropout = nn.Dropout(args.drop_rate)
    self.classify = Classifier(args,target_size)
    
  def model_setup(self, args):
    print(f"Setting up {args.model} model")

    self.encoder = BertModel.from_pretrained('bert-base-uncased')
    
    self.encoder.resize_token_embeddings(len(self.tokenizer))  # transformer_check

  def forward(self, inputs, targets):
    outputs = self.encoder(**inputs) # taking the the output of [CLS] token
    pooled_output = outputs[1]
    pooled_output = self.dropout(pooled_output)
    logits = self.classify(pooled_output)
    return logits
    
    # task1: use initialization for setting different strategies/techniques to better fine-tune the BERT model

class SupConModel(IntentModel):
  def __init__(self, args, tokenizer, target_size, feat_dim=768):
    super().__init__(args, tokenizer, target_size)
    self.head = nn.Linear(feat_dim, args.hidden_dim)
    
  def model_setup(self, args):
    print(f"Setting up {args.model} model")

    # task1: get a pretrained model of 'bert-base-uncased'
    self.encoder = BertModel.from_pretrained('bert-base-uncased')
    
    self.encoder.resize_token_embeddings(len(self.tokenizer))  # transformer_check

    # task1: initialize a linear head layer
 
    def forward(self, inputs, targets):
        outputs = self.encoder(**inputs)

        # taking the the output of [CLS] token...

        pooled_output = outputs[1] #pooler output
        # pooled_output = outputs[0][:,0,:] #[CLS] output

        pooled_output = self.dropout(pooled_output)
        feat = F.normalize(self.head(pooled_output), dim=1)
        return feat