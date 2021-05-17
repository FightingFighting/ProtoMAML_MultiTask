#!/usr/bin/env python
# coding: utf-8

#import spacy
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
#import torchvision

import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
# from textwrap import wrap
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse

import transformers
from transformers import RobertaTokenizer, AdamW, get_linear_schedule_with_warmup

from data import load_emotion_data, create_data_loader

from models.EmoClassifier import EmoClassifier

import logging


#### DEFINE TRAINING STEP
def train_epoch(
        model,
        data_loader,
        loss_fn,
        optimizer,
        scheduler
):
  model = model.train()

  losses = []
  correct_predictions = 0
  n_examples = 0

  progress = tqdm(data_loader)
  for d in progress:
    input_ids = d["input_ids"]
    attention_mask = d["attention_mask"]
    targets = d["targets"]

    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)

    correct_predictions += torch.sum(preds == targets)
    n_examples += len(targets)
    losses.append(loss.item())
    progress.set_postfix({'loss': loss.item()})

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

  return correct_predictions.double() / n_examples, np.mean(losses)



#### DEFINE EVALUATION STEP
def eval_model(model, data_loader, loss_fn):
  model = model.eval()
  losses = []
  correct_predictions = 0
  n_examples = 0

  for d in data_loader:
    input_ids = d["input_ids"]
    attention_mask = d["attention_mask"]
    targets = d["targets"]

    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    _, preds = torch.max(outputs, dim=1)

    loss = loss_fn(outputs, targets)

    correct_predictions += torch.sum(preds == targets)
    n_examples += len(targets)
    losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)



#### TRAINING
def train(model, train_data_loader, val_data_loader, args, loss_fn):


  val_acc, val_loss = eval_model(
    model,
    val_data_loader,
    loss_fn,
  )
  logging.info(f'Val loss {val_loss} accuracy {val_acc}')
  print(f'Val loss {val_loss} accuracy {val_acc}')

  # Define optimizer
  optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=False)
  total_steps = len(train_data_loader) * args.epochs

  scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
  )

  # Logging
  writer = SummaryWriter(f'{args.output_dir}/logs/emotion = {args.emotion}, epochs = {args.epochs}, lr = {args.lr}, batch_size = {args.batch_size}')

  history = defaultdict(list)
  best_accuracy = 0

  # Start training
  for epoch in range(args.epochs):
    logging.info('-' * 20)
    logging.info(f'Epoch {epoch + 1}/{args.epochs}')

    print('-' * 20)
    print(f'Epoch {epoch + 1}/{args.epochs}')

    train_acc, train_loss = train_epoch(
      model,
      train_data_loader,
      loss_fn,
      optimizer,
      scheduler
    )

    logging.info(f'Train loss {train_loss} accuracy {train_acc}')
    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(
      model,
      val_data_loader,
      loss_fn,
    )

    logging.info(f'Val loss {val_loss} accuracy {val_acc}')
    print(f'Val loss {val_loss} accuracy {val_acc}')

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

    writer.add_scalar('Training loss', train_loss, global_step=epoch+1)
    writer.add_scalar('Training accuracy', train_acc, global_step=epoch+1)
    writer.add_scalar('Validation loss', val_loss, global_step=epoch+1)
    writer.add_scalar('Validation accuracy', val_acc, global_step=epoch+1)

    if val_acc > best_accuracy:
      torch.save(model,
                 os.path.join(args.output_dir, "models",f'best_{args.emotion}_model.pkl'))
      best_accuracy = val_acc



#### DETERIMINE TEST ACCURACY (INFERENCE)
def test(val_data_loader, args, loss_fn):
  #load best model
  model_path = os.path.join(args.output_dir, "models",f'best_{args.emotion}_model.pkl')
  best_model=torch.load(model_path)

  test_acc, test_loss = eval_model(
    best_model,
    val_data_loader,
    loss_fn
  )

  logging.info(f'test loss {test_loss} accuracy {test_acc}')
  print(f'test loss {test_loss} accuracy {test_acc}')



def main(args):
  args.output_dir = os.path.join(args.output_dir,args.emotion)
  os.makedirs(os.path.join(args.output_dir,"logs"))
  os.makedirs(os.path.join(args.output_dir, "models"))
  logging.basicConfig(filename=os.path.join(args.output_dir,"logs","output.log"),
                      level=logging.DEBUG,
                      format='%(asctime)s %(message)s')

  # Reproducibility
  torch.manual_seed(args.seed)
  torch.backends.cudnn.deterministic = True

  # Load metalearning data
  tokenizer = RobertaTokenizer.from_pretrained(args.bert_name)
  train_data, val_data, test_data = load_emotion_data(args.emotion, args.seed)

  train_data_loader = create_data_loader(train_data, tokenizer, args.max_len, args.batch_size, args)
  val_data_loader = create_data_loader(val_data, tokenizer, args.max_len, args.batch_size, args)
  test_data_loader = create_data_loader(test_data, tokenizer, args.max_len, args.batch_size, args)

  # Determine number of classes
  n_classes = len(train_data['emotion_ind'].value_counts())
  print(f'n_classes = {n_classes}')

  # Instantiate BERT model
  model = EmoClassifier(args.bert_name, n_classes)
  model.to(args.device)

  # Define loss function
  loss_fn = nn.CrossEntropyLoss()

  # Start training
  train(model, train_data_loader, val_data_loader, args, loss_fn)

  # test model
  test(test_data_loader, args, loss_fn)

if __name__ == '__main__':
  # Read command line arguments
  parser = argparse.ArgumentParser(description='Train emotion classification model.')

  # device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  parser.add_argument('--device', default = device, type=str,
                      help='the device name')
  parser.add_argument('--max_len', type=int, default=180, help='Maximum input length.')
  parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
  parser.add_argument('--emotion', default='offensive',
                      choices=['offensive', 'sarcasm', 'fear', 'anger', 'joy', 'sadness', 'hate'],
                      help='Emotion to be classified.')

  parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train for.')
  parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate.')
  parser.add_argument('--seed', type=int, default=3, help='Seed to use for pytorch and data splits.')

  parser.add_argument('--bert_name', default = 'roberta-base', type=str,
                      help='the bert name')

  parser.add_argument('--output_dir', default = './output', type=str,
                      help='output path')

  args = parser.parse_args()

  main(args)
