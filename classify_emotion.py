#!/usr/bin/env python
# coding: utf-8

#import spacy
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



#### DEFINE TRAINING STEP
def train_epoch(
        model,
        data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler
):
  model = model.train()

  losses = []
  correct_predictions = 0
  n_examples = 0

  progress = tqdm(data_loader)
  for d in progress:
    optimizer.zero_grad()

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

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

  return correct_predictions.double() / n_examples, np.mean(losses)



#### DEFINE EVALUATION STEP
def eval_model(model, data_loader, loss_fn, device):
  model = model.eval()

  losses = []
  correct_predictions = 0
  n_examples = 0

  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

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
def train(model, train_data_loader, val_data_loader, args):
  # Define optimizer
  optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=False)
  total_steps = len(train_data_loader) * args.epochs

  scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
  )

  # Define loss function
  loss_fn = nn.CrossEntropyLoss().to(args.device)

  # Logging
  writer = SummaryWriter(f'logs/BB {args.emotion}, epochs = {args.emotion}, lr = {args.lr}, batch_size = {args.batch_size}')

  history = defaultdict(list)
  best_accuracy = 0

  # Start training
  for epoch in range(args.epochs):

    print(f'Epoch {epoch + 1}/{args.epochs}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
      model,
      train_data_loader,
      loss_fn,
      optimizer,
      args.device,
      scheduler
    )

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(
      model,
      val_data_loader,
      loss_fn,
      args.device
    )

    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

    writer.add_scalar('Training loss', train_loss, global_step=epoch+1)
    writer.add_scalar('Training accuracy', train_acc, global_step=epoch+1)
    writer.add_scalar('Validation loss', val_loss, global_step=epoch+1)
    writer.add_scalar('Validation accuracy', val_acc, global_step=epoch+1)

    if val_acc > best_accuracy:
      torch.save(model.state_dict(), f'best_{args.emotion}_model.bin')
      best_accuracy = val_acc



#### DETERIMINE TEST ACCURACY (INFERENCE)
def eval():
  ## TO BE ADDED
  pass




def main(args):

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
  model = EmoClassifier(args.bert_name, n_classes).to(args.device)

  # Start training
  train(model, train_data_loader, val_data_loader, args)

if __name__ == '__main__':
  # Read command line arguments
  parser = argparse.ArgumentParser(description='Train emotion classification model.')

  # device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  parser.add_argument('--device', default = device, type=str,
                      help='the device name')
  parser.add_argument('--max_len', type=int, default=32, help='Maximum input length.')
  parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
  parser.add_argument('--emotion', choices=['offensive', 'sarcasm', 'fear', 'anger', 'joy', 'sadness', 'hate'], default='hate', help='Emotion to be classified.')

  parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train for.')
  parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate.')
  parser.add_argument('--seed', type=int, default=3, help='Seed to use for pytorch and data splits.')

  parser.add_argument('--tasks_selected', type=str, default=["offensive", "sarcasm", "hate"],
                      help='the task names which are selected')

  parser.add_argument('--num_task_eachtime', type=int, default=1,
                      help='the number of task which is select')
  parser.add_argument('--num_sample_pertask', type=int, default=20, help='Number of epochs to train for.')

  parser.add_argument('--bert_name', default = 'roberta-base', type=str,
                      help='the bert name')


  args = parser.parse_args()

  main(args)
