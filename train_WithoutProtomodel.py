#!/usr/bin/env python
# coding: utf-8

#import spacy
import sys
import os
import torch
from torch.utils.tensorboard import SummaryWriter
#import torchvision
import itertools
import json
from collections import defaultdict
from torch import nn, optim
import numpy as np
from tqdm import tqdm
import argparse
from transformers import RobertaTokenizer

from data import load_emotion_data, create_data_loader

from models.EmoClassifier_multiTask import EmoClassifier_MulTask

import logging

from scipy.spatial.distance import cosine

#### DEFINE TRAINING STEP
def train_epoch(
        model,
        data_loader,
        loss_fn,
        optimizer,
        args,
):
  data_loader_iter = {}
  losses_all = {}
  correct_predictions_all = {}
  n_examples_all = {}
  for emo, data_ler in data_loader.items():
    data_loader_iter[emo] = iter(data_ler)
    losses_all[emo] = []
    correct_predictions_all[emo] = 0
    n_examples_all[emo] = 0

  model.train()
  max_train_step_perEpoch =  max([len(loader) for loader in data_loader.values()])
  min_train_step_perEpoch =  min([len(loader) for loader in data_loader.values()])
  train_step_perEpoch = int((max_train_step_perEpoch + min_train_step_perEpoch) / 2.0)
  
  for train_step in range(train_step_perEpoch):
    print(f"train_step: {train_step}/{train_step_perEpoch}")
    # read one batch data from all tasks
    sample_batch_alltasks={}
    gradient_allTasks={}
    gradient_contact_allTasks={}

    for task_name, task_loader_iter in data_loader_iter.items():
      gradient_allTasks[task_name] = {}
      gradient_contact_allTasks[task_name] = {
        "all_model": torch.tensor([]).to("cpu"),
        "encoder":torch.tensor([]).to("cpu"),
        "fc_layer":torch.tensor([]).to("cpu")
      }
      try:
        sample_batch = next(task_loader_iter)
        sample_batch_alltasks[task_name]=sample_batch
      except StopIteration:
        data_loader_iter[task_name]= iter(data_loader[task_name])
        sample_batch = next(data_loader_iter[task_name])
        sample_batch_alltasks[task_name]=sample_batch

    # train tasks one batch per task
    for emo, batch_oneTaks in sample_batch_alltasks.items():

      for n, fc_layer in model.fc_layer_allTask.items():
        if n == emo:
          for p in fc_layer.parameters():
            p.requires_grad = True
        else:
          for p in fc_layer.parameters():
            p.requires_grad = False



      input_ids = batch_oneTaks["input_ids"]
      attention_mask = batch_oneTaks["attention_mask"]
      targets = batch_oneTaks["targets"]

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        emotion = emo
      )

      _, preds = torch.max(outputs, dim=1)
      loss = loss_fn(outputs, targets)
      optimizer.zero_grad()


      grads_oneTask = torch.autograd.grad(loss/2.0, filter(lambda p: p.requires_grad, model.parameters()))

      # record gradient
      for ind,(name, para) in enumerate([(n,p) for n, p in model.named_parameters() if p.requires_grad==True]):
        gradient_allTasks[emo][name] = grads_oneTask[ind]

      correct_predictions_all[emo] += torch.sum(preds == targets)
      n_examples_all[emo] += len(targets)
      losses_all[emo].append(loss.item())


    # update parameters
    for idx, (name_p, para) in enumerate(model.named_parameters()):
      grad_mean = 0.0
      if "fc_layer" not in name_p:
        for emo in sample_batch_alltasks.keys():
          grad_mean += gradient_allTasks[emo][name_p]
      else:
        for emo in sample_batch_alltasks.keys():
          try:
            grad_mean += gradient_allTasks[emo][name_p]
          except KeyError:
            continue
      grad_mean /= len(sample_batch_alltasks.keys())
      model.state_dict()[name_p] -= grad_mean*optimizer.defaults["lr"]


    if train_step %10 == 0 and len(data_loader_iter.keys())>=2:
      for t_n, gs in gradient_allTasks.items():
        for g_n, g in gs.items():
          if "fc_layer" in g_n:
            gradient_contact_allTasks[t_n]["fc_layer"] = torch.cat((gradient_contact_allTasks[t_n]["fc_layer"],gradient_allTasks[t_n][g_n].cpu().flatten()))
          else:
            gradient_contact_allTasks[t_n]["encoder"] = torch.cat((gradient_contact_allTasks[t_n]["encoder"],gradient_allTasks[t_n][g_n].cpu().flatten()))

          gradient_contact_allTasks[t_n]["all_model"] = torch.cat((gradient_contact_allTasks[t_n]["all_model"],gradient_allTasks[t_n][g_n].cpu().flatten()))

      #calculate cos similarity
      for m in ["all_model","encoder","fc_layer"]:
        for e1, e2 in itertools.combinations(gradient_contact_allTasks.keys(), 2):
          s =  1 - cosine(gradient_contact_allTasks[e1][m].numpy(), gradient_contact_allTasks[e2][m].numpy())
          model.similarity[e1+"-"+e2][m].append(s)


  for k,v in correct_predictions_all.items():
    correct_predictions_all[k] = (v.double() / n_examples_all[k]).item()
    losses_all[k] = np.mean(losses_all[k])

  return correct_predictions_all, losses_all



#### DEFINE EVALUATION STEP
def eval_model(model, data_loader, loss_fn):
  model.eval()

  acc_all={}
  loss_all={}
  for emo, data_ler in data_loader.items():
    losses = []
    correct_predictions = 0
    n_examples = 0

    for d in data_ler:
      input_ids = d["input_ids"]
      attention_mask = d["attention_mask"]
      targets = d["targets"]

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        emotion = d["task"][0]
      )
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      n_examples += len(targets)
      losses.append(loss.item())

    acc, loss = correct_predictions.double() / n_examples, np.mean(losses)
    acc_all[emo]=acc.item()
    loss_all[emo]=loss

  model.train()
  return acc_all, loss_all



#### TRAINING
def train(model, train_data_loader, val_data_loader, args, loss_fn, emotions_all):

  val_acc, val_loss = eval_model(
    model,
    val_data_loader,
    loss_fn
  )

  logging.info(f'Val loss {val_loss} accuracy {val_acc}')
  print(f'Val loss {val_loss} accuracy {val_acc}')

  # Define optimizer
  optimizer = optim.SGD(model.parameters(), lr=args.lr)

  # Logging
  writer = SummaryWriter(f'{args.output_dir}/logs/emotion = {args.emotion}, epochs = {args.epochs}, lr = {args.lr}, batch_size = {args.batch_size}')

  history = defaultdict(list)
  best_accuracy_averageAllTask = 0
  best_acc_epoch = 0

  model.similarity = {}
  gradient_combination = itertools.combinations(train_data_loader.keys(), 2)
  for e1, e2 in gradient_combination:
    model.similarity[e1+"-"+e2] = {
      "all_model": [],
      "encoder":[],
      "fc_layer":[]
    }

  # Start training
  for epoch in range(args.epochs):
    logging.info('-' * 20)
    logging.info(f'Epoch {epoch + 1}/{args.epochs}')

    print('-' * 20)
    print(f'Epoch {epoch + 1}/{args.epochs}')

    train_acc, train_loss,  = train_epoch(
      model,
      train_data_loader,
      loss_fn,
      optimizer,
      args,
    )

    logging.info(f'Train loss {train_loss} accuracy {train_acc}')
    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(
      model,
      val_data_loader,
      loss_fn
    )

    logging.info(f'Val loss {val_loss} accuracy {val_acc}')
    print(f'Val loss {val_loss} accuracy {val_acc}')

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

    for k in train_loss.keys():
      writer.add_scalar(f'Training loss_{k}', train_loss[k], global_step=epoch+1)
      writer.add_scalar(f'Training accuracy_{k}', train_acc[k], global_step=epoch+1)
      writer.add_scalar(f'Validation loss_{k}', val_loss[k], global_step=epoch+1)
      writer.add_scalar(f'Validation accuracy_{k}', val_acc[k], global_step=epoch+1)

    acc_averageAllTask = 0
    for k, acc in val_acc.items():
      acc_averageAllTask += acc
    acc_averageAllTask /= len(val_acc.items())

    if acc_averageAllTask > best_accuracy_averageAllTask:
      torch.save(model,
                 os.path.join(args.output_dir, "models", f'best_{args.emotion}_model.pkl'))
      best_accuracy_averageAllTask = acc_averageAllTask
      best_acc_epoch = epoch
      print(f"update best acc_averageAllTask_val: {best_accuracy_averageAllTask}, save best model")
      logging.info(f"update best acc_averageAllTask_val: {best_accuracy_averageAllTask}, save best model")

  print(f"best acc_averageAllTask_val: {best_accuracy_averageAllTask} at epoch: {best_acc_epoch}")
  logging.info(f"best acc_averageAllTask_val: {best_accuracy_averageAllTask} at epoch: {best_acc_epoch}")

  # save similarity
  s_path=os.path.join(args.output_dir, "gradient_similarities", "similarity.json")
  with open(s_path, 'w') as f:
    f.write(json.dumps(model.similarity))



#### DETERIMINE TEST ACCURACY (INFERENCE)
def test(test_data_loader, args, loss_fn):
  #load best model
  model_path = os.path.join(args.output_dir, "models", f'best_{args.emotion}_model.pkl')
  best_model=torch.load(model_path)

  for emo, data_loader in test_data_loader.items():

    test_acc, test_loss = eval_model(
      best_model,
      test_data_loader,
      loss_fn
    )

    logging.info(f'test loss {emo} {test_loss[emo]} accuracy {emo} {test_acc[emo]}')
    print(f'test loss {emo} {test_loss[emo]} accuracy {emo} {test_acc[emo]}')



def main(args):
  args.output_dir = os.path.join(args.output_dir, args.emotion)
  os.makedirs(os.path.join(args.output_dir,"logs"), exist_ok=True)
  os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)
  os.makedirs(os.path.join(args.output_dir, "gradient_similarities"), exist_ok=True)
  logging.basicConfig(filename=os.path.join(args.output_dir,"logs","output.log"),
                      level=logging.DEBUG,
                      format='%(asctime)s %(message)s')

  # Reproducibility
  torch.manual_seed(args.seed)
  torch.backends.cudnn.deterministic = True

  # Load metalearning data
  tokenizer = RobertaTokenizer.from_pretrained(args.bert_name)
  train_data, val_data, test_data = load_emotion_data("meta_all", args.seed)

  # Determine number of classes
  n_classes=[]
  emotions_all = args.emotion.split("&")
  train_data_loader = {}
  val_data_loader = {}
  test_data_loader = {}
  train_data_loader_list = []

  for emo in emotions_all:
    c_num = train_data[train_data['task'] == emo]['emotion_ind'].value_counts()
    n_classes.append(len(c_num))

    train_data_task = train_data[train_data['task'] == emo]
    val_data_task = val_data[val_data['task'] == emo]
    test_data_task = test_data[test_data['task'] == emo]

    train_data_loader[emo] = create_data_loader(train_data_task, tokenizer, args.max_len, args.batch_size, args)
    val_data_loader[emo] = create_data_loader(val_data_task, tokenizer, args.max_len, args.batch_size, args)
    test_data_loader[emo]= create_data_loader(test_data_task, tokenizer, args.max_len, args.batch_size, args)

    train_data_loader_list.append(train_data_loader[emo])

  print(f'n_classes = {n_classes}')


  # Instantiate BERT model
  model = EmoClassifier_MulTask(args.bert_name, n_classes, emotions_all)
  model.to(args.device)

  # Define loss function
  loss_fn = nn.CrossEntropyLoss()

  # Start training
  train(model, train_data_loader, val_data_loader, args, loss_fn, emotions_all)

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
  parser.add_argument('--emotion', default='offensive&sarcasm&fear&anger&joy&sadness&hate',
                      help="Emotion to be classified. it can be chosen from ['offensive', 'sarcasm', 'fear', 'anger', 'joy', 'sadness', 'hate'] or combination of them")


  parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train for.')
  parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate.')
  parser.add_argument('--seed', type=int, default=3, help='Seed to use for pytorch and data splits.')

  parser.add_argument('--bert_name', default = 'roberta-base', type=str,
                      help='the bert name')

  parser.add_argument('--output_dir', default = './output', type=str,
                      help='output path')

  args = parser.parse_args()

  main(args)
