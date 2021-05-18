import torch
import torch.nn as nn

import argparse

import os
import numpy as np

from transformers import RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from data import load_emotion_data, create_data_loader, creat_metadataLoader

from models.maml_multiTask import MAML_multiTask_framework
from models.EmoClassifier_multiTask import EmoClassifier_MulTask
from models.protomaml import ProtoMAML_framework
import logging

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

from models.EmoClassifier_multiTask import EmoClassifier_MulTask



def train(model, train_data_loader, val_data_loader, args, loss_fn, emotions_all):

    # Logging
    model.best_accuracy = {}
    for emo in emotions_all:
        model.best_accuracy[emo] = 0

    # Start training
    for epoch in range(args.epochs):
        logging.info(f'**********************MAML Epoch {epoch + 1}/{args.epochs}**********************')
        print(f'**********************MAML Epoch {epoch + 1}/{args.epochs}**********************')

        train_acc, train_loss = model.train_maml_epoch(
            train_data_loader,
            loss_fn
        )

        logging.info(f'Train maml query loss {train_loss} accuracy {train_acc}')
        print(f'Train maml query loss {train_loss} accuracy {train_acc}')

        model.eval_maml_epoch(
            train_data_loader,
            val_data_loader,
            loss_fn,
        )




def main(args):
    args.output_dir = os.path.join(args.output_dir,args.emotion+"_MAML")
    os.makedirs(os.path.join(args.output_dir,"logs"),exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "models"),exist_ok=True)
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
    for emo in emotions_all:
        c_num = train_data[train_data['task'] == emo]['emotion_ind'].value_counts()
        n_classes.append(len(c_num))

        train_data_task = train_data[train_data['task'] == emo]
        val_data_task = val_data[val_data['task'] == emo]
        test_data_task = test_data[test_data['task'] == emo]

        train_data_loader[emo] = create_data_loader(train_data_task, tokenizer, args.max_len, args.batch_size, args)
        val_data_loader[emo] = create_data_loader(val_data_task, tokenizer, args.max_len, args.batch_size, args)
        test_data_loader[emo]= create_data_loader(test_data_task, tokenizer, args.max_len, args.batch_size, args)



    print(f'n_classes = {n_classes}')

    # Load metalearning data
    # tokenizer = RobertaTokenizer.from_pretrained(args.bert_name)
    # train_data, val_data, test_data = load_emotion_data("meta_all", args.seed)
    # train_data_loader = creat_metadataLoader(train_data, tokenizer, args.max_len, args.tasks_selected, args.num_task_eachtime, args.num_sample_pertask, args)

    #build model
    classifier = EmoClassifier_MulTask(args.bert_name, n_classes, emotions_all)
    MAML_Classifier_model = MAML_multiTask_framework(args, classifier)
    MAML_Classifier_model.to(args.device)

    # optimizer
    criterion = nn.CrossEntropyLoss()


    #train
    train(MAML_Classifier_model, train_data_loader, val_data_loader, args, criterion, emotions_all)

    #test
    MAML_Classifier_model.test_classifier(test_data_loader, args, criterion)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--device', default = device, type=str,
                        help='the device name')
    # parser.add_argument('--num_episodes', default = 10, type=int,
    #                     help='Number of episodes per epoch')
    parser.add_argument('--epochs', default = 1, type=int,
                        help='Number of epoch')
    parser.add_argument('--classifier_epochs', default = 1, type=int,
                        help='Number of epoch')
    parser.add_argument('--num_class', default = 2, type=int,
                        help='Number of class')
    parser.add_argument('--lr_alpha', default = 0.00001, type=float,
                        help='learning rate')
    parser.add_argument('--lr_beta', default = 0.00001, type=float,
                        help='learning rate')
    parser.add_argument('--train_step_per_episode', default = 1, type=int,
                        help='the dim of hidden layer in the encoder')
    parser.add_argument('--bert_name', default = 'roberta-base', type=str,
                        help='the bert name')
    parser.add_argument('--max_len', default = 180, type=int,
                        help='the bert name')

    parser.add_argument('--seed', type=int, default=3, help='Seed to use for pytorch and data splits.')

    # parser.add_argument('--tasks_selected', type=str, default=["offensive", "sarcasm"],
    #                     help='the task names which are selected')

    # parser.add_argument('--num_task_eachtime', type=int, default=2,
    #                     help='the number of task which is select')
    # parser.add_argument('--num_sample_pertask', type=int, default=50, help='Number of epochs to train for.')
    parser.add_argument('--emotion', default="hate&offensive",
                        choices=['offensive', 'sarcasm', 'fear', 'anger', 'joy', 'sadness', 'hate'],
                        help='Emotion to be classified.')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')

    parser.add_argument('--output_dir', default = './output', type=str,
                        help='output path')




    config = parser.parse_args()

    main(config)

