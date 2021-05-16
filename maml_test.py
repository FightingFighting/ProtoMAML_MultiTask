import torch
import torch.nn as nn

import argparse

import os
import numpy as np

from transformers import RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from data import load_emotion_data, create_data_loader, create_metadataLoader

from models.maml import MAML_framework
from models.EmoClassifier import EmoClassifier
from models.protomaml import ProtoMAML_framework

def main(opts):

    # Load metalearning data
    tokenizer = RobertaTokenizer.from_pretrained(opts.bert_name)
    train_data, val_data, test_data = load_emotion_data("meta_all", opts.seed)
    train_data_loader = create_metadataLoader(train_data, tokenizer, opts.max_len, opts.tasks_selected, opts.num_task_eachtime, opts.num_sample_pertask, opts)

    #build model
    classifier = EmoClassifier(opts.bert_name, opts.num_class)
    MAML_Classifier_model = MAML_framework(opts, classifier)
    MAML_Classifier_model.to(opts.device)

    # optimizer
    criterion = nn.CrossEntropyLoss()

    #train
    MAML_Classifier_model.train()
    MAML_Classifier_model.train_maml(train_data_loader, criterion)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--device', default = device, type=str,
                        help='the device name')
    parser.add_argument('--num_episodes', default = 10, type=int,
                        help='Number of episodes per epoch')
    parser.add_argument('--num_epoch', default = 1000, type=int,
                        help='Number of epoch')
    parser.add_argument('--num_class', default = 2, type=int,
                        help='Number of class')
    parser.add_argument('--lr_alpha', default = 0.0001, type=float,
                        help='learning rate')
    parser.add_argument('--lr_beta', default = 0.0001, type=float,
                        help='learning rate')
    parser.add_argument('--train_step_per_episode', default = 1, type=int,
                        help='the dim of hidden layer in the encoder')
    parser.add_argument('--bert_name', default = 'roberta-base', type=str,
                        help='the bert name')
    parser.add_argument('--max_len', default = 180, type=int,
                        help='the bert name')
    parser.add_argument('--seed', type=int, default=3, help='Seed to use for pytorch and data splits.')

    parser.add_argument('--tasks_selected', type=str, default=["offensive", "sarcasm"],
                        help='the task names which are selected')

    parser.add_argument('--num_task_eachtime', type=int, default=3,
                        help='the number of task which is select')
    parser.add_argument('--num_sample_pertask', type=int, default=20, help='Number of epochs to train for.')


    config = parser.parse_args()

    main(config)

