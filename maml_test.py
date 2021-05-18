import torch
import torch.nn as nn

import argparse

import os
import numpy as np

from transformers import RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from data import load_emotion_data, create_data_loader, creat_metadataLoader

from models.maml import MAML_framework
from models.EmoClassifier import EmoClassifier
from models.protomaml import ProtoMAML_framework

def main(opts):

    # Load metalearning data
    # tokenizer = RobertaTokenizer.from_pretrained(opts.bert_name)
    # train_data, val_data, test_data = load_emotion_data("meta_all", opts.seed)
    # train_data_loader = creat_metadataLoader(train_data, tokenizer, opts.max_len, opts.tasks_selected, opts.num_task_eachtime, opts.num_sample_pertask, opts)
    # Load metalearning data
    tokenizer = RobertaTokenizer.from_pretrained(opts.bert_name)
    train_data, val_data, test_data = load_emotion_data("meta_all", opts.seed)

    # Determine number of classes
    n_classes=[]
    emotions_all = opts.emotion.split("&")
    train_data_loader = {}
    val_data_loader = {}
    test_data_loader = {}
    for emo in emotions_all:
        c_num = train_data[train_data['task'] == emo]['emotion_ind'].value_counts()
        n_classes.append(len(c_num))

        train_data_task = train_data[train_data['task'] == emo]
        val_data_task = val_data[val_data['task'] == emo]
        test_data_task = test_data[test_data['task'] == emo]

        train_data_loader[emo] = create_data_loader(train_data_task, tokenizer, opts.max_len, opts.num_sample_pertask, opts)
        val_data_loader[emo] = create_data_loader(val_data_task, tokenizer, opts.max_len, opts.num_sample_pertask, opts)
        test_data_loader[emo]= create_data_loader(test_data_task, tokenizer, opts.max_len, opts.num_sample_pertask, opts)

    print(f'n_classes = {n_classes}')


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

    parser.add_argument('--num_epoch', default = 1000, type=int,
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

    parser.add_argument('--num_sample_pertask', type=int, default=40, help='')
    parser.add_argument('--emotion', default="hate&offensive",
                        choices=['offensive', 'sarcasm', 'fear', 'anger', 'joy', 'sadness', 'hate'],
                        help='Emotion to be classified.')

    config = parser.parse_args()

    main(config)

