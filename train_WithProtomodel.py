import torch
import torch.nn as nn

import argparse

import os
import numpy as np
import itertools
from transformers import RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from data import load_emotion_data, create_data_loader, creat_metadataLoader

from models.EmoClassifier_multiTask import EmoClassifier_MulTask
from models.protomaml_multiTask import ProtoMAML_multiTask_framework
import logging

def main(opts):
    opts.output_dir = os.path.join(opts.output_dir, opts.emotion+"_proto")
    os.makedirs(os.path.join(opts.output_dir,"logs"), exist_ok=True)
    os.makedirs(os.path.join(opts.output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(opts.output_dir, "gradient_similarities"), exist_ok=True)
    logging.basicConfig(filename=os.path.join(opts.output_dir,"logs","output.log"),
                        level=logging.DEBUG,
                        format='%(asctime)s %(message)s')
    logging.info(f"lr_alpha:{opts.lr_alpha}")
    logging.info(f"lr_beta:{opts.lr_beta}")


    # Load metalearning data
    emotions_all = opts.emotion.split("&")
    tokenizer = RobertaTokenizer.from_pretrained(opts.bert_name)
    train_data, val_data, test_data = load_emotion_data("meta_all", opts.seed, opts.balance_datasets)
    n_class=[]
    for emo in emotions_all:
        c_num = train_data[train_data['task'] == emo]['emotion_ind'].value_counts()
        n_class.append(len(c_num))
    train_data_loader_meta = creat_metadataLoader(train_data, tokenizer, opts.max_len, emotions_all, opts.num_sample_perclass, opts)
    # val_data_loader_meta = creat_metadataLoader(val_data, tokenizer, opts.max_len, emotions_all, opts.num_sample_perclass, opts)


    # Load iter data
    # train_data_loader = {}
    # val_data_loader = {}
    # test_data_loader = {}
    # for emo in emotions_all:
    #     train_data_task = train_data[train_data['task'] == emo]
    #     val_data_task = val_data[val_data['task'] == emo]
    #     test_data_task = test_data[test_data['task'] == emo]
    #
    #     train_data_loader[emo] = create_data_loader(train_data_task, tokenizer, opts.max_len, opts.batch_size, opts)
    #     val_data_loader[emo] = create_data_loader(val_data_task, tokenizer, opts.max_len, opts.batch_size, opts)
    #     test_data_loader[emo]= create_data_loader(test_data_task, tokenizer, opts.max_len, opts.batch_size, opts)


    #build model
    classifier = EmoClassifier_MulTask(opts.bert_name, n_class, emotions_all)
    ProtoMAML_multiTask_Classifier_model = ProtoMAML_multiTask_framework(opts, classifier)
    ProtoMAML_multiTask_Classifier_model.to(opts.device)

    # optimizer
    criterion = nn.CrossEntropyLoss()

    ProtoMAML_multiTask_Classifier_model.similarity = {}
    gradient_combination = itertools.combinations(emotions_all, 2)
    for e1, e2 in gradient_combination:
        ProtoMAML_multiTask_Classifier_model.similarity[e1+"-"+e2] = {
          "all_model": [],
          "encoder":[],
          "fc_layer":[]
        }

    #train meta
    ProtoMAML_multiTask_Classifier_model.train()
    ProtoMAML_multiTask_Classifier_model.train_protomaml(train_data_loader_meta, criterion)

    # ProtoMAML_multiTask_Classifier_model.train_initmodel(train_data_loader, val_data_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--device', default = device, type=str,
                      help='the device name')

    parser.add_argument('--num_epoch_meta', default = 100, type=int,
                      help='Number of epoch')
    parser.add_argument('--num_class', default = 2, type=int,
                      help='Number of class, all task was set to same class')
    parser.add_argument('--lr_alpha', default = 0.001, type=float,
                      help='learning rate')
    parser.add_argument('--lr_beta', default = 0.0001, type=float,
                      help='learning rate')
    parser.add_argument('--train_step_per_episode', default = 1, type=int,
                      help='the dim of hidden layer in the encoder')
    parser.add_argument('--bert_name', default = 'roberta-base', type=str,
                      help='the bert name')
    parser.add_argument('--max_len', default = 180, type=int,
                      help='the bert name')
    parser.add_argument('--batch_size', default = 32, type=int,
                        help='batch size')
    parser.add_argument('--seed', type=int, default=3, help='Seed to use for pytorch and data splits.')

    parser.add_argument('--num_sample_perclass', type=int, default=10, help='')
    parser.add_argument('--emotion', default='sarcasm&fear&anger&joy&sadness&hate&offensive',
                    help="Emotion to be classified. it can be chosen from ['offensive', 'sarcasm', 'fear', 'anger', 'joy', 'sadness', 'hate'] or combination of them")


    parser.add_argument('--output_dir', default = './output', type=str,
                        help='output_dir')
    parser.add_argument('--balance_datasets', default = False, type=bool,
                        help='make datasets have same samplez')

    config = parser.parse_args()

    main(config)


