import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from numpy.random import choice
import emoji
import re
import numpy as np
from itertools import cycle
import csv

DATA_SEED = 3
CONVERT_TO_BINARY = False
SILENT = False


#### define function to convert (multiclass)score to binary indicator
def to_emotion_ind(label):
    if label in ['NOT','not_sarcastic', 0]:
        return 0
    else:
        return 1

def to_emotion_sarcasm(ind):
    if ind == 1 :
        return "sarcasm"
    else:
        return "non_sarcasm"

def to_emotion_offensive(ind):
    if ind == 1 :
        return "offensive"
    else:
        return "non_offensive"

def to_emotion_hate(ind):
    if ind == 1 :
        return "hate"
    else:
        return "non_hate"


### define function to clean tweets (remove mentions RT emoji's, emoticons and url's

emoticon_string = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth      
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
    )"""

def give_emoji_free_text(text):
    return emoji.get_emoji_regexp().sub(r'', text)

def sanitize(string):
    """ Sanitize one string """

    # remove graphical emoji
    string = give_emoji_free_text(string)

    # remove textual emoji
    string = re.sub(emoticon_string,'',string)

    # remove user
    # assuming user has @ in front
    string = re.sub(r"""(?:@[\w_]+)""",'',string)

    #remove hashtag
    for punc in '#':
        string = string.replace(punc, '')

    # remove 't.co/' links
    string = re.sub(r'http//t.co\/[^\s]+', '', string, flags=re.MULTILINE)

    # remove RT
    string = re.sub(r'RT[\s]+','',string)

    # remove hyperlink
    string = re.sub(r'https?:\/\/S+','',string)


    return string



#### READING iSarcasm data
def load_isarcasm_data(seed):
    isarc_train = pd.read_csv('data/iSarcasm/isarc_train_text.csv')
    isarc_test = pd.read_csv('data/iSarcasm/isarc_test_text.csv')

    isarc_train['task'] = 'sarcasm'
    isarc_test['task'] = 'sarcasm'

    isarc_train['emotion_ind'] = isarc_train['sarcasm_label'].apply(to_emotion_ind)
    isarc_test['emotion_ind'] = isarc_test['sarcasm_label'].apply(to_emotion_ind)

    isarc_train['emotion'] = isarc_train['emotion_ind'].apply(to_emotion_sarcasm)
    isarc_test['emotion'] = isarc_test['emotion_ind'].apply(to_emotion_sarcasm)

    isarc_train, isarc_val = train_test_split(isarc_train, test_size=0.3, random_state=seed)
    return isarc_train, isarc_val, isarc_test

#### READING abusive language data (olid / offensive)
def load_olid_data(seed):
    olid_train = pd.read_csv('data/olid/olid-training-v1.0.tsv', sep='\t')
    olid_test_twt_a = pd.read_csv('data/olid/testsetlevela.tsv', sep='\t')
    #olid_test_twt_b = pd.read_csv('data/olid/testsetlevelb.tsv', sep='\t', index_col='id')
    #olid_test_twt_c = pd.read_csv('data/olid/testsetlevelc.tsv', sep='\t', index_col='id')
    olid_labels_a = pd.read_csv('data/olid/labels-levela.csv', header=None, names=['id', 'off_label'])
    #olid_labels_b = pd.read_csv('data/olid/labels-levelb.csv', header=None, names=['id', 'off_label'], index_col='id')
    #olid_labels_c = pd.read_csv('data/olid/labels-levelc.csv', header=None, names=['id', 'off_label'], index_col='id')

    #olid_train = olid_train[['tweet','subtask_a']]

    #merge testdata
    olid_test_a = pd.merge(olid_test_twt_a, olid_labels_a, on='id', how='inner')
    #olid_test_b = pd.merge(olid_test_twt_b, olid_labels_b, left_index=True, right_index=True)
    #olid_test_c = pd.merge(olid_test_twt_c, olid_labels_c, on='id', how='inner')

    olid_train['task'] = 'offensive'
    olid_test_a['task'] = 'offensive'

    olid_train['emotion_ind'] = olid_train['subtask_a'].apply(to_emotion_ind)
    olid_test_a['emotion_ind'] = olid_test_a['off_label'].apply(to_emotion_ind)

    olid_train['emotion'] = olid_train['emotion_ind'].apply(to_emotion_offensive)
    olid_test_a['emotion'] = olid_test_a['emotion_ind'].apply(to_emotion_offensive)

    olid_train, olid_val = train_test_split(olid_train, test_size=0.3, random_state=seed)
    return olid_train, olid_val, olid_test_a


#READING tweeteval hate data
def load_tweeteval_data(seed):
    hate_train_text = pd.read_csv('data/tweeteval_hate/train_text.txt', sep='\t', header=None, names=['tweet'], quoting=csv.QUOTE_NONE)
    hate_train_labels = pd.read_csv('data/tweeteval_hate/train_labels.txt', sep='\t', header=None, names=['emotion_ind'])
    hate_train = hate_train_text.join(hate_train_labels)

    hate_val_text = pd.read_csv('data/tweeteval_hate/val_text.txt', sep='\t', header=None, names=['tweet'], quoting=csv.QUOTE_NONE)
    hate_val_labels = pd.read_csv('data/tweeteval_hate/val_labels.txt', sep='\t', header=None, names=['emotion_ind'])
    hate_val = hate_val_text.join(hate_val_labels)

    hate_test_text = pd.read_csv('data/tweeteval_hate/test_text.txt', sep='\t', header=None, names=['tweet'], quoting=csv.QUOTE_NONE)
    hate_test_labels = pd.read_csv('data/tweeteval_hate/test_labels.txt', sep='\t', header=None, names=['emotion_ind'])
    hate_test = hate_test_text.join(hate_test_labels)

    hate_train['emotion'] = hate_train['emotion_ind'].apply(to_emotion_hate)
    hate_val['emotion'] = hate_val['emotion_ind'].apply(to_emotion_hate)
    hate_test['emotion'] = hate_test['emotion_ind'].apply(to_emotion_hate)


    hate_train['task'] = 'hate'
    hate_val['task'] = 'hate'
    hate_test['task'] = 'hate'

    return hate_train, hate_val, hate_test

def load_all_data(seed):
    # LOAD DATASETS
    isarc_train, isarc_val, isarc_test = load_isarcasm_data(seed)
    olid_train, olid_val, olid_test = load_olid_data(seed)
    # sem_emo_train, sem_emo_val, sem_emo_test = load_semeval_data(CONVERT_TO_BINARY)
    hate_train, hate_val, hate_test = load_tweeteval_data(seed)

    ## CONCATENATE ALL DATASETS and clean tweets (remove mentions, emoticons etc)
    train_all = pd.concat([
                              # sem_emo_train[['tweet', 'emotion', 'emotion_ind', 'task']],
                              olid_train[['tweet', 'emotion', 'emotion_ind', 'task']],
                              isarc_train[['tweet', 'emotion', 'emotion_ind', 'task']],
                              hate_train[['tweet', 'emotion', 'emotion_ind', 'task']]])
    train_all.tweet = train_all.tweet.apply(give_emoji_free_text).apply(sanitize)


    val_all = pd.concat([
                             # sem_emo_val[['tweet', 'emotion', 'emotion_ind', 'task']],
                             olid_val[['tweet', 'emotion', 'emotion_ind', 'task']],
                             isarc_val[['tweet', 'emotion', 'emotion_ind', 'task']],
                             hate_val[['tweet', 'emotion', 'emotion_ind', 'task']]])
    val_all.tweet = val_all.tweet.apply(give_emoji_free_text).apply(sanitize)

    test_all = pd.concat([
                             # sem_emo_test[['tweet', 'emotion', 'emotion_ind', 'task']],
                             olid_test[['tweet', 'emotion', 'emotion_ind', 'task']],
                             isarc_test[['tweet', 'emotion', 'emotion_ind', 'task']],
                             hate_test[['tweet', 'emotion', 'emotion_ind', 'task']]])
    test_all.tweet = test_all.tweet.apply(give_emoji_free_text).apply(sanitize)

    if not SILENT:
        ## PRINT COUNTS PER EMOTION
        print('*' * 20)
        print('train_all')
        print(train_all['emotion'].value_counts())

        print('*' * 20)
        print('val_all')
        print(val_all['emotion'].value_counts())

        print('*' * 20)
        print('test_all')
        print(test_all['emotion'].value_counts())

    # RETURN ALL DATA
    return train_all, val_all, test_all



def load_emotion_data(emotion, seed):
    # LOAD DATA
    train_all, val_all, test_all = load_all_data(seed)
    if emotion == "meta_all":
        return train_all, val_all, test_all

    ## SUBSET DATA ON SELECTED EMOTION (SEE parameters)
    train = train_all[train_all['task'] == emotion]
    val = val_all[val_all['task'] == emotion]
    test = test_all[test_all['task'] == emotion]


    if not SILENT:
        ## PRINT COUNTS OF THE CLASSES FOR SELECTED EMOTION
        print('*' * 20)
        print(f'{emotion} train')
        print(train['emotion_ind'].value_counts())

        print('*' * 20)
        print(f'{emotion} val')
        print(val['emotion_ind'].value_counts())

        print('*' * 20)
        print(f'{emotion} test')
        print(test['emotion_ind'].value_counts())

    return train, val, test





class MetaDataset(Dataset):

    def __init__(self, tasks_dataset, tasks_selected):
        self.tasks_selected = tasks_selected
        self.tasks_dataset = tasks_dataset
        self.tasks_dataset_iter={}
        for t_name, dataloader  in self.tasks_dataset.items():
            self.tasks_dataset_iter[t_name] = iter(dataloader)

    def __len__(self):
        return len(self.tasks_selected)

    def __getitem__(self, idx):
        task_name = self.tasks_selected[idx]
        try:
            sample_batch = next(self.tasks_dataset_iter[task_name])
        except StopIteration:
            self.tasks_dataset_iter[task_name] = iter(self.tasks_dataset[task_name])
            sample_batch = next(self.tasks_dataset_iter[task_name])

        return sample_batch


class EmotionDataset(Dataset):

    def __init__(self, dataset, tokenizer, max_len, opts):

        self.dataset = dataset
        self.targets = dataset.emotion_ind.to_numpy()
        self.tasks = dataset.task.to_numpy()
        self.tweets = dataset.tweet.to_numpy()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.opts = opts

        # split dataset according to the class
        self.dataset_classes={}
        self.class_name = self.dataset.emotion.unique()
        for c_n in self.class_name:
            self.dataset_classes[c_n] = self.dataset[self.dataset["emotion"]==c_n]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        tweet = str(self.tweets[item])
        target = self.targets[item]
        input_ids, attention_mask = self.tokenize(tweet)
        targets = torch.tensor(target, dtype=torch.long).to(self.opts.device)
        task = self.tasks[0]
        return  tweet, input_ids, attention_mask, targets, task


    def tokenize(self, tweet):
        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            #padding_to_max_length=True,
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].flatten().to(self.opts.device)
        attention_mask = encoding['attention_mask'].flatten().to(self.opts.device)

        return input_ids, attention_mask

def create_data_loader(df, tokenizer, max_len, batch_size, opts):
    ds = EmotionDataset(
        dataset=df,
        tokenizer=tokenizer,
        max_len=max_len,
        opts=opts
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collation_fn_emo
    )

def collation_fn_emo(inputs):
    (tweet, input_ids, attention_mask, targets, task) = zip(*inputs)

    batch = {
        'tweet_text': tweet,
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'targets': torch.stack(targets),
        'task': task
    }

    return batch

def create_metadataLoader(dataset, tokenizer, max_len, tasks_selected, num_task_eachtime, num_sample_pertask, opts):
    tasks_dataloader = {}
    for emo in tasks_selected:
        dataset_emo = dataset[dataset['task'] == emo]
        emo_dataset = EmotionDataset(dataset_emo, tokenizer, max_len, opts)
        emo_dataloader = DataLoader(emo_dataset, batch_size=num_sample_pertask*2, num_workers=0, shuffle=True, collate_fn=collation_fn_emo_meta)
        tasks_dataloader[emo] = emo_dataloader

    meatdata = MetaDataset(tasks_dataloader, tasks_selected)
    meatdata_loader = DataLoader(meatdata, batch_size=num_task_eachtime, num_workers=0, shuffle=True, collate_fn=collation_fn_meta)
    return meatdata_loader

# def collation_fn_emo_meta(inputs):
#     #split suport and query
#     (tweet, input_ids, attention_mask, targets, task) = zip(*inputs)

#     length = len(tweet)
#     support_length = int(length/2)

#     support = {
#                 'tweet_text': tweet[0:support_length],
#                 'input_ids': torch.stack(input_ids[0:support_length]),
#                 'attention_mask': torch.stack(attention_mask[0:support_length]),
#                 'targets': torch.stack(targets[0:support_length]),
#                 'task': task[0:support_length]
#               }

#     query = {
#                 'tweet_text': tweet[support_length:],
#                 'input_ids': torch.stack(input_ids[support_length:]),
#                 'attention_mask': torch.stack(attention_mask[support_length:]),
#                 'targets': torch.stack(targets[support_length:]),
#                 'task': task[support_length:]
#             }

#     return (support, query)
def collation_fn_emo_meta(inputs):
    (tweet, input_ids, attention_mask, targets, task) = zip(*inputs)

    return {
                'tweet_text': tweet,
                'input_ids': torch.stack(input_ids),
                'attention_mask': torch.stack(attention_mask),
                'targets': torch.stack(targets),
                'task': task
            }

# def collation_fn_meta(inputs):

#     return inputs
def collation_fn_meta(inputs):
    support = inputs[0]
    query = inputs[1]

    return [(support, query)]
