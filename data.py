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

def to_emotion(inten):
    if "0" in inten:
        if "sadness" in inten:
            return "non_sadness"
        elif "joy" in inten:
            return "non_joy"
        elif "anger" in inten:
            return "non_anger"
        elif "fear" in inten:
            return "non_fear"
    elif "1" in inten or "2" in inten or "3" in inten:
        if "sadness" in inten:
            return "sadness"
        elif "joy" in inten:
            return "joy"
        elif "anger" in inten:
            return "anger"
        elif "fear" in inten:
            return "fear"



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



#READING SemEval data
def load_semeval_data(convert_to_binary):
    sem_anger_train = pd.read_csv('data/SemEval/EI-oc-En-anger-train.txt', sep='\t')
    sem_anger_dev = pd.read_csv('data/SemEval/2018-EI-oc-En-anger-dev.txt', sep='\t')
    sem_anger_test = pd.read_csv('data/SemEval/2018-EI-oc-En-anger-test-gold.txt', sep='\t')

    sem_fear_train = pd.read_csv('data/SemEval/EI-oc-En-fear-train.txt', sep='\t')
    sem_fear_dev = pd.read_csv('data/SemEval/2018-EI-oc-En-fear-dev.txt', sep='\t')
    sem_fear_test = pd.read_csv('data/SemEval/2018-EI-oc-En-fear-test-gold.txt', sep='\t')

    sem_joy_train = pd.read_csv('data/SemEval/EI-oc-En-joy-train.txt', sep='\t')
    sem_joy_dev = pd.read_csv('data/SemEval/2018-EI-oc-En-joy-dev.txt', sep='\t')
    sem_joy_test = pd.read_csv('data/SemEval/2018-EI-oc-En-joy-test-gold.txt', sep='\t')

    sem_sad_train = pd.read_csv('data/SemEval/EI-oc-En-sadness-train.txt', sep='\t')
    sem_sad_dev = pd.read_csv('data/SemEval/2018-EI-oc-En-sadness-dev.txt', sep='\t')
    sem_sad_test = pd.read_csv('data/SemEval/2018-EI-oc-En-sadness-test-gold.txt', sep='\t')

    sem_emo_train = pd.concat([sem_anger_train.rename(columns={"ID": "id", "Tweet": "tweet", "Affect Dimension": "emotion"})
                                  ,sem_fear_train.rename(columns={"ID": "id", "Tweet": "tweet", "Affect Dimension": "emotion"})
                                  ,sem_joy_train.rename(columns={"ID": "id", "Tweet": "tweet", "Affect Dimension": "emotion"})
                                  ,sem_sad_train.rename(columns={"ID": "id", "Tweet": "tweet", "Affect Dimension": "emotion"})])

    sem_emo_dev = pd.concat([sem_anger_dev.rename(columns={"ID": "id", "Tweet": "tweet", "Affect Dimension": "emotion"})
                                ,sem_fear_dev.rename(columns={"ID": "id", "Tweet": "tweet", "Affect Dimension": "emotion"})
                                ,sem_joy_dev.rename(columns={"ID": "id", "Tweet": "tweet", "Affect Dimension": "emotion"})
                                ,sem_sad_dev.rename(columns={"ID": "id", "Tweet": "tweet", "Affect Dimension": "emotion"})])

    sem_emo_test = pd.concat([sem_anger_test.rename(columns={"ID": "id", "Tweet": "tweet", "Affect Dimension": "emotion"})
                                 ,sem_fear_test.rename(columns={"ID": "id", "Tweet": "tweet", "Affect Dimension": "emotion"})
                                 ,sem_joy_test.rename(columns={"ID": "id", "Tweet": "tweet", "Affect Dimension": "emotion"})
                                 ,sem_sad_test.rename(columns={"ID": "id", "Tweet": "tweet", "Affect Dimension": "emotion"})])

    sem_emo_train[['emotion_ind', 'emo_desc']]=sem_emo_train['Intensity Class'].str.split(':', expand=True).apply(pd.to_numeric, errors='ignore')

    sem_emo_dev[['emotion_ind', 'emo_desc']]=sem_emo_dev['Intensity Class'].str.split(':', expand=True).apply(pd.to_numeric, errors='ignore')

    sem_emo_test[['emotion_ind', 'emo_desc']]=sem_emo_test['Intensity Class'].str.split(':', expand=True).apply(pd.to_numeric, errors='ignore')

    if convert_to_binary:
        sem_emo_train['emotion_ind'] = sem_emo_train['emotion_ind'].apply(to_emotion_ind)
        sem_emo_dev['emotion_ind'] = sem_emo_dev['emotion_ind'].apply(to_emotion_ind)
        sem_emo_test['emotion_ind'] = sem_emo_test['emotion_ind'].apply(to_emotion_ind)


    sem_emo_train['task'] = sem_emo_train['emotion']
    sem_emo_dev['task'] = sem_emo_dev['emotion']
    sem_emo_test['task'] = sem_emo_test['emotion']

    sem_emo_train['emotion'] = sem_emo_train['Intensity Class'].apply(to_emotion)
    sem_emo_dev['emotion'] = sem_emo_dev['Intensity Class'].apply(to_emotion)
    sem_emo_test['emotion'] = sem_emo_test['Intensity Class'].apply(to_emotion)




    return sem_emo_train, sem_emo_dev, sem_emo_test




#READING tweeteval hate data
def load_tweeteval_data(seed):
    hate_train_text = pd.read_csv('data/tweeteval_hate/train_text.txt', sep='\n', header=None, names=['tweet'], skip_blank_lines=False, quoting=csv.QUOTE_NONE)
    hate_train_labels = pd.read_csv('data/tweeteval_hate/train_labels.txt', sep='\n', header=None, names=['emotion_ind'])
    hate_train = hate_train_text.join(hate_train_labels)
    hate_train.dropna(inplace=True)

    hate_val_text = pd.read_csv('data/tweeteval_hate/val_text.txt', sep='\n', header=None, names=['tweet'], skip_blank_lines=False, quoting=csv.QUOTE_NONE)
    hate_val_labels = pd.read_csv('data/tweeteval_hate/val_labels.txt', sep='\n', header=None, names=['emotion_ind'])
    hate_val = hate_val_text.join(hate_val_labels)
    hate_val.dropna(inplace=True)

    hate_test_text = pd.read_csv('data/tweeteval_hate/test_text.txt', sep='\n', header=None, names=['tweet'], skip_blank_lines=False, quoting=csv.QUOTE_NONE)
    hate_test_labels = pd.read_csv('data/tweeteval_hate/test_labels.txt', sep='\n', header=None, names=['emotion_ind'])
    hate_test = hate_test_text.join(hate_test_labels)
    hate_test.dropna(inplace=True)

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
    sem_emo_train, sem_emo_val, sem_emo_test = load_semeval_data(True)
    hate_train, hate_val, hate_test = load_tweeteval_data(seed)

    ## CONCATENATE ALL DATASETS and clean tweets (remove mentions, emoticons etc)
    train_all = pd.concat([
                              sem_emo_train[['tweet', 'emotion', 'emotion_ind', 'task']],
                              olid_train[['tweet', 'emotion', 'emotion_ind', 'task']],
                              isarc_train[['tweet', 'emotion', 'emotion_ind', 'task']],
                              hate_train[['tweet', 'emotion', 'emotion_ind', 'task']]])
    train_all.tweet = train_all.tweet.apply(give_emoji_free_text).apply(sanitize)


    val_all = pd.concat([
                             sem_emo_val[['tweet', 'emotion', 'emotion_ind', 'task']],
                             olid_val[['tweet', 'emotion', 'emotion_ind', 'task']],
                             isarc_val[['tweet', 'emotion', 'emotion_ind', 'task']],
                             hate_val[['tweet', 'emotion', 'emotion_ind', 'task']]])
    val_all.tweet = val_all.tweet.apply(give_emoji_free_text).apply(sanitize)

    test_all = pd.concat([
                             sem_emo_test[['tweet', 'emotion', 'emotion_ind', 'task']],
                             olid_test[['tweet', 'emotion', 'emotion_ind', 'task']],
                             isarc_test[['tweet', 'emotion', 'emotion_ind', 'task']],
                             hate_test[['tweet', 'emotion', 'emotion_ind', 'task']]])
    test_all.tweet = test_all.tweet.apply(give_emoji_free_text).apply(sanitize)

    if not SILENT:
        ## PRINT COUNTS PER EMOTION
        print('*' * 20)
        print('train_all')
        print(train_all['task'].value_counts())

        print('*' * 20)
        print('val_all')
        print(val_all['task'].value_counts())

        print('*' * 20)
        print('test_all')
        print(test_all['task'].value_counts())

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

        loaders_length = []
        self.tasks_dataset_iter={}
        for task_name, task_loaders in self.tasks_dataset.items():
            class_loader={}
            for class_name, loader in task_loaders.items():
                loaders_length.append(len(loader))
                class_loader[class_name]=iter(loader)
            self.tasks_dataset_iter[task_name]=class_loader
        self.max_length = max(loaders_length)

    def __len__(self):

        return self.max_length

    def __getitem__(self, idx):
        sample_batch_all={}
        for task_name, task_loaders in self.tasks_dataset_iter.items():
            sapmle_batch_oneclass={}
            for class_name, loader in task_loaders.items():
                try:
                    sample_batch = next(loader)
                    sapmle_batch_oneclass[class_name]=sample_batch
                except StopIteration:
                    self.tasks_dataset_iter[task_name][class_name] = iter(self.tasks_dataset[task_name][class_name])
                    sample_batch = next(self.tasks_dataset_iter[task_name][class_name])
                    sapmle_batch_oneclass[class_name]=sample_batch
            sample_batch_all[task_name]=sapmle_batch_oneclass

        return sample_batch_all

class EmotionDataset(Dataset):

    def __init__(self, dataset, tokenizer, max_len, opts):

        self.dataset = dataset
        self.targets = dataset.emotion_ind.to_numpy()
        self.tasks = dataset.task.to_numpy()
        self.tweets = dataset.tweet.to_numpy()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.opts = opts

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
        collate_fn=collation_fn_emo,
        shuffle=True,
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

def creat_metadataLoader(dataset, tokenizer, max_len, tasks_selected, num_sample_perclass, opts):
    tasks_dataloader = {}
    for emo in tasks_selected:
        dataset_emo = dataset[dataset['task'] == emo]

        # split dataset according to the class
        class_name = dataset_emo.emotion.unique()
        dataset_classes_loader = {}
        for c_n in class_name:
            dataset_oneclass = dataset_emo[dataset_emo["emotion"]==c_n]
            emo_dataset_oneclass = EmotionDataset(dataset_oneclass, tokenizer, max_len, opts)
            emo_dataloader = DataLoader(emo_dataset_oneclass, batch_size=num_sample_perclass, num_workers=0, shuffle=True)
            dataset_classes_loader[c_n] = emo_dataloader

        tasks_dataloader[emo] = dataset_classes_loader

    meatdata = MetaDataset(tasks_dataloader, tasks_selected)
    meatdata_loader = DataLoader(meatdata, batch_size=1, num_workers=0, shuffle=True, collate_fn=collation_fn_meta)
    return meatdata_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def collation_fn_meta(batch_tasks):

    batch_tasks_merge = {}
    for task_name, data_task in batch_tasks[0].items():
        support = {
            'tweet_text': (),
            'input_ids': torch.tensor([],dtype=torch.int64).to(device),
            'attention_mask': torch.tensor([],dtype=torch.int64).to(device),
            'targets': torch.tensor([],dtype=torch.int64).to(device),
            'task': ()
        }

        query = {
            'tweet_text': (),
            'input_ids': torch.tensor([],dtype=torch.int64).to(device),
            'attention_mask': torch.tensor([],dtype=torch.int64).to(device),
            'targets': torch.tensor([],dtype=torch.int64).to(device),
            'task': ()
        }

        for class_name, data_task_oneClass in data_task.items():
            (tweet, input_ids, attention_mask, targets, task) = data_task_oneClass
            length = len(tweet)
            support_length = int(length/2)


            support['tweet_text'] = support['tweet_text']+tweet[0:support_length]
            support['input_ids'] = torch.cat((support['input_ids'],input_ids[0:support_length]))
            support['attention_mask'] = torch.cat((support['attention_mask'], attention_mask[0:support_length]))
            support['targets'] = torch.cat((support['targets'], targets[0:support_length]))
            support['task'] = support['task']+task[0:support_length]

            query['tweet_text'] = query['tweet_text']+tweet[support_length:]
            query['input_ids'] = torch.cat((query['input_ids'],input_ids[support_length:]))
            query['attention_mask'] = torch.cat((query['attention_mask'], attention_mask[support_length:]))
            query['targets'] = torch.cat((query['targets'], targets[support_length:]))
            query['task'] = query['task']+task[support_length:]

        batch_tasks_merge[task_name]={
                                      "support": support,
                                      "query": query
                                      }

    return batch_tasks_merge

