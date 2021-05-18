import torch
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy
import numpy as np
import tqdm
import logging
from transformers import RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
import os
class MAML_multiTask_framework(nn.Module):
    def __init__(self, args, classifier):
        super(MAML_multiTask_framework, self).__init__()
        self.args = args
        self.classifier_init = classifier

    def generate_newModel_instance(self, task_name):
        classifier_new = deepcopy(self.classifier_init)
        for t_name, fc in classifier_new.fc_layer_allTask.items():
            if t_name != task_name:
                fc.weight.requires_grad = False
                fc.bias.requires_grad = False
            else:
                fc.weight.data.zero_()
                fc.bias.data.zero_()
        # for ind, (name, para) in enumerate(classifier_new.named_parameters()):
        #     if "fc_layer_task" in name:
                # para.requires_grad = False
            # else:
            #     para.requires_grad = True
        return classifier_new

    def split_supportAndquery(self, dataset_all):
        support_length = int(len(dataset_all['targets'])/2)
        support = {
            'tweet_text': dataset_all['tweet_text'][0:support_length],
            'input_ids': dataset_all['input_ids'][0:support_length],
            'attention_mask': dataset_all['attention_mask'][0:support_length],
            'targets': dataset_all['targets'][0:support_length],
            'task': dataset_all['task'][0:support_length]
        }

        query = {
            'tweet_text': dataset_all['tweet_text'][support_length:],
            'input_ids': dataset_all['input_ids'][support_length:],
            'attention_mask': dataset_all['attention_mask'][support_length:],
            'targets': dataset_all['targets'][support_length:],
            'task': dataset_all['task'][support_length:]
        }

        return (support, query)

    def train_maml_epoch(self, data_iter_train, criterion):
        losses_all = {}
        correct_predictions_all = {}
        n_examples_all = {}
        for emotion in data_iter_train.keys():
            losses_all[emotion] = []
            correct_predictions_all[emotion] = 0
            n_examples_all[emotion] = 0

        for data_batch_tasks in zip(*data_iter_train.values()):
            grads_batch_tasks = {}
            step_g = 0
            #for each task/episode
            for indx, data_per_task in enumerate(data_batch_tasks):
                step_g += 1
                task_name = data_per_task['task'][0]

                # split support set and query set
                data_per_task = self.split_supportAndquery(data_per_task)

                # copy model
                self.classifier_init.train()
                self.classifier_episode = self.generate_newModel_instance(task_name)
                self.classifier_episode.train()
                # optimizer_task = optim.SGD(filter(lambda p: p.requires_grad, self.classifier_episode.parameters()), lr=self.args.lr_alpha)
                optimizer_task = AdamW(filter(lambda p: p.requires_grad, self.classifier_episode.parameters()), lr=self.args.lr_alpha, correct_bias=False)
                # train episode
                grads_query, loss_query, correct_num_query, num_query = self.train_episode(criterion, data_per_task, optimizer_task)

                # accumulate loss_query and acc
                correct_predictions_all[task_name] += correct_num_query
                n_examples_all[task_name] += num_query
                losses_all[task_name].append(loss_query.item())


                # accumulate grads_query
                if grads_batch_tasks == {}:
                    for ind, (name, para) in enumerate(filter(lambda n_p : n_p[1].requires_grad, self.classifier_episode.named_parameters())):
                        if "fc_layer" not  in name:
                            grads_batch_tasks[name] = grads_query[ind]
                else:
                    for ind, (name, para) in enumerate(filter(lambda n_p : n_p[1].requires_grad, self.classifier_episode.named_parameters())):
                        if "fc_layer" not  in name:
                            grads_batch_tasks[name] += grads_query[ind]

            #update initial parameters
            self.update_model_init_parameters(grads_batch_tasks,step_g)

        for k,v in correct_predictions_all.items():
            correct_predictions_all[k] = (v.double() / n_examples_all[k]).item()
            losses_all[k] = np.mean(losses_all[k])

        return correct_predictions_all, losses_all

    def train_episode(self, criterion, data_per_task, optimizer_task):

        support_set, query_set = data_per_task
        x_support_set, y_support_set = (support_set['input_ids'], support_set['attention_mask'], support_set['task'][0]), support_set['targets']
        x_query_set, y_query_set = (query_set['input_ids'],query_set['attention_mask'], query_set['task'][0]), query_set['targets']

        for i in range(self.args.train_step_per_episode):
            preds_support = self.classifier_episode(*x_support_set)
            loss_support = criterion(preds_support,y_support_set)

            optimizer_task.zero_grad()
            loss_support.backward()
            optimizer_task.step()

        #generate loss for query set
        preds_query = self.classifier_episode(*x_query_set)
        loss_query = criterion(preds_query, y_query_set)
        correct_query_num = torch.sum(torch.argmax(preds_query,dim=-1) == y_query_set)

        optimizer_task.zero_grad()
        grads_query = torch.autograd.grad(loss_query, filter(lambda p: p.requires_grad, self.classifier_episode.parameters()))

        return grads_query, loss_query, correct_query_num, len(y_query_set)

    def update_model_init_parameters(self, grads_all_tasks, step_g):
        for name, grad in grads_all_tasks.items():
            self.classifier_init.state_dict()[name] -= grad*self.args.lr_beta*(1.0/step_g)

    def eval_maml_epoch(self, train_data_loader, val_data_loader, criterion):
        classifier_temp = deepcopy(self.classifier_init)
        for t_name, fc in classifier_temp.fc_layer_allTask.items():
            fc.weight.data.zero_()
            fc.bias.data.zero_()

        self.trainAndeval_classifier(classifier_temp, train_data_loader, val_data_loader, criterion)


    def eval_model(self,model, data_loader, loss_fn):
        model = model.eval()

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

        return acc_all, loss_all

    def trainAndeval_classifier(self, classifier_temp, train_data_loader, val_data_loader, criterion):
        logging.info('----------------Start eval trained maml model--------------------')
        print('------------------Start eval trained maml model----------------------')

        def train_epoch(
                model,
                data_loader,
                loss_fn,
                optimizer,
                scheduler,
        ):
            model = model.train()

            losses_all = {}
            correct_predictions_all = {}
            n_examples_all = {}
            for emotion in data_loader.keys():
                losses_all[emotion] = []
                correct_predictions_all[emotion] = 0
                n_examples_all[emotion] = 0


            for ds in zip(*data_loader.values()):
                for d in ds:
                    emo = d["task"][0]

                    input_ids = d["input_ids"]
                    attention_mask = d["attention_mask"]
                    targets = d["targets"]

                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        emotion = emo
                    )

                    _, preds = torch.max(outputs, dim=1)
                    loss = loss_fn(outputs, targets)

                    correct_predictions_all[emo] += torch.sum(preds == targets)
                    n_examples_all[emo] += len(targets)
                    losses_all[emo].append(loss.item())

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()

            for k,v in correct_predictions_all.items():
                correct_predictions_all[k] = (v.double() / n_examples_all[k]).item()
                losses_all[k] = np.mean(losses_all[k])

            return correct_predictions_all, losses_all


        val_acc, val_loss = self.eval_model(
            classifier_temp,
            val_data_loader,
            criterion
        )
        logging.info(f'Val loss {val_loss} accuracy {val_acc}')
        print(f'Val loss {val_loss} accuracy {val_acc}')

        optimizer_temp = AdamW(filter(lambda p: p.requires_grad, classifier_temp.parameters()), lr=self.args.lr_alpha, correct_bias=False)
        total_steps = min([len(loader) for loader in train_data_loader.values()]) * self.args.classifier_epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer_temp,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )


        for epoch in range(self.args.classifier_epochs):

            logging.info(f'--------train initialization model Epoch {epoch + 1}/{self.args.classifier_epochs}---------')
            print(f'--------train initialization model Epoch {epoch + 1}/{self.args.classifier_epochs}--------')

            train_acc, train_loss = train_epoch(
                classifier_temp,
                train_data_loader,
                criterion,
                optimizer_temp,
                scheduler
            )

            logging.info(f'Train loss {train_loss} accuracy {train_acc}')
            print(f'Train loss {train_loss} accuracy {train_acc}')

            val_acc, val_loss = self.eval_model(
                classifier_temp,
                val_data_loader,
                criterion
            )

            logging.info(f'Val loss {val_loss} accuracy {val_acc}')
            print(f'Val loss {val_loss} accuracy {val_acc}')

            for k, acc in val_acc.items():
                if acc > self.best_accuracy[k]:
                    torch.save(classifier_temp,
                               os.path.join(self.args.output_dir, "models",f'best_{self.args.emotion}_{k}_model.pkl'))
                    torch.save(self.classifier_init,
                               os.path.join(self.args.output_dir, "models",f'best_{self.args.emotion}_{k}_InitializationModel.pkl'))
                    self.best_accuracy[k] = acc

        logging.info('-----------------End eval trained maml model-------------------')
        print('--------------------End eval trained maml model---------------------')

    def test_classifier(self, test_data_loader, args, loss_fn):
        logging.info('-----------------start test trained maml model-------------------')
        print('--------------------start test trained maml model---------------------')
        for emo, data_loader in test_data_loader.items():
            #load best model
            model_path = os.path.join(args.output_dir, "models",f'best_{args.emotion}_{emo}_model.pkl')
            best_model=torch.load(model_path)

            test_acc, test_loss = self.eval_model(
                best_model,
                test_data_loader,
                loss_fn
            )



            logging.info(f'test loss {emo} {test_loss[emo]} accuracy {emo} {test_acc[emo]}')
            print(f'test loss {emo} {test_loss[emo]} accuracy {emo} {test_acc[emo]}')
        logging.info('-----------------end test trained maml model-------------------')
        print('--------------------end test trained maml model---------------------')