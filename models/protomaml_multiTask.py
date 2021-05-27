import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
from copy import deepcopy
from .maml import MAML_framework
from scipy.spatial.distance import cosine
import os
import logging
import json

class ProtoMAML_multiTask_framework(MAML_framework):
    def __init__(self, args, classifier):
        super(ProtoMAML_multiTask_framework, self).__init__(args, classifier)

    def generate_newModel_instance(self, x_support_set, y_support_set, emo):

        # generate all prototypes
        classifier_new = deepcopy(self.classifier_init)

        c_prototypes = self.get_allPrototypes(x_support_set, y_support_set,classifier_new) #(num_class, dim_encoder_hidden)
        classifier_new.fc_layer_allTask[emo].weight.data = c_prototypes*2.0 #(num_class, dim_encoder_hidden)
        classifier_new.fc_layer_allTask[emo].bias.data = -1.0*c_prototypes.norm(dim=1)**2 #(num_class,)

        return classifier_new

    def get_allPrototypes(self,x_support_set, y_support_set,classifier_new):
        """
         calculate the prototype for all class
        :param x_support_set: (batch_size,sent_len)
        :param y_support_set: (batch,)
        :return:
        """
        classifier_new.eval()
        all_prototypes_mix = classifier_new.encoder(x_support_set[0],x_support_set[1]).pooler_output #(batch_size,dim_encoder_hidden)
        all_prototypes_mix = all_prototypes_mix.detach()
        classifier_new.train()

        all_prototypes = []
        for class_id in range(self.args.num_class):
            y_support_set_numpy = y_support_set.cpu().numpy()
            indexs = np.argwhere(y_support_set_numpy==class_id)
            prototypes_selected = torch.index_select(all_prototypes_mix,0,torch.tensor(indexs.reshape(-1)).to(self.args.device))
            prototype = torch.mean(prototypes_selected,dim=0)
            all_prototypes.append(prototype)
        all_prototypes = torch.stack(all_prototypes)


        return all_prototypes

    def update_Prototypes_forEpisodeModel(self, x_query_set, y_query_set, emo):

        self.classifier_episode.eval()
        all_prototypes_mix = self.classifier_episode.encoder(x_query_set[0],x_query_set[1]).pooler_output #(batch_size,dim_encoder_hidden)
        # all_prototypes_mix = all_prototypes_mix.detach()
        self.classifier_episode.train()

        all_prototypes = []
        for class_id in range(self.args.num_class):
            y_support_set_numpy = y_query_set.cpu().numpy()
            indexs = np.argwhere(y_support_set_numpy==class_id)
            prototypes_selected = torch.index_select(all_prototypes_mix,0,torch.tensor(indexs.reshape(-1)).to(self.args.device))
            prototype = torch.mean(prototypes_selected,dim=0)
            all_prototypes.append(prototype)
        all_prototypes = torch.stack(all_prototypes)

        self.classifier_episode.fc_layer_allTask[emo].weight.data = all_prototypes*2.0 #(num_class, dim_encoder_hidden)
        self.classifier_episode.fc_layer_allTask[emo].bias.data = -1.0*all_prototypes.norm(dim=1)**2 #(num_class,)


    def train_protomaml(self, data_iter_meta_train, criterion):

        best_acc_query = 0
        best_acc_epoch_query = 0
        for epoch in range(self.args.num_epoch_meta):
            print(f"-----------------Epoch: {epoch}/{self.args.num_epoch_meta}-----------------------")
            logging.info(f"-----------------Epoch: {epoch}/{self.args.num_epoch_meta}-----------------------")


            loss_query_oneEpoch, acc_query_oneEpoch = self.train_protomaml_epoch( data_iter_meta_train, criterion)

            print(f" epoch: {epoch}, query loss: {np.mean(loss_query_oneEpoch)}, query acc: {np.mean(acc_query_oneEpoch)} ")
            logging.info(f"epoch: {epoch}, query loss: {np.mean(loss_query_oneEpoch)}, query acc: {np.mean(acc_query_oneEpoch)}")


            if best_acc_query < np.mean(acc_query_oneEpoch):
                torch.save(self.classifier_init,
                           os.path.join(self.args.output_dir, "models", f'best_{self.args.emotion}_init_model.pkl'))
                best_acc_query = np.mean(acc_query_oneEpoch)
                best_acc_epoch_query = epoch
                print(f"update best_acc_query: {best_acc_query}, save best model")
                logging.info(f"update best_acc_query: {best_acc_query}, save best model")

            # save similarity
            s_path=os.path.join(self.args.output_dir, "gradient_similarities", "similarity_meta_init.json")
            with open(s_path, 'w') as f:
                f.write(json.dumps(self.similarity))

        print(f"best_acc_query: {best_acc_query} at epoch: {best_acc_epoch_query}")
        logging.info(f"best_acc_val: {best_acc_query} at epoch: {best_acc_epoch_query}")




    def train_protomaml_epoch(self, data_iter_train, criterion):
        loss_oneEpoch = []
        acc_oneEpoch = []
        # sample batch of tasks
        for indx_batch_tasks, data_batch_tasks in enumerate(data_iter_train):

            self.gradient_allTasks = {}
            self.gradient_contact_allTasks={}
            for task_name in data_batch_tasks.keys():
                self.gradient_allTasks[task_name] = {}
                self.gradient_contact_allTasks[task_name] = {
                    "all_model": torch.tensor([]).to("cpu"),
                    "encoder":torch.tensor([]).to("cpu"),
                    "fc_layer":torch.tensor([]).to("cpu")
                }

            grads_batch_tasks = {}
            loss_batch_tasks = []
            acc_batch_tasks = []

            #for each task/episode
            for task_name, data_per_task in data_batch_tasks.items():

                support_set, query_set = data_per_task['support'],  data_per_task['query']
                x_support_set, y_support_set = (support_set['input_ids'], support_set['attention_mask'], support_set['task'][0]), support_set['targets']

                # copy model
                self.classifier_episode = self.generate_newModel_instance(x_support_set, y_support_set, task_name)
                optimizer_task = optim.SGD(filter(lambda p: p.requires_grad, self.classifier_episode.parameters()), lr=self.args.lr_alpha)

                for n, fc_layer in self.classifier_episode.fc_layer_allTask.items():
                    if n == task_name:
                        for p in fc_layer.parameters():
                            p.requires_grad = True
                    else:
                        for p in fc_layer.parameters():
                            p.requires_grad = False

                # train episode
                grads_query, loss_query, accuracy_query = self.train_episode(criterion, data_per_task, optimizer_task)

                # accumulate loss_query and acc
                loss_batch_tasks.append(loss_query)
                acc_batch_tasks.append(accuracy_query)

                loss_oneEpoch.append(loss_query)
                acc_oneEpoch.append(accuracy_query)


                # accumulate grads_query
                if grads_batch_tasks == {}:
                    for ind, (name, para) in enumerate(filter(lambda n_p : n_p[1].requires_grad, self.classifier_episode.named_parameters())):
                        grads_batch_tasks[name] = grads_query[ind]
                else:
                    for ind, (name, para) in enumerate(filter(lambda n_p : n_p[1].requires_grad, self.classifier_episode.named_parameters())):
                        if "fc_layer" not in name:
                            grads_batch_tasks[name] += grads_query[ind]
                        else:
                            grads_batch_tasks[name] = grads_query[ind]

            if indx_batch_tasks % 10 == 0 and len(data_batch_tasks.keys())>=2:
                self.calculate_Similarity()

            #update initial parameters
            self.update_model_init_parameters(grads_batch_tasks)

            print("indx_batch_tasks:", indx_batch_tasks," loss:", np.mean(loss_batch_tasks), " acc:", np.mean(acc_batch_tasks))

        return loss_oneEpoch, acc_oneEpoch

    def calculate_Similarity(self):
        for t_n, gs in self.gradient_allTasks.items():
            for g_n, g in gs.items():
                if "fc_layer" in g_n:
                    self.gradient_contact_allTasks[t_n]["fc_layer"] = torch.cat((self.gradient_contact_allTasks[t_n]["fc_layer"],self.gradient_allTasks[t_n][g_n].flatten()))
                else:
                    self.gradient_contact_allTasks[t_n]["encoder"] = torch.cat((self.gradient_contact_allTasks[t_n]["encoder"],self.gradient_allTasks[t_n][g_n].flatten()))

                self.gradient_contact_allTasks[t_n]["all_model"] = torch.cat((self.gradient_contact_allTasks[t_n]["all_model"],self.gradient_allTasks[t_n][g_n].flatten()))

        #calculate cos similarity
        for m in ["all_model","encoder","fc_layer"]:
            for e1, e2 in itertools.combinations(self.gradient_contact_allTasks.keys(), 2):
                s =  1 - cosine(self.gradient_contact_allTasks[e1][m].numpy(), self.gradient_contact_allTasks[e2][m].numpy())
                self.similarity[e1+"-"+e2][m].append(s)


    def train_episode(self, criterion, data_per_task, optimizer_task):
        self.classifier_episode.train()
        support_set, query_set = data_per_task['support'],  data_per_task['query']
        x_support_set, y_support_set = (support_set['input_ids'], support_set['attention_mask'], support_set["task"][0]), support_set['targets']
        x_query_set, y_query_set = (query_set['input_ids'],query_set['attention_mask'], query_set['task'][0]), query_set['targets']

        for i in range(self.args.train_step_per_episode):
            preds_support = self.classifier_episode(*x_support_set)
            loss_support = criterion(preds_support,y_support_set)

            optimizer_task.zero_grad()
            loss_support.backward()
            optimizer_task.step()

            for ind, (name, para) in enumerate(filter(lambda n_p : n_p[1].requires_grad, self.classifier_episode.named_parameters())):
                self.gradient_allTasks[support_set["task"][0]][name] = para.grad.detach().cpu()

        #generate loss for query set
        # self.update_Prototypes_forEpisodeModel(x_query_set, y_query_set, query_set["task"][0])
        preds_query = self.classifier_episode(*x_query_set)
        loss_query = criterion(preds_query, y_query_set)
        accuracy_query = torch.sum(torch.argmax(preds_query,dim=-1) == y_query_set) / len(y_query_set)

        optimizer_task.zero_grad()
        grads_query = torch.autograd.grad(loss_query, filter(lambda p: p.requires_grad, self.classifier_episode.parameters()))

        return grads_query, loss_query.cpu().detach().numpy(), accuracy_query.cpu().detach().numpy()

    def eval_init_model(self, data_iter_meta_val, data_iter_val,criterion):

        data_batch_tasks_forTraining=next(iter(data_iter_meta_val))

        acc_val_allTask=[]
        loss_val_allTask=[]
        for task_name, data_per_task in data_batch_tasks_forTraining.items():

            support_set, query_set = data_per_task['support'],  data_per_task['query']
            x_support_set, y_support_set = (support_set['input_ids'], support_set['attention_mask'], support_set['task'][0]), support_set['targets']

            # copy model
            self.classifier_episode = self.generate_newModel_instance(x_support_set, y_support_set, task_name)
            optimizer_task = optim.SGD(filter(lambda p: p.requires_grad, self.classifier_episode.parameters()), lr=self.args.lr_alpha)

            for n, fc_layer in self.classifier_episode.fc_layer_allTask.items():
                if n == task_name:
                    for p in fc_layer.parameters():
                        p.requires_grad = True
                else:
                    for p in fc_layer.parameters():
                        p.requires_grad = False

            # train one batch
            for i in range(self.args.train_step_per_episode):
                preds_support = self.classifier_episode(*x_support_set)
                loss_support = criterion(preds_support,y_support_set)

                optimizer_task.zero_grad()
                loss_support.backward()
                optimizer_task.step()

            # val
            correct_num = 0
            loss_oneTask = []
            n_sample = 0
            dataloader = data_iter_val[task_name]
            for batch_data in dataloader:
                input_ids = batch_data['input_ids']
                attention_mask = batch_data['attention_mask']
                y_target = batch_data['targets']
                input_data = (input_ids,attention_mask,task_name)
                preds = self.classifier_episode(*input_data)
                loss = criterion(preds,y_target)
                loss_oneTask.append(loss.cpu().detach().numpy())
                correct_num += torch.sum(torch.argmax(preds,dim=-1) == y_target)
                n_sample += len(y_target)
            loss_oneTask = np.mean(loss_oneTask)
            loss_val_allTask.append(loss_oneTask)
            accuracy_val = (correct_num / n_sample).cpu().detach().numpy()
            acc_val_allTask.append(accuracy_val)

        return np.mean(acc_val_allTask), np.mean(loss_val_allTask)

    # def train_initmodel(self):
    #
    #     # load best init model
    #     model_path = os.path.join(self.args.output_dir, "models", f'best_{self.args.emotion}_init_model.pkl')
    #     bestInit_model=torch.load(model_path)

