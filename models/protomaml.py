import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import sys
import os

from copy import deepcopy
from .maml import MAML_framework
from scipy.spatial.distance import cosine

class ProtoMAML_framework(MAML_framework):
    def __init__(self, args, classifier):
        super(ProtoMAML_framework, self).__init__(args, classifier)

    def generate_newModel_instance(self, x_support_set, y_support_set):

        # generate all prototypes
        c_prototypes = self.get_allPrototypes(x_support_set, y_support_set) #(num_class, dim_encoder_hidden)

        classifier_new = deepcopy(self.classifier_init)
        classifier_new.fc_layer.weight.data = c_prototypes*2.0 #(num_class, dim_encoder_hidden)
        classifier_new.fc_layer.bias.data = -1.0*c_prototypes.norm(dim=1)**2 #(num_class,)

        return classifier_new

    def get_allPrototypes(self,x_support_set, y_support_set):
        """
         calculate the prototype for all class
        :param x_support_set: (batch_size,sent_len)
        :param y_support_set: (batch,)
        :return:
        """
        self.classifier_init.eval()
        all_prototypes_mix = self.classifier_init.encoder(*x_support_set).pooler_output #(batch_size,dim_encoder_hidden)
        all_prototypes_mix = all_prototypes_mix.detach()
        self.classifier_init.train()

        all_prototypes = []
        for class_id in range(self.args.num_class):
            y_support_set_numpy = y_support_set.cpu().numpy()
            indexs = np.argwhere(y_support_set_numpy==class_id)
            prototypes_selected = torch.index_select(all_prototypes_mix,0,torch.tensor(indexs.reshape(-1)).to(self.args.device))
            prototype = torch.mean(prototypes_selected,dim=0)
            all_prototypes.append(prototype)
        all_prototypes = torch.stack(all_prototypes)


        return all_prototypes

    def update_Prototypes_forEpisodeModel(self, x_query_set, y_query_set):

        self.classifier_episode.eval()
        all_prototypes_mix = self.classifier_episode.encoder(*x_query_set).pooler_output #(batch_size,dim_encoder_hidden)
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

        self.classifier_episode.fc_layer.weight.data = all_prototypes*2.0 #(num_class, dim_encoder_hidden)
        self.classifier_episode.fc_layer.bias.data = -1.0*all_prototypes.norm(dim=1)**2 #(num_class,)


    def train_protomaml(self, data_iter_train, criterion):
        
        gradient_data_outer = pd.DataFrame()
        gradient_data_inner = pd.DataFrame()
        
        for epoch in range(self.args.num_epoch) :
            # sample batch of tasks
            for indx_batch_tasks, data_batch_tasks in enumerate(data_iter_train):

                grads_batch_tasks = {}
                loss_batch_tasks = []
                acc_batch_tasks = []
                
                outer_grad_1 = {}
                outer_grad_2 = {}
                inner_grad_1 = {}
                inner_grad_2 = {}
                emotion_1 = ""
                emotion_2 = ""
                
                assert len(data_batch_tasks.values()) == 2 # gradient conflict code assumes two datasets are used

                #for each task/episode
                for data_per_task in data_batch_tasks.values():

                    support_set, query_set = data_per_task['support'],  data_per_task['query']
                    x_support_set, y_support_set = (support_set['input_ids'], support_set['attention_mask']), support_set['targets']

                    # copy model
                    self.classifier_episode = self.generate_newModel_instance(x_support_set, y_support_set)
                    optimizer_task = optim.SGD(filter(lambda p: p.requires_grad, self.classifier_episode.parameters()), lr=self.args.lr_alpha)

                    # train episode
                    grads_query, loss_query, accuracy_query, inner_grad = self.train_episode(criterion, data_per_task, optimizer_task)

                    # accumulate loss_query and acc
                    loss_batch_tasks.append(loss_query)
                    acc_batch_tasks.append(accuracy_query)

                    # accumulate grads_query
                    if grads_batch_tasks == {}:
                        inner_grad_1 = inner_grad
                        emotion_1 = support_set['task'][0]
                        for ind, (name, para) in enumerate(self.classifier_episode.named_parameters()):
                            grads_batch_tasks[name] = grads_query[ind]
                            outer_grad_1[name] = grads_query[ind].detach().cpu()
                    else:
                        inner_grad_2 = inner_grad
                        emotion_2 = support_set['task'][0]
                        for ind, (name, para) in enumerate(self.classifier_episode.named_parameters()):
                            grads_batch_tasks[name] += grads_query[ind]
                            outer_grad_2[name] = grads_query[ind].detach().cpu()

                #update initial parameters
                self.update_model_init_parameters(grads_batch_tasks)
                
                # compute gradient similarity
                episode_similarity_outer = {}
                episode_similarity_inner = {}
                for name, p in self.classifier_episode.named_parameters():
                    sim_outer = 1 - cosine(outer_grad_1[name].flatten(), outer_grad_2[name].flatten())
                    episode_similarity_outer[name] = sim_outer
                    sim_inner = 1 - cosine(inner_grad_1[name].flatten(), inner_grad_2[name].flatten())
                    episode_similarity_inner[name] = sim_inner
                gradient_data_outer = gradient_data_outer.append(episode_similarity_outer, ignore_index=True)
                gradient_data_inner = gradient_data_inner.append(episode_similarity_inner, ignore_index=True)
                save_dir = os.path.join("gradient_similarities", emotion_1 + "_" + emotion_2)
                os.makedirs(save_dir, exist_ok=True)
                gradient_data_outer.to_csv(os.path.join(save_dir, emotion_1 + "_" + emotion_2 + "_outer.csv"))
                gradient_data_inner.to_csv(os.path.join(save_dir, emotion_1 + "_" + emotion_2 + "_inner.csv"))

                print("indx_batch_tasks:", indx_batch_tasks," loss:", np.mean(loss_batch_tasks), " acc:", np.mean(acc_batch_tasks))

    def train_episode(self, criterion, data_per_task, optimizer_task):

        support_set, query_set = data_per_task['support'],  data_per_task['query']
        x_support_set, y_support_set = (support_set['input_ids'], support_set['attention_mask']), support_set['targets']
        x_query_set, y_query_set = (query_set['input_ids'],query_set['attention_mask']), query_set['targets']
        
        inner_grad = {}

        for i in range(self.args.train_step_per_episode):
            preds_support = self.classifier_episode(*x_support_set)
            loss_support = criterion(preds_support,y_support_set)

            optimizer_task.zero_grad()
            loss_support.backward()
            
            # Accumulate gradients
            for ind, (name, para) in enumerate(self.classifier_episode.named_parameters()):
                if name not in inner_grad:
                    inner_grad[name] = para.grad.detach().cpu()
                else:
                    inner_grad[name] += para.grad.detach().cpu()
            
            optimizer_task.step()

        #generate loss for query set
        self.update_Prototypes_forEpisodeModel(x_query_set, y_query_set)
        preds_query = self.classifier_episode(*x_query_set)
        loss_query = criterion(preds_query, y_query_set)
        accuracy_query = torch.sum(torch.argmax(preds_query,dim=-1) == y_query_set) / len(y_query_set)

        optimizer_task.zero_grad()
        grads_query = torch.autograd.grad(loss_query, filter(lambda p: p.requires_grad, self.classifier_episode.parameters()))

        return grads_query, loss_query.cpu().detach().numpy(), accuracy_query.cpu().detach().numpy(), inner_grad
