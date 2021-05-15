import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy
from .maml import MAML_framework

class ProtoMAML_framework(MAML_framework):
    def __init__(self, args, classifier):
        super(ProtoMAML_framework, self).__init__(args, classifier)

    def generate_newModel_instance(self, x_support_set, y_support_set):
        classifier_new = deepcopy(self.classifier_init)

        # generate all prototypes
        c_prototypes = self.get_allPrototypes(x_support_set, y_support_set) #(num_class, dim_encoder_hidden)

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
        all_prototypes_mix = self.classifier_init.encoder(*x_support_set).pooler_output #(batch_size,dim_encoder_hidden)

        all_prototypes = []
        for class_id in range(self.args.num_class):
            y_support_set_numpy = y_support_set.cpu().numpy()
            indexs = np.argwhere(y_support_set_numpy==class_id)
            prototypes_selected = torch.index_select(all_prototypes_mix,0,torch.tensor(indexs.reshape(-1)).to(self.args.device))
            prototype = torch.mean(prototypes_selected,dim=0)
            all_prototypes.append(prototype)
        all_prototypes = torch.stack(all_prototypes)

        all_prototypes_d = all_prototypes.detach()

        return all_prototypes_d

    def train_protomaml(self, data_iter_train, criterion):

        for epoch in range(self.args.num_epoch) :

            # sample batch of tasks
            for indx_batch_tasks, data_batch_tasks in enumerate(data_iter_train):

                grads_batch_tasks = {}
                loss_batch_tasks = []
                acc_batch_tasks = []
                #for each task/episode
                for data_per_task in data_batch_tasks:
                    support_set, query_set = data_per_task
                    x_support_set, y_support_set = (support_set['input_ids'], support_set['attention_mask']), support_set['targets']

                    # copy model
                    self.classifier_episode = self.generate_newModel_instance(x_support_set, y_support_set)
                    optimizer_task = optim.SGD(filter(lambda p: p.requires_grad, self.classifier_episode.parameters()), lr=self.args.lr_alpha)

                    # train episode
                    grads_query, loss_query, accuracy_query = self.train_episode(criterion, data_per_task, optimizer_task)

                    # accumulate loss_query and acc
                    loss_batch_tasks.append(loss_query)
                    acc_batch_tasks.append(accuracy_query)

                    # accumulate grads_query
                    if grads_batch_tasks == {}:
                        for ind, (name, para) in enumerate(self.classifier_episode.named_parameters()):
                            grads_batch_tasks[name] = grads_query[ind]
                    else:
                        for ind, (name, para) in enumerate(self.classifier_episode.named_parameters()):
                            grads_batch_tasks[name] += grads_query[ind]

                #update initial parameters
                self.update_model_init_parameters(grads_batch_tasks)

                print("indx_batch_tasks:", indx_batch_tasks," loss:", np.mean(loss_batch_tasks), " acc:", np.mean(acc_batch_tasks))

