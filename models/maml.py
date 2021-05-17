import torch
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy
import numpy as np

class MAML_framework(nn.Module):
    def __init__(self, args, classifier):
        super(MAML_framework, self).__init__()
        self.args = args
        self.classifier_init = classifier

    def generate_newModel_instance(self):
        classifier_new = deepcopy(self.classifier_init)
        return classifier_new

    def train_maml(self, data_iter_train, criterion):

        for epoch in range(self.args.num_epoch) :

            # sample batch of tasks
            for indx_batch_tasks, data_batch_tasks in enumerate(data_iter_train):

                grads_batch_tasks = {}
                loss_batch_tasks = []
                acc_batch_tasks = []
                #for each task/episode
                for data_per_task in data_batch_tasks:

                    # copy model
                    self.classifier_episode = self.generate_newModel_instance()
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

                print("epoch:", epoch, "indx_batch_tasks:", indx_batch_tasks," loss:", np.mean(loss_batch_tasks), " acc:", np.mean(acc_batch_tasks))


    def train_episode(self, criterion, data_per_task, optimizer_task):

        support_set, query_set = data_per_task
        x_support_set, y_support_set = (support_set['input_ids'], support_set['attention_mask']), support_set['targets']
        x_query_set, y_query_set = (query_set['input_ids'],query_set['attention_mask']), query_set['targets']

        for i in range(self.args.train_step_per_episode):
            preds_support = self.classifier_episode(*x_support_set)
            loss_support = criterion(preds_support,y_support_set)
            
            # Print batch if loss is NaN
            if loss_support != loss_support:
                print("\nNAN LOSS!")
                for s in x_support_set:
                    print(list(s))
                raise RuntimeError("NaN loss")

            optimizer_task.zero_grad()
            loss_support.backward()
            optimizer_task.step()

        #generate loss for query set
        preds_query = self.classifier_episode(*x_query_set)
        loss_query = criterion(preds_query, y_query_set)
        accuracy_query = torch.sum(torch.argmax(preds_query,dim=-1) == y_query_set) / len(y_query_set)

        optimizer_task.zero_grad()
        grads_query = torch.autograd.grad(loss_query, filter(lambda p: p.requires_grad, self.classifier_episode.parameters()))

        return grads_query, loss_query.cpu().detach().numpy(), accuracy_query.cpu().detach().numpy()


    def update_model_init_parameters(self, grads_all_tasks):
        for name, grad in grads_all_tasks.items():
            self.classifier_init.state_dict()[name] -= grad*self.args.lr_beta

