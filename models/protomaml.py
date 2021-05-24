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
        c_prototypes = self.get_allPrototypes(x_support_set, y_support_set).detach() #(num_class, dim_encoder_hidden)

        classifier_new = deepcopy(self.classifier_init)
        classifier_new.fc_layer.weight.data = c_prototypes*2.0 #(num_class, dim_encoder_hidden)
        classifier_new.fc_layer.bias.data = -1.0*c_prototypes.norm(dim=1)**2 #(num_class,)

        return classifier_new

    def get_allPrototypes(self, x_support_set, y_support_set):
        """
         calculate the prototype for all class
        :param x_support_set: (batch_size,sent_len)
        :param y_support_set: (batch,)
        :return:
        """
        #self.classifier_init.eval()
        all_prototypes_mix = self.classifier_init.encoder(*x_support_set).pooler_output #(batch_size,dim_encoder_hidden)
        #all_prototypes_mix = all_prototypes_mix.detach()
        #self.classifier_init.train()

        all_prototypes = []
        y_support_set_numpy = y_support_set.cpu().numpy()
        for class_id in range(self.args.num_class):
            indexs = np.argwhere(y_support_set_numpy==class_id)
            prototypes_selected = torch.index_select(all_prototypes_mix,0,torch.tensor(indexs.reshape(-1)).to(self.args.device))
            #prototypes_selected = all_prototypes_mix[torch.nonzero(y_support_set == class_id)].squeeze()
            prototype = torch.mean(prototypes_selected,dim=0)
            all_prototypes.append(prototype)
        all_prototypes = torch.stack(all_prototypes)

        return all_prototypes


    def train_protomaml(self, data_iter_train, criterion):
        
        grad_sim = pd.DataFrame(columns = ["sim_k", "sim_init", "sim_both"])
        
        optimizer_meta = optim.SGD(self.classifier_init.encoder.parameters(), lr=self.args.lr_beta)
        
        for epoch in range(self.args.num_epoch) :
            # sample batch of tasks
            for indx_batch_tasks, data_batch_tasks in enumerate(data_iter_train):

                loss_batch_tasks = []
                acc_batch_tasks = []
                
                emotion_1 = list(data_batch_tasks.keys())[0]
                emotion_2 = list(data_batch_tasks.keys())[1]
                task_grads_k = []
                task_grads_init = []
                task_grads_both = []
                
                assert len(data_batch_tasks.values()) == 2 # gradient conflict code assumes two datasets are used
                
                # Compute gradients for random batch of each dataset
                # This is not used to update the model, only as a reference measure
                '''ref_batch = next(iter(data_iter_train))
                first = True
                for data_per_task in ref_batch.values():
                    support_set, query_set = data_per_task['support'],  data_per_task['query']
                    x_support_set, y_support_set = (support_set['input_ids'], support_set['attention_mask']), support_set['targets']
                    self.classifier_episode = self.generate_newModel_instance(x_support_set, y_support_set)
                    optimizer_task = optim.SGD(filter(lambda p: p.requires_grad, self.classifier_episode.parameters()), lr=self.args.lr_alpha)
                    outer_grad, _, _, inner_grad = self.train_episode(criterion, data_per_task, optimizer_task)
                    ref_grad_1 = [g.detach().flatten() for g in outer_grad]
                    break'''

                #for each task/episode
                for t, data_per_task in enumerate(data_batch_tasks.values()):

                    support_set, query_set = data_per_task['support'],  data_per_task['query']
                    x_support_set, y_support_set = (support_set['input_ids'], support_set['attention_mask']), support_set['targets']

                    # copy model
                    self.classifier_episode = self.generate_newModel_instance(x_support_set, y_support_set)
                    optimizer_task = optim.SGD(filter(lambda p: p.requires_grad, self.classifier_episode.parameters()), lr=self.args.lr_alpha)

                    # train episode
                    grads_query_k, loss_query, accuracy_query, encoded_query = self.train_episode(criterion, data_per_task, optimizer_task)

                    # accumulate loss_query and acc
                    loss_batch_tasks.append(loss_query)
                    acc_batch_tasks.append(accuracy_query)
                    
                    assert len(grads_query_k) == len(list(self.classifier_episode.named_parameters()))
                    
                    
                    
                    # Calculate prototype initialization (with gradients)
                    prototypes = self.get_allPrototypes(x_support_set, y_support_set)
                    
                    # Reintroduce prototypes to computation graph
                    p_weight = prototypes*2.0 #(num_class, dim_encoder_hidden)
                    p_bias = -1.0*prototypes.norm(dim=1)**2 #(num_class,)
                    #self.classifier_episode.fc_layer.weight.data = p_weight + (self.classifier_episode.fc_layer.weight.data - p_weight).detach()
                    #self.classifier_episode.fc_layer.bias.data = p_bias + (self.classifier_episode.fc_layer.bias.data - p_bias).detach()
                    #self.classifier_episode.fc_layer.weight.data = torch.nn.Parameter(p_weight, requires_grad=True)
                    #self.classifier_episode.fc_layer.bias.data = torch.nn.Parameter(p_bias, requires_grad=True)
                    #print(self.classifier_episode.fc_layer.weight.is_leaf) # TRUE
                    
                    fc_weight = p_weight + (self.classifier_episode.fc_layer.weight.data - p_weight).detach()
                    fc_bias = p_bias + (self.classifier_episode.fc_layer.bias.data - p_bias).detach()
                    
                    # Forward encoded query set through fc layer
                    #preds_query = self.classifier_init.fc_layer(encoded_query)
                    preds_query = encoded_query @ fc_weight.T + fc_bias
                    
                    # Calculate gradients for base model
                    y_query_set = query_set['targets']
                    loss_query = criterion(preds_query, y_query_set)
                    grads_query_init = torch.autograd.grad(loss_query, self.classifier_init.encoder.parameters())
                    
                    # Save gradients for conflict calculation
                    grads_k = []
                    grads_init = []
                    grads_both = []
                    for ind, (name, _) in enumerate(self.classifier_episode.encoder.named_parameters()):
                        grads_k.append(grads_query_k[ind].detach().flatten())
                        grads_init.append(grads_query_init[ind].detach().flatten())
                        grads_both.append(grads_query_k[ind].detach().flatten() + grads_query_init[ind].detach().flatten())
                    task_grads_k.append(torch.cat(grads_k))
                    task_grads_init.append(torch.cat(grads_init))
                    task_grads_both.append(torch.cat(grads_both))
                    
                    # Store in model
                    for ind, param in enumerate(self.classifier_init.encoder.parameters()):
                        grad_query = grads_query_k[ind] + grads_query_init[ind]
                        if param.grad is None:
                            param = grad_query
                        else:
                            param += grad_query
                
                #update initial parameters
                optimizer_meta.step()
                optimizer_meta.zero_grad()
                
                # Compute and save similarities
                sim_k = torch.nn.functional.cosine_similarity(task_grads_k[0], task_grads_k[1], 0).item()
                sim_init = torch.nn.functional.cosine_similarity(task_grads_init[0], task_grads_init[1], 0).item()
                sim_both = torch.nn.functional.cosine_similarity(task_grads_both[0], task_grads_both[1], 0).item()
                grad_sim.loc[len(grad_sim)] = [sim_k, sim_init, sim_both]
                grad_sim.to_csv("sim_" + emotion_1 + "_" + emotion_2 + ".csv")
                
                for t in [0, 1]:
                    print(f"k_{t}: {np.linalg.norm(task_grads_k[t].cpu())}")
                    print(f"init_{t}: {np.linalg.norm(task_grads_init[t].cpu())}")
                    print(f"both_{t}: {np.linalg.norm(task_grads_both[t].cpu())}")
                
                # save checkpoint
                if indx_batch_tasks % 10 == 0:
                    model_dir = "trained_models"
                    os.makedirs(model_dir, exist_ok=True)
                    torch.save({"args": self.args, "epoch": epoch, "step": indx_batch_tasks, "state_dict": self.state_dict}, os.path.join(model_dir, emotion_1 + "_" + emotion_2 + ".pt"))

                print("indx_batch_tasks:", indx_batch_tasks," loss:", np.mean(loss_batch_tasks), " acc:", np.mean(acc_batch_tasks), " similarity:", sim_both)

    def train_episode(self, criterion, data_per_task, optimizer_task):

        support_set, query_set = data_per_task['support'],  data_per_task['query']
        x_support_set, y_support_set = (support_set['input_ids'], support_set['attention_mask']), support_set['targets']
        x_query_set, y_query_set = (query_set['input_ids'],query_set['attention_mask']), query_set['targets']
        
        for i in range(self.args.train_step_per_episode):
            preds_support, _ = self.classifier_episode(*x_support_set)
            loss_support = criterion(preds_support, y_support_set)

            optimizer_task.zero_grad()
            loss_support.backward()
            
            optimizer_task.step()

        #generate loss for query set
        preds_query, encoded_query = self.classifier_episode(*x_query_set)
        loss_query = criterion(preds_query, y_query_set)
        accuracy_query = torch.sum(torch.argmax(preds_query,dim=-1) == y_query_set) / len(y_query_set)

        optimizer_task.zero_grad()
        grads_query = torch.autograd.grad(loss_query, filter(lambda p: p.requires_grad, self.classifier_episode.parameters()))

        return grads_query, loss_query.cpu().detach().numpy(), accuracy_query.cpu().detach().numpy(), encoded_query