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

    def get_allPrototypes(self, x_support_set, y_support_set):
        """
         calculate the prototype for all class
        :param x_support_set: (batch_size,sent_len)
        :param y_support_set: (batch,)
        :return:
        """
        all_prototypes_mix = self.classifier_init.encoder(*x_support_set).pooler_output #(batch_size,dim_encoder_hidden)

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
        
        grad_sim = pd.DataFrame()
        
        optimizer_meta = optim.SGD(self.classifier_init.encoder.parameters(), lr=self.args.lr_beta)
        
        for epoch in range(self.args.num_epoch) :
            # sample batch of tasks
            for indx_batch_tasks, data_batch_tasks in enumerate(data_iter_train):
                
                loss_batch_tasks = []
                acc_batch_tasks = []
                
                emotions = list(data_batch_tasks.keys())
                task_grads_k = []
                task_grads_init = []
                task_grads_both = []
                
                #for each task/episode
                for t, data_per_task in enumerate(data_batch_tasks.values()):
                    
                    # STEP 1 sample data
                    support_set, query_set = data_per_task['support'],  data_per_task['query']
                    x_support_set, y_support_set = (support_set['input_ids'], support_set['attention_mask']), support_set['targets']
                    
                    # STEP 2 duplicate model
                    self.classifier_episode = deepcopy(self.classifier_init)
                    for ind, param in enumerate(self.classifier_episode.encoder.parameters()):
                        if param.grad is not None:
                            print("new model has gradient!")
                            param.grad = None
                    
                    # STEP 3 calculate prototypes
                    prototypes = self.get_allPrototypes(x_support_set, y_support_set) #(num_class, dim_encoder_hidden)
                    
                    # STEP 4 initialize final layer using prototypes
                    #classifier_episode.fc_layer.weight.data = c_prototypes.detach() * 2.0 #(num_class, dim_encoder_hidden)
                    #classifier_episode.fc_layer.bias.data = -1.0 * c_prototypes.detach().norm(dim=1)**2 #(num_class,)
                    p_weight = 2 * prototypes #(num_class, dim_encoder_hidden)
                    p_bias = -prototypes.norm(dim=1)**2 #(num_class,)
                    fc_weight = p_weight.detach()
                    fc_bias = p_bias.detach()
                    
                    # STEP 5 train episode
                    x_support_set, y_support_set = (support_set['input_ids'], support_set['attention_mask']), support_set['targets']
                    all_params_inner = list(self.classifier_episode.parameters()) + [fc_weight, fc_bias]
                    optimizer_task = optim.SGD(all_params_inner, lr=self.args.lr_alpha)
                    for i in range(self.args.train_step_per_episode):
                        encoded_support = self.classifier_episode.encoder(*x_support_set).pooler_output
                        #print(encoded_support.mean().item(), encoded_support.std().item())
                        preds_support = encoded_support @ fc_weight.T + fc_bias
                        #print(fc_weight.mean().item(), fc_weight.std().item())
                        #print(fc_bias.mean().item(), fc_bias.std().item())
                        #print(preds_support)
                        loss_support = criterion(preds_support, y_support_set)
                        optimizer_task.zero_grad()
                        loss_support.backward()
                        optimizer_task.step()
                        #print(loss_support.detach().item())
                        #print("")
                    #sys.exit()
                    
                    # STEP 6 reintroduce prototypes to computation graph
                    fc_weight = p_weight + (fc_weight - p_weight).detach()
                    fc_bias = p_bias + (fc_bias - p_bias).detach()
                    
                    # STEP 7 get gradients from query set
                    x_query_set, y_query_set = (query_set['input_ids'],query_set['attention_mask']), query_set['targets']
                    encoded_query = self.classifier_episode.encoder(*x_query_set).pooler_output
                    preds_query = encoded_query @ fc_weight.T + fc_bias # Have to do it this way to keep gradients
                    loss_query = criterion(preds_query, y_query_set)
                    accuracy_query = torch.sum(torch.argmax(preds_query,dim=-1) == y_query_set) / len(y_query_set)
                    
                    n_params = len(list(self.classifier_init.encoder.parameters()))
                    params_k = self.classifier_episode.encoder.parameters()
                    params_init = self.classifier_init.encoder.parameters()
                    all_params = list(params_k) + list(params_init)
                    grads_query_all = torch.autograd.grad(loss_query, all_params)
                    grads_query_k = grads_query_all[:n_params]
                    grads_query_init = grads_query_all[n_params:]
                    
                    assert len(grads_query_k) == len(list(self.classifier_init.encoder.named_parameters())) # sanity check
                    assert len(grads_query_init) == len(list(self.classifier_init.encoder.named_parameters())) # sanity check
                    
                    # STEP 8 store in model
                    for ind, param in enumerate(self.classifier_init.encoder.parameters()):
                        grad_query = grads_query_k[ind] + grads_query_init[ind]
                        #grad_query = grads_query_k[ind]
                        if param.grad is None:
                            param.grad = grad_query
                        else:
                            param.grad += grad_query
                    
                    # save loss_query and acc
                    loss_batch_tasks.append(loss_query.detach().cpu().numpy())
                    acc_batch_tasks.append(accuracy_query.detach().cpu().numpy())
                    
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
                
                # Outer loop update
                #optimizer_meta.step()
                #optimizer_meta.zero_grad()
                for param in self.classifier_init.encoder.parameters():
                    if param.requires_grad:
                        param.data -= param.grad * self.args.lr_beta
                    param.grad = None
                
                # Compute and save similarities
                sims = {}
                sims["loss"] = np.mean(loss_batch_tasks)
                sims["sim_k"] = torch.nn.functional.cosine_similarity(task_grads_k[0], task_grads_k[1], 0).item()
                sims["sim_init"] = torch.nn.functional.cosine_similarity(task_grads_init[0], task_grads_init[1], 0).item()
                sim_both = torch.nn.functional.cosine_similarity(task_grads_both[0], task_grads_both[1], 0).item()
                sims["sim_both"] = sim_both
                grad_sim = grad_sim.append(sims, ignore_index=True)
                emotion_string = "_".join(emotions)
                grad_sim.to_csv("sim_" + emotion_string + ".csv")
                
                '''for t in [0, 1]:
                    print(f"k_{t}: {np.linalg.norm(task_grads_k[t].cpu())}")
                    print(f"init_{t}: {np.linalg.norm(task_grads_init[t].cpu())}")
                    print(f"both_{t}: {np.linalg.norm(task_grads_both[t].cpu())}")'''
                
                # save checkpoint
                if indx_batch_tasks % 10 == 0:
                    model_dir = "trained_models"
                    os.makedirs(model_dir, exist_ok=True)
                    torch.save({"args": self.args, "epoch": epoch, "step": indx_batch_tasks, "state_dict": self.state_dict}, os.path.join(model_dir, emotion_string + ".pt"))

                print("indx_batch_tasks:", indx_batch_tasks, " loss:", np.mean(loss_batch_tasks), " acc:", np.mean(acc_batch_tasks), " similarity:", sim_both)