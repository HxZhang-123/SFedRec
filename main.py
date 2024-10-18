import pandas as pd
import copy
import numpy as np
from torch import nn, optim

from argparse import ArgumentParser
from pathlib import Path
from load import read_dataset
from Aggregator import aggregator
from privacy import *


parser = ArgumentParser()
read_sid = False
parser.add_argument(
    '--dataset-dir', type=Path, default='dataset', help='the dataset set directory'
)
parser.add_argument(
    '--epochs', type=int, default=20, help='the maximum number of training epochs'
)
parser.add_argument(
    '--epochs_local', type=int, default=4, help='the maximum number of local training epochs'
)
parser.add_argument('--clip', type=float, default = 0.1)
parser.add_argument('--laplace_lambda', type=float, default = 0.01)
args = parser.parse_args()

df_train, df_valid, df_test, stats = read_dataset(args.dataset_dir)
#unique_test_userids = df_test['userId'].unique()
unique_train_userids = df_train['userId'].unique()
unique_userids = np.random.choice(unique_train_userids, 200, replace=False)
parameter_list=[]
model_save_dir = 'model_save_path'
num_users = getattr(stats, 'num_users', None)
num_items = stats.num_items
args.num_items = num_items
max_len = stats.max_len
lr = 0.1
embedding_dim = 32
user_embedding = nn.Embedding(num_users, embedding_dim, max_norm=1).share_memory()
item_embedding = nn.Embedding(num_items, embedding_dim, max_norm=1).share_memory()
best_result = 0
save_best = []
total_MRR = 0
total_HR = 0
total_num = 0
total_best_epoch = []
save_userID = []
save_best_result_MRR = []
save_best_result_HR = []
save_userID_num = []
save_best_epoch = []
saver = {
    'user_id':[0],
    'best_result_HR':[0],
    'best_epoch':[0],
    'best_result_MRR':[0],
    'userID_num':[0]
    }
Ks = [20]
max_K = max(Ks)


grad = []
it = []
loss_all = []


results_dict = {user_id: {} for user_id in unique_userids}
client_list = []

private_gradients = []
user_private_grad = {}
from Client import Client
for user_id in unique_userids:
    client = Client(user_id, df_train, embedding_dim, user_embedding, item_embedding, args)
    client_list.append(client)
results = {}
for epoch in range(args.epochs):
    parameter_list = []
    save_dict_uncommon = {}
    for index in range(0, len(client_list)):
        for epoch_local in range(args.epochs_local):
            res = client_list[index].train(unique_userids[index], df_train, user_embedding, item_embedding, args, model_save_dir)

        model_grad, returned_item, loss = res
        results_dict[unique_userids[index]][epoch] = {
            'loss': loss
        }
        del loss
        private_grad = gen_private_param(model_grad)
        grad.append(model_grad)
        it.append(returned_item)
        parameter_list.append(res)
        user_private_grad[index] = private_grad

    clipped_param, none_positions, grad_list = parameter_clip(parameter_list)

    result, uncommon = filter_clipped_param_by_length(clipped_param)


    for u in uncommon:
        element = grad[u]
        save_dict_uncommon[u] = element

    reversed_save_dict_uncommon = {k: save_dict_uncommon[k] for k in sorted(save_dict_uncommon.keys(), reverse=True)}
    gradient_model = aggregator(result)
    old_model_grad = copy.deepcopy(grad_list)
    new_grad_list, new_none_positions_list = del_wrong_client(uncommon, grad_list, none_positions)
    for i in range(len(new_grad_list)):
        for j in range(len(new_grad_list[i])):
            new_grad_list[i][j] = new_grad_list[i][j] - lr * gradient_model[j]
        new_grad_list[i].insert(14, user_private_grad[i])
        for pos in new_none_positions_list[i]:
            new_grad_list[i].insert(pos, None)
    old_new_grad_list = copy.deepcopy(new_grad_list)
    for key, value in save_dict_uncommon.items():
        new_grad_list.insert(key, value)
    for index in range(0, len(client_list)):
        client_list[index].update_local(new_grad_list[index])
    print(results_dict)
for index in range(0, len(client_list)):
    resulted = client_list[index].test(unique_userids[index], df_test, model_save_dir)
    results[index] = resulted
total_hr = sum(item["HR"] for item in results.values())
total_mrr = sum(item["MRR"] for item in results.values())
print(total_hr)
print(total_mrr)
