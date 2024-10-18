import copy

import pandas as pd
import numpy as np

import torch
import dgl
import gc

from torch import nn, optim

from load import read_social_network
from load import build_Hetgraph
from load import AugmentedDataset
from load import sample_blocks
from load import seq_to_weighted_graph
from model import SFedRec


class Client():
    def __init__(self, user_id, df_train, embedding_dim, user_embedding, item_embedding, args):
        self.user_id = user_id
        self.df_train = df_train
        self.embedding_dim = embedding_dim
        self.user_embedding = user_embedding
        self.item_embedding = item_embedding
        self.args = args

        self.social_network, self.neighbors = read_social_network(args.dataset_dir / 'edges.txt', user_id)
        self.knowledge_graph, self.unique_values, self.item_mapping, self.user_nodes, self.item_nodes, self.set = self.build_graph(
            user_id, df_train, args)
        self.model = self.build_model(embedding_dim, item_embedding,
                                      self.knowledge_graph, args)


    def build_graph(self, user_id, df_train, args):
        social_network, neighbors = read_social_network(args.dataset_dir / 'edges.txt', user_id)
        knowledge_graph, unique_values, item_mapping, user_nodes, item_nodes = build_Hetgraph(df_train, social_network, user_id, do_count_clipping=True)
        user_df_train = df_train[(df_train["userId"] == user_id)]
        user_df_train_set = copy.deepcopy(user_df_train)
        user_df_train['items'] = user_df_train['items'].apply(lambda x: [item_mapping[i] for i in x])
        set = AugmentedDataset(user_df_train, read_sid=False)

        return knowledge_graph, unique_values, item_mapping, user_nodes, item_nodes, set

    def reverse_map(self, item_mapping):
        reverse_mapping = {v: k for k, v in item_mapping.items()}
        return reverse_mapping

    def build_model(self, embedding_dim, item_embedding, knowledge_graph, args):
        model = SFedRec(item_embedding, embedding_dim, knowledge_graph, args.num_items, num_layers=2)
        return model    #构建本地模型

    def build_extra_input(self, user_id):
        num_neighbors = len(self.neighbors)
        reverse_mapping = self.reverse_map(self.item_mapping)
        Labels = []
        Seqs = []
        for idx in range(len(self.set)):
            samples = self.set[idx]
            rating = samples[-1]   # 获取样本中的最后一个元素，即评分
            seqs = samples[-2]
            Seqs.append(seqs)
            Labels.append(rating)
        combined_data = list(zip(Seqs, Labels))
        result = [(user_id, data[0], data[1]) for data in combined_data]
        uids, seqens, labels = zip(*result)
        labels = torch.LongTensor(labels)
        new_labels_id = torch.tensor([reverse_mapping.get(item.item(), item.item()) for item in labels])
        fanouts = num_neighbors
        fanout_list = [fanouts]
        used_nodes = np.array(self.user_nodes)
        itemd_nodes = np.array(self.item_nodes)
        extra_inputs = sample_blocks(self.knowledge_graph, used_nodes, itemd_nodes, fanout_list)
        del result, uids, seqens, labels, fanouts, fanout_list, combined_data, used_nodes, itemd_nodes
        return extra_inputs, reverse_mapping, new_labels_id, Seqs, self.unique_values

    def build_feat(self, user_id, df_train, user_embedding, item_embedding, args):
        extra_inputs, reverse_mapping, new_labels_id, Seqs, unique_values =self.build_extra_input(user_id)
        del new_labels_id, Seqs, unique_values
        graphs_inputs, used_nodes = extra_inputs
        del extra_inputs
        iidds = used_nodes['item']  #item nodes in extra_inputs

        unique = []
        for item in iidds:
            if item not in unique:
                unique.append(item)  # 独一的item
        self.mapped_values = [reverse_mapping[tensor.item()] for tensor in unique]
        del reverse_mapping
        uidds = used_nodes['user']  #user nodes in extra_inputs

        uniqueu = []
        for user in uidds:
            if user not in uniqueu:
                uniqueu.append(user)

        feats = {
            'user': user_embedding(torch.LongTensor(uniqueu)),
            'item': item_embedding(torch.LongTensor(self.mapped_values)),
        }

        return feats

    def train(self, user_id, df_train, user_embedding, item_embedding, args, model_save_dir):

        extra_inputs, reverse_mapping, new_labels_id, Seqs, unique_values = self.build_extra_input(user_id)
        feats = self.build_feat(user_id, df_train, user_embedding, item_embedding, args)

        params = self.model.parameters()
        lr = 1e-2
        optimizer = optim.AdamW(params, lr, weight_decay=0)
        self.model.train()
        logit = []
        # 假设 graphs 是一个包含 X 行图的列表
        for seq in Seqs:
            # 获取当前序列的新项目
            graph = seq_to_weighted_graph(seq)
            new_seq_id = [reverse_mapping[id_] for id_ in seq]

            logits = self.model(graph, new_seq_id, extra_inputs, feats)
            logit.append(logits)
            # 遍历当前图中的节点数据，并将其转换为整数类型的张量

        logits_tensor = torch.stack(logit)
        if logits_tensor.shape[0] == 1 and logits_tensor.shape[1] == 1:
            flattened_tensor = logits_tensor.squeeze(0)
        else:
            flattened_tensor = logits_tensor.squeeze()
        criterion = nn.CrossEntropyLoss()
        loss = criterion(flattened_tensor, new_labels_id)
        self.model.zero_grad()
        loss.backward(retain_graph=True)
        del flattened_tensor, logits_tensor, logit, graph, logits
        gc.collect()
        optimizer.step()
        # 保存模型
        model_save_path = f"{model_save_dir}/model_client_{user_id}.pth"
        torch.save(self.model.state_dict(), model_save_path)
        # print(f"Model saved at {model_save_path}")
        del model_save_path
        gc.collect()
        model_grad = []
        num = 0
        #w = self.model.state_dict()
        for param in list(self.model.parameters()):
            if param.grad == None:
                continue
            gradient = self.LDP(param.grad)
            model_grad.append(gradient)
        returned_item = unique_values
        del extra_inputs, reverse_mapping, new_labels_id, Seqs, unique_values
        res = (model_grad, returned_item, loss.detach())
        return res

    def LDP(self, tensor):
        tensor_mean = torch.abs(torch.mean(tensor))
        tensor = torch.clamp(tensor, min=-self.args.clip, max=self.args.clip)
        noise = np.random.laplace(0, tensor_mean * self.args.laplace_lambda)
        tensor += noise
        return tensor

    def send_model(self):
        return self.model.state_dict()

    def receive_model(self, new_state_dict):
        self.model.load_state_dict(new_state_dict)

    def update_local(self, new_param):
        paradata = list(self.model.parameters())
        with torch.no_grad():  # 确保不会跟踪梯度操作
            for i in range(len(new_param)):
                if i != 0:
                    continue
                paradata[i].grad = new_param[i]

    def gen_private_param(self, model_grad):
        private_grad = model_grad[24]

    def test(self, user_id, df_test, model_save_dir):
        best_result = 0
        save_best = []
        total_MRR = 0
        total_HR = 0
        total_num = 0
        total_best_epoch = []
        best_model_save_dir = 'E:\models_best'
        user_df_test = df_test[(df_test["userId"] == user_id)]
        read_sid = False
        set_test = AugmentedDataset(user_df_test, read_sid)
        Labels_test = []
        Seqs_test = []
        test_ids = []
        for idx in range(len(set_test)):
            samples = set_test[idx]
            test_id = samples[0]
            rating_test = samples[-1]  # 获取样本中的最后一个元素，即评分
            seqs_test = samples[-2]
            test_ids.append(test_id)
            Seqs_test.append(seqs_test)
            Labels_test.append(rating_test)
        combined_data_test = list(zip(test_ids, Seqs_test, Labels_test))

        result_test = [(data[0], data[1], data[2]) for data in combined_data_test]

        uids_test, seqens_test, labels_test = zip(*result_test)

        uids_test = torch.LongTensor(uids_test)
        labels_test = torch.LongTensor(labels_test)
        iids_test = np.concatenate(seqens_test)
        new_iids_test = pd.factorize(iids_test, sort=True)

        inputs_test = [uids_test]
        cur_idx_test = 0
        new_seqs_test = []
        graphs_test = []

        test_users_ids = list(range(0, user_id+1))
        feat = {
            'user': self.user_embedding(torch.LongTensor(test_users_ids)),
            'item': self.item_embedding(torch.LongTensor(self.mapped_values)),
        }

        for seq_test in Seqs_test:
            # 获取当前序列的新项目
            graph_test = seq_to_weighted_graph(seq_test)
            graphs_test.append(graph_test)
        test_data = list(zip(test_ids, Seqs_test, graphs_test))
        logit_test = []

        for test_user_id, Seqstest, graphtest in test_data:
            # 加载对应客户端的模型
            model_save_path = f"{model_save_dir}/model_client_{test_user_id}.pth"
            state_dict = torch.load(model_save_path)
            self.model.load_state_dict(state_dict)
            self.model.eval_HKG(feat)
            self.model.eval()
            with torch.no_grad():
                logits_test = self.model(graphtest, Seqstest)
            logit_test.append(logits_test)
        logits_tensor_test = torch.stack(logit_test)
        flattened_tensor_test = logits_tensor_test.squeeze()
        if logits_tensor_test.shape[0] == 1 and logits_tensor_test.shape[1] == 1:
            flattened_tensor_test = logits_tensor_test.squeeze(0)
        else:
            flattened_tensor_test = logits_tensor_test.squeeze()
        from collections import defaultdict

        results = defaultdict(float)
        topk = torch.topk(flattened_tensor_test, k=20, sorted=True)[1]
        num = flattened_tensor_test.size(0)
        labels_test = labels_test.unsqueeze(-1)
        for K in [20]:
            hit_ranks = torch.where(topk[:, :K] == labels_test)[1] + 1
            hit_ranks = hit_ranks.float().cpu()
            if hit_ranks.numel() > best_result:
                best_result = hit_ranks.numel()
                best_HR = hit_ranks.numel()
                best_MRR = hit_ranks.reciprocal().sum().item()
                total_HR += best_HR
                #total_best_epoch.append(best_epoch)
                total_MRR += best_MRR
                total_num += num
                best_model = self.model
                best_model_save_path = f"{best_model_save_dir}/model_client_{user_id}.pth"
                torch.save(best_model.state_dict(), best_model_save_path)
            results[f'HR@{K}'] += hit_ranks.numel()
            results[f'MRR@{K}'] += hit_ranks.reciprocal().sum().item()
            results[f'NDCG@{K}'] += torch.log2(1 + hit_ranks).reciprocal().sum().item()
        return results
