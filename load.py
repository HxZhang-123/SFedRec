import itertools
from collections import Counter

import pandas as pd
import numpy as np

import torch
import dgl


class AugmentedDataset:
    def __init__(self, df, read_sid=False, sort_by_length=True):
        if read_sid:
            df = df[['sessionId', 'userId', 'items']]
        else:
            df = df[['userId', 'items']]
        self.sessions = df.values
        session_lens = df['items'].apply(len)
        index = create_index(session_lens)
        if sort_by_length:
            ind = np.argsort(index[:, 1])[::-1]
            index = index[ind]
        self.index = index

    def __getitem__(self, idx):
        sidx, lidx = self.index[idx]
        sess = self.sessions[sidx]
        seq = sess[-1][:lidx]
        label = sess[-1][lidx]
        item = (*sess[:-1], seq, label)
        return item

    def __len__(self):
        return len(self.index)


def create_index(session_lens):
    num_sessions = len(session_lens)
    session_idx = np.repeat(np.arange(num_sessions), session_lens - 1)
    label_idx = map(lambda l: range(1, l), session_lens)
    label_idx = itertools.chain.from_iterable(label_idx)
    label_idx = np.fromiter(label_idx, dtype=np.int64)
    idx = np.column_stack((session_idx, label_idx))
    return idx


def read_sessions(filepath):
    df = pd.read_csv(filepath, sep='\t')
    df['items'] = df['items'].apply(lambda x: [int(i) for i in x.split(',')])
    return df


def read_dataset(dataset_dir):
    stats = pd.read_csv(dataset_dir / 'stats.txt', sep='\t').iloc[0]
    df_train = read_sessions(dataset_dir / 'train.txt')
    df_valid = read_sessions(dataset_dir / 'valid.txt')
    df_test = read_sessions(dataset_dir / 'test.txt')
    return df_train, df_valid, df_test, stats

def read_social_network(csv_file, user_id):
    df = pd.read_csv(csv_file, sep='\t')
    filtered_df = df[(df["follower"] == user_id) | (df["followee"] == user_id)]

    # Find the unique node ID
    unique_nodes = pd.concat([filtered_df["follower"], filtered_df["followee"]]).unique()

    # Create a dictionary to map the original node ID to consecutive node IDs
    node_mapping = {node_id: i for i, node_id in enumerate(unique_nodes)}

    # Build a graph using the mapped node IDs
    src = filtered_df['follower'].map(node_mapping).values
    dst = filtered_df['followee'].map(node_mapping).values
    g = dgl.graph((src, dst))

    # Use the original node ID as the attribute of the node
    g.ndata['original_id'] = torch.tensor(list(unique_nodes))
    neighbors = sorted([node_id for node_id in unique_nodes if node_id != user_id])
    print(g)
    edges = g.edges(order='eid')
    for src, dst in zip(edges[0], edges[1]):
        src_node_id = g.ndata['original_id'][src].item()
        dst_node_id = g.ndata['original_id'][dst].item()
        print(f"Edge: ({src_node_id}, {dst_node_id})")

    return g, neighbors


def build_Hetgraph(df_train, social_network, user_id, do_count_clipping=True):
    print('building heterogeneous knowledge graph for each client:')
    followed_edges = social_network.edges()
    clicks = Counter()
    transits = Counter()
    df_train = df_train[(df_train["userId"] == user_id)]

    unique_values = set()
    for item_list in df_train['items']:
        unique_values.update(item_list)
    item_mapping = {item_id: new_id for new_id, item_id in enumerate(unique_values)}
    # Create a new column to store the mapped values
    df_train['mapped_items'] = df_train['items'].apply(
        lambda item_list: [item_mapping[value] for value in item_list])

    for _, row in df_train.iterrows():
        uid = row['userId']
        seq = row['mapped_items']
        for iid in seq:
            # Use the counter object to count the clicks on items clicked by users and the number of transitions between items
            clicks[(uid, iid)] += 1
        transits.update(zip(seq, seq[1:]))
    clicks_u, clicks_i = zip(*clicks.keys())
    prev_i, next_i = zip(*transits.keys())
    kg = dgl.heterograph({
        ('user', 'followedby', 'user'): followed_edges,
        ('user', 'clicks', 'item'): (clicks_u, clicks_i),
        ('item', 'clickedby', 'user'): (clicks_i, clicks_u),
        ('item', 'transitsto', 'item'): (prev_i, next_i),
    })
    click_cnts = np.array(list(clicks.values()))
    transit_cnts = np.array(list(transits.values()))
    if do_count_clipping:
        click_cnts = clip_counts(click_cnts)
        transit_cnts = clip_counts(transit_cnts)
    click_cnts = torch.LongTensor(click_cnts) - 1
    transit_cnts = torch.LongTensor(transit_cnts) - 1
    kg.edges['clicks'].data['cnt'] = click_cnts
    kg.edges['clickedby'].data['cnt'] = click_cnts
    kg.edges['transitsto'].data['cnt'] = transit_cnts
    unique_values = sorted(unique_values)
    user_nodes = kg.nodes('user')
    item_nodes = kg.nodes('item')
    return kg, unique_values, item_mapping, user_nodes, item_nodes


def find_max_count(counts):
    max_cnt = np.max(counts)
    density = np.histogram(counts, bins=np.arange(1, max_cnt + 2), range=(1, max_cnt + 1), density=True)[0]
    cdf = np.cumsum(density)
    for i in range(max_cnt):
        if cdf[i] > 0.95:
            return i + 1
    return max_cnt


def clip_counts(counts):
    max_cnt = find_max_count(counts)
    counts = np.minimum(counts, max_cnt)
    return counts

def label_last(g, last_nid):
    is_last = torch.zeros(g.number_of_nodes(), dtype=torch.int32)
    is_last[last_nid] = 1
    g.ndata['last'] = is_last
    return g

def seq_to_weighted_graph(seq):
    iid, seq_nid, cnt = np.unique(seq, return_inverse=True, return_counts=True)
    num_nodes = len(iid)

    if len(seq_nid) > 1:
        counter = Counter(zip(seq_nid, seq_nid[1:]))
        src, dst = zip(*counter.keys())
        weight = torch.FloatTensor(list(counter.values()))
    else:
        src = torch.LongTensor([])
        dst = torch.LongTensor([])
        weight = torch.FloatTensor([])

    g = dgl.graph((src, dst), num_nodes=num_nodes)
    g.ndata['iid'] = torch.LongTensor(iid)
    g.ndata['cnt'] = torch.FloatTensor(cnt)
    g.edata['w'] = weight.view(g.num_edges(), 1)
    label_last(g, seq_nid[-1])
    return g

def sample_blocks(g, uniq_uids, uniq_iids, fanouts):
    uniq_iids = uniq_iids.astype(int)
    seeds = {'user': torch.LongTensor(uniq_uids), 'item': torch.LongTensor(uniq_iids)}
    blocks = []

    for fanout in fanouts:
        if fanout <= 0:
            frontier = dgl.in_subgraph(g, seeds)
        else:
            frontier = dgl.sampling.sample_neighbors(
                g, seeds, fanout, copy_ndata=False, copy_edata=True
            )
        block = dgl.to_block(frontier, seeds)
        seeds = {ntype: block.srcnodes[ntype].data[dgl.NID] for ntype in block.srctypes}
        blocks.insert(0, block)
    return blocks, seeds