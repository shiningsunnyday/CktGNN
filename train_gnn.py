import argparse
import os
import pickle
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
import networkx as nx
import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool
from tqdm import tqdm
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='VAE experiments on Ckt-Bench-101')
    parser.add_argument('--data-fold-name', default='CktBench101', help='dataset fold name')
    parser.add_argument('--data-name', default='ckt_bench_101', help='circuit benchmark dataset name')
    parser.add_argument('--target', choices=['fom', 'pos'])
    parser.add_argument('--ablate-feats', action='store_true')
    args = parser.parse_args()
    return args

# Convert NetworkX graph to PyTorch Geometric Data object
def convert_data(graph, label, ablate=False):
    # Assuming node features are stored in 'feat' attribute and label are provided separately        
    assert sorted(list(graph)) == list(range(len(graph)))
    feats = []
    for n in range(len(graph)):
        if ablate:
            feat = torch.zeros((10,), dtype=torch.float)  
            feat[graph.nodes[n]['type']] = 1
        else: 
            feat = torch.zeros((11,), dtype=torch.float)
            feat[graph.nodes[n]['type']] = 1
            feat[-1] = graph.nodes[n]['feat']
        feats.append(feat)
    x = torch.stack(feats, dim=0)
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
    y = torch.tensor([label], dtype=torch.float)  # Assuming labels is a list or array    
    return Data(x=x, edge_index=edge_index, y=y)


# Convert NetworkX graph to PyTorch Geometric Data object
def convert_data_with_node_label(graph):
    # Assuming node features are stored in 'feat' attribute and label are provided separately        
    assert sorted(list(graph)) == list(range(len(graph)))
    feats = []
    ys = []
    for n in range(len(graph)):
        feat = torch.zeros((10,), dtype=torch.float)
        feat[graph.nodes[n]['type']] = 1
        y = graph.nodes[n]['feat']
        feats.append(feat)
        ys.append([y])
    x = torch.stack(feats, dim=0)
    ys = torch.tensor(ys)
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()    
    return Data(x=x, edge_index=edge_index, y=ys)


# Define a simple GNN model
class GNN(torch.nn.Module):
    def __init__(self, in_d, h_d, out_d, num_layers=2):
        super(GNN, self).__init__()
        self.conv1 = GINConv(nn.Sequential(nn.Linear(in_d, h_d), nn.ReLU(), nn.Linear(h_d, h_d)))
        for l in range(2, num_layers):
            setattr(self, f"conv{l}", GINConv(nn.Sequential(nn.Linear(h_d, h_d), nn.ReLU(), nn.Linear(h_d, h_d))))
        setattr(self, f"conv{num_layers}", GINConv(nn.Sequential(nn.Linear(h_d, h_d), nn.ReLU(), nn.Linear(h_d, out_d))))
        self.num_layers = num_layers


    def forward(self, data, reduce=True):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for l in range(1, self.num_layers):
            x = getattr(self, f"conv{l}")(x, edge_index)
            x = F.relu(x)
        x = getattr(self, f"conv{self.num_layers}")(x, edge_index)
        if reduce:
            x = global_mean_pool(x, batch)
            x = x.flatten()
        return x


def main(args):
    data_name = args.data_name
    data_type = 'igraph'
    args.file_dir = os.path.dirname(os.path.realpath('__file__'))
    args.data_dir = os.path.join(args.file_dir, 'OCB/{}'.format(args.data_fold_name))
    pkl_name = os.path.join(args.data_dir, data_name + '.pkl')
    with open(pkl_name, 'rb') as f:
        all_datasets =  pickle.load(f)
    train_dataset = all_datasets[0]
    test_dataset = all_datasets[1]
    train_data = [train_dataset[i][1] for i in range(len(train_dataset))]
    test_data = [test_dataset[i][1] for i in range(len(test_dataset))]

    train_data = [g.to_networkx() for g in train_data]
    test_data = [g.to_networkx() for g in test_data]


    if args.target == 'pos' or args.ablate_feats:
        in_dim = 10
    else:
        in_dim = 11
    model = GNN(in_dim, 256, 1, num_layers=10)
    optimizer = torch.optim.Adam(model.parameters())

    # Training loop
    num_epochs = 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)    
    if args.target == 'pos':
        reduce = False
        all_feats = sum([[g.nodes[n]['feat'] for n in g] for g in train_data], [])
        train_mean = np.mean(all_feats)
        train_std = np.std(all_feats)
        for g in train_data+test_data:
            for n in g:
                g.nodes[n]['feat'] = (g.nodes[n]['feat']-train_mean)/train_std
        train_data_list = [convert_data_with_node_label(g) for i, g in enumerate(train_data)]
        test_data_list = [convert_data_with_node_label(g) for i, g in enumerate(test_data)]
    else:
        perf_name = os.path.join(args.data_dir, 'perform101.csv')
        perform_df = pd.read_csv(perf_name, index_col=0)
        # perform_df = perform_df.iloc[:len(train_data)]        
        assert len(train_data)+len(test_data) == len(perform_df)
        reduce = True
        train_mean = perform_df.iloc[:len(train_data)][args.target].mean()
        train_std = perform_df.iloc[:len(train_data)][args.target].std()        
        train_data_list = [convert_data(g, (perform_df.iloc[i][args.target]-train_mean)/train_std, args.ablate_feats) for i, g in enumerate(train_data)]
        test_data_list = [convert_data(g, (perform_df.iloc[i+len(train_data)][args.target]-train_mean)/train_std, args.ablate_feats) for i, g in enumerate(test_data)]

    # Create DataLoader
    batch_size = 32
    train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data_list, batch_size=1, shuffle=False)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        total_train_len = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data, reduce=reduce)
            loss_all = (out-data.y)**2
            loss = loss_all.mean()
            loss.backward()
            optimizer.step()
            total_train_loss += loss_all.sum().item()
            total_train_len += out.shape[0]
        train_loss = total_train_loss/total_train_len
        print(f'Epoch {epoch+1}/{num_epochs}, Train MSE Loss: {train_loss}')
        model.eval()
        total_test_loss = 0
        total_test_len = 0
        for data in test_loader:
            data = data.to(device)
            out = model(data, reduce=reduce)
            # out = out*train_std+train_mean
            # y = data.y*train_std+train_mean
            loss = (out-data.y).abs()
            loss = loss.mean()
            total_test_loss += loss.item()*out.shape[0]
            total_test_len += out.shape[0]
        total_test_loss = total_test_loss/total_test_len
        # test_loss = np.sqrt(total_test_loss)
        test_loss = total_test_loss
        print(f'Epoch {epoch+1}/{num_epochs}, Test MAE Loss: {test_loss}')
    print('Training complete!')

if __name__ == "__main__":
    args = get_args()
    breakpoint()
    main(args)
