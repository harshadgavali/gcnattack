import time
import argparse

import torch
import numpy as np

from utils import load_data

from gcn.train import get_model, test, normalize_adj

from defense import defense
from attack import attack

args = argparse.Namespace(dropout=0.5, epochs=100, 
                fastmode=False, hidden=16, lr=0.01, 
                seed=42, weight_decay=0.0005,
                use_gpu=True, verbose=False,
                defense_alpha=0.2, division_delta=1e-8,
                m=20, attack_delta=0.01, attack_delta_degree=False)
args.use_gpu = args.use_gpu and torch.cuda.is_available()
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.use_gpu:
    torch.cuda.manual_seed(args.seed)

idxs = {
    'train': torch.LongTensor(range(0, 270)), 
    'val': torch.LongTensor(range(270, 550)), 
    'test': torch.LongTensor(550+np.random.choice(1000, size=100, replace=False))
}

# Load data
adj, features, labels = load_data()
if args.use_gpu:
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    for key in idxs:
        idxs[key] = idxs[key].cuda()

args.attack_delta = int(args.attack_delta * adj.shape[0])


def train_attack_defense(adj, features, use_defense=False, use_attack=False):
    # load params
    if use_defense:
        adj, features = defense(adj, features, args)

    t_total = time.time()
    model, _ = get_model(adj, features, labels, idxs, args)
    print("Optimization Finished!")
    # print("Total time elapsed: {:.4f}s".format(time.time() - t_total), flush=True)

    # Testing
    if use_attack:
        t_total = time.time()
        for i, node in enumerate(idxs['test'].cpu().numpy()):
            t_total = time.time()
            print(i, "of", list(idxs['test'].size()), end=" ")
            if args.attack_delta_degree:
                args.attack_delta = adj[node].sum()
            adj, features = attack(model, adj, features, labels, node, use_defense, args)
            print("time =", time.time() - t_total)
        if use_defense:
            adj, features = defense(adj, features, args)
        
        adj_norm = normalize_adj(adj, args)
        _ = test(model, adj_norm, features, labels, idxs, args)
        # print("Total time elapsed: {:.4f}s".format(time.time() - t_total), flush=True)

print()
train_attack_defense(adj, features, use_defense=False, use_attack=True)
print()
train_attack_defense(adj, features, use_defense=True, use_attack=True)
print()
args.attack_delta_degree = True
train_attack_defense(adj, features, use_defense=False, use_attack=True)
# print()
