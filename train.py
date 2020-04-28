import time
import argparse

import torch
import numpy as np

from utils import load_data

from gcn.train import get_model

from defense import defense
from attack import attack

args = argparse.Namespace(dropout=0.5, epochs=100, 
                fastmode=False, hidden=16, lr=0.01, 
                seed=42, weight_decay=0.0005,
                use_gpu=False, verbose=False,
                defense_alpha=0.2, division_delta=1e-8,
                m=2, attack_delta=0.01)
args.use_gpu = args.use_gpu and torch.cuda.is_available()
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.use_gpu:
    torch.cuda.manual_seed(args.seed)

idxs = {
    'train': torch.LongTensor(range(0, 140)), 
    'val': torch.LongTensor(range(140, 500)), 
    'test': torch.LongTensor(500+np.random.choice(1000, size=100, replace=False))
}

# Load data
adj, features, labels = load_data()
if args.use_gpu:
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    for key in idxs:
        idxs[key] = idxs[key].cuda()


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
        for node in idxs['test'].cpu().numpy():
            adj, features = attack(model, adj, features, labels, node, args)
        if use_defense:
            adj, features = defense(adj, features, args)
        _, _ = get_model(adj, features, labels, idxs, args)
        # print("Total time elapsed: {:.4f}s".format(time.time() - t_total), flush=True)


train_attack_defense(adj, features, use_defense=False, use_attack=False)
print()
train_attack_defense(adj, features, use_defense=False, use_attack=True)
print()
# train_attack_defense(adj, features, use_defense=True, use_attack=True)
# print()