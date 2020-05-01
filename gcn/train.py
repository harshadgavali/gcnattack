import time

import torch
import torch.nn.functional as F
import torch.optim as optim

from gcn.models import GCN
from utils import accuracy

def normalize_adj(adj, args):
    """Row-normalize sparse matrix"""
    adj = adj + (adj.T - adj) * (adj.T > adj)
    I = torch.eye(adj.shape[0])
    if args.use_gpu:
        I = I.cuda()
    adj = adj + I

    rowsum = adj.sum(axis=1)
    r_inv = torch.pow(rowsum.float()+args.division_delta, -1)
    r_mat_inv = torch.diag(r_inv)
    adj = torch.matmul(r_mat_inv, adj)

    return adj

def train(epoch, model, optimizer, adj_norm, features, labels, idxs, args):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(adj_norm, features)
    loss_train = F.nll_loss(output[idxs['train']], labels[idxs['train']])
    acc_train = accuracy(output[idxs['train']], labels[idxs['train']])
    loss_train.backward()
    optimizer.step()

    if args.verbose:
        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(adj_norm, features)
        loss_val = F.nll_loss(output[idxs['val']], labels[idxs['val']])
        acc_val = accuracy(output[idxs['val']], labels[idxs['val']])
        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.4f}'.format(acc_val.item()),
            'time: {:.4f}s'.format(time.time() - t))


def test(model, adj_norm, features, labels, idxs, args):
    model.eval()
    output = model(adj_norm, features)
    loss_test = F.nll_loss(output[idxs['test']], labels[idxs['test']])
    acc_test = accuracy(output[idxs['test']], labels[idxs['test']])

    print("Test set results:",
        "loss= {:.4f}".format(loss_test.item()),
        "accuracy= {:.4f}".format(acc_test.item()))
    
    return acc_test.item()


def get_model(adj, features, labels, idxs, args):
    # Model and optimizer
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)
    if args.use_gpu:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    adj_norm = normalize_adj(adj, args)
    
    # train
    for epoch in range(args.epochs):
        train(epoch, model, optimizer, adj_norm, features, labels, idxs, args)
    
    # test
    acc_test = test(model, adj_norm, features, labels, idxs, args)
    return model, acc_test
    

def gradient(model, adj_norm, features, labels, i, j, args, grad_adj):
    model.zero_grad()
    output = model(adj_norm, features)
    loss_train = F.nll_loss(output, labels)
    loss_train.backward()
    grad = adj_norm.grad.data[i, j] if grad_adj else features.grad.data[i, j]
    return grad

def igradient_features(model, adj_norm, features, labels, i, j, args):
    features = features.clone()
    features[i, j] = 0
    features_igrad = 0
    for k in range(0, args.m+1):
        features[i, j] = k/args.m
        features_new = features.clone().requires_grad_(True)
        features_igrad += gradient(model, adj_norm, features_new, labels, i, j, args, grad_adj=False)
    return features_igrad / (args.m+1)

def igradient_adj(model, adj_norm, features, labels, i, j, adj, args):
    adj = adj.clone()
    adj[i, i] = adj[j, j] = 1
    adj[i, j], adj[j, i] = 0, 0
    adj_igrad = 0
    for k in range(0, args.m+1):
        adj[i, j], adj[j, i] = k/args.m, k/args.m
        adj_norm_new = adj_norm.clone()
        adj_norm_new[i] = adj[i] / adj[i].sum()
        adj_norm_new[j] = adj[j] / adj[j].sum()
        adj_norm_new = adj_norm_new.clone().requires_grad_(True)
        adj_igrad += gradient(model, adj_norm_new, features, labels, i, j, args, grad_adj=True)
    return adj_igrad / (args.m+1)
