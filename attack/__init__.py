
import torch
from defense import defense
from gcn.train import igradient_adj, igradient_features, normalize_adj

def total_jaccard_simil(adj_symm, simil, node, j, args):
    adj_f = adj_symm.clone().float()
    new_val = (1+adj_symm[node, j])%2
    adj_f[node, j] = new_val
    adj_f[j, node] = new_val

    adj_nodej = adj_f[:, [node, j]]
    M11 = torch.matmul(adj_f, adj_nodej)
    M01 = torch.matmul(adj_f, 1-adj_nodej)
    M10 = torch.matmul(1-adj_f, adj_nodej)
    simil_nodej = M11 / (M10 + M01 + M11 + args.division_delta)

    simil_nodej[node, 0] = 0
    simil_nodej[j, 1]    = 0
    simil_nodej[node, 1] = 0

    mask = torch.tril(adj_symm, -1).bool()
    mask[:, [node, j]] = False
    mask[[node, j], :] = False
    
    return simil[mask].sum() + simil_nodej.sum()

def attack(model, adj, features, labels, node, args):
    n = adj.shape[0]
    importance = dict()
    
    adj_norm = normalize_adj(adj, args)

    if bool(args.attack_simil_alpha):
        adj_symm = adj + (adj.T - adj) * (adj.T > adj)

        adj_f = adj_symm.float()
        M11 = torch.matmul(adj_f, adj_f.T)
        M01 = torch.matmul(adj_f, 1-adj_f.T)
        M10 = torch.matmul(1-adj_f, adj_f.T)
        simil = M11 / (M10 + M01 + M11 + args.division_delta)
        mask = torch.tril(adj_symm, -1).bool()

        simil_normalize = mask.sum()
        simil_original  = simil[mask].sum() / simil_normalize
        simil_normalize += 1

    for j in range(n):
        if j != node:
            grad = igradient_adj(model, adj_norm, features, labels, node, j, adj, args)
            importance[(node, j, 'a')] = grad * ( 1 - 2 * adj[node, j].bool().int())

            if bool(args.attack_simil_alpha):
                importance[(node, j, 'a')] += total_jaccard_simil(adj_symm, simil, node, j, args) / simil_normalize

    for j in range(features.shape[1]):
        grad = igradient_features(model, adj_norm, features, labels, node, j, args)
        importance[(node, j, 'f')] = grad * ( 1 - 2 * features[node, j].bool().int())

        if bool(args.attack_simil_alpha):
            importance[(node, j, 'f')] += simil_original

    sorted_importance = list(sorted(importance.keys(), key=lambda x: importance[x], reverse=True))

    # modify
    adj_mod, features_mod = adj.clone(), features.clone()
    if args.use_gpu:
        adj_mod, features_mod = adj.clone().cpu(), features.clone().cpu()

    for i in range(int(args.attack_delta)):
        i, j, tp = sorted_importance[i]
        if tp == 'f':
            features_mod[i, j] = not features_mod[i, j].bool()
        else:
            adj_mod[i, j] = not adj_mod[i, j].bool()
            adj_mod[j, i] = adj_mod[i, j]

    if args.use_gpu:
        adj_mod, features_mod = adj_mod.cuda(), features_mod.cuda()
    return adj_mod, features_mod
