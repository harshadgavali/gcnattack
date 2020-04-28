import torch

def defense(adj, features, args):
    adj_f = adj.float()

    M11 = torch.matmul(adj_f, adj_f.T)
    M01 = torch.matmul(adj_f, 1-adj_f.T)
    M10 = torch.matmul(1-adj_f, adj_f.T)
    simil = M11 / (M10 + M01 + M11 + args.division_delta)

    adj = (adj.bool() & (simil >= args.defense_alpha)).int()

    return adj, features
