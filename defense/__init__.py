import torch

def defense(adj, features, args):
    adj_f = adj + (adj.T - adj) * (adj.T > adj)
    adj_f = adj_f.float()

    M11 = torch.matmul(adj_f, adj_f.T)
    M01 = torch.matmul(adj_f, 1-adj_f.T)
    M10 = torch.matmul(1-adj_f, adj_f.T)
    simil = M11 / (M10 + M01 + M11 + args.division_delta)
    
    adj = (adj.bool() & (simil >= args.defense_alpha)).int()

    return adj, features.clone()

def cosine_defense(adj, features, args):
    adj_f = adj + (adj.T - adj) * (adj.T > adj)
    adj_f = adj_f.float()
    
    # cosine similarity
    norm = adj_f.norm(p=2, dim=1, keepdim=True)
    # for avoiding division by zero adding small value
    epsilon = 1e-7
    norm=norm+epsilon
    adj_norm = adj.div(norm)
    adj_norm_transpose = adj_norm.t()
    simil=torch.mm(adj_norm,adj_norm_transpose)

    adj = (adj.bool() & (simil >= args.defense_alpha)).int()

    return adj, features.clone()

def pearson_correlation_defense(adj, features, args):
    adj_f = adj + (adj.T - adj) * (adj.T > adj)
    adj_f = adj_f.float()
    
    # pearson correlation
    adj_m = torch.mean(adj_f, 1, True)
    adj_v=adj-adj_m
    norm = adj_v.norm(p=2, dim=1, keepdim=True)
    # for avoiding division by zero adding small value
    epsilon = 1e-7
    norm=norm+epsilon
    adj_norm = adj_v.div(norm)
    adj_norm_transpose = adj_norm.t()
    simil=torch.mm(adj_norm,adj_norm_transpose)
    simil=(simil+1)/2

    adj = (adj.bool() & (simil >= args.defense_alpha)).int()

    return adj, features.clone()

def eigen_defense(adj, features, args):
    adj_f = adj + (adj.T - adj) * (adj.T > adj)
    adj_f = adj_f.float()

    _, evecs = torch.symeig(adj_f, eigenvectors=True)
    simil = torch.matmul(evecs.T, evecs)

    adj = (adj.bool() & (simil >= args.defense_alpha)).int()

    return adj, features.clone()