
from defense import defense
from gcn.train import igradient_adj, igradient_features, normalize_adj

def attack(model, adj, features, labels, node, use_defense, args):
    n = adj.shape[0]
    importance = dict()

    if use_defense:
      adj, features = defense(adj, features, args)
    adj_norm = normalize_adj(adj, args)

    for j in range(n):
        if j != node:
            grad = igradient_adj(model, adj_norm, features, labels, node, j, adj, args)
            importance[(node, j, 'a')] = grad * ( 1 - 2 * adj[node, j].bool().int())

    for j in range(features.shape[1]):
        grad = igradient_features(model, adj_norm, features, labels, node, j, args)
        importance[(node, j, 'f')] = grad * ( 1 - 2 * features[node, j].bool().int())

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
