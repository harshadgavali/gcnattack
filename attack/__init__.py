

from gcn.train import igradient_adj, igradient_features, normalize_adj

def attack(model, adj, features, labels, node, args):
    n = adj.shape[0]
    importance = dict()

    adj_norm = normalize_adj(adj, args)

    for j in range(n):
        if j != node:
            if j%100 == 0:
                print("a", j, flush=True)
            importance[(node, j, 'a')] = igradient_adj(model, adj_norm, features, labels, node, j, adj, args)

    for j in range(features.shape[1]):
        if j%100 == 0:
            print("f", j, flush=True)
        importance[(node, j, 'f')] = igradient_features(model, adj, features, labels, node, j, args)

    sorted_importance = list(sorted(importance.keys(), key=lambda x: importance[x], reverse=True))

    # modify
    adj_mod, features_mod = adj.clone(), features.clone()
    if args.use_gpu:
        adj_mod, features_mod = adj.clone().cpu(), features.clone().cpu()

    for i in range(int(n * args.attack_delta)):
        i, j, tp = sorted_importance[i]
        if tp == 'f':
            features_mod[i, j] = not features_mod[i, j].bool()
            features_mod[j, i] = features_mod[i, j]
        else:
            adj_mod[i, j] = not adj_mod[i, j].bool()
            adj_mod[j, i] = adj_mod[i, j]

    if args.use_gpu:
        adj_mod, features_mod = adj_mod.cuda(), features_mod.cuda()
    return adj_mod, features_mod