
from utils.gradient import igradient_x, igradient_adj
def attack(node, model, adj, x, labels, m, delta):
    n = adj.shape[0]
    features = dict()

    # print(len(adj_sparse.row), x.shape)
    # for id, (i, j, _) in enumerate(zip(adj_sparse.row, adj_sparse.col, adj_sparse.data)):
    #     print(id, len(adj_sparse.row))
    #     features[(i, j, 'a')] = igradient_adj(model, x, adj, labels, i, j, m)

    for j in range(n):
        features[(node, j, 'a')] = igradient_adj(model, x, adj, labels, node, j, m)

    for j in range(x.shape[1]):
        features[(node, j, 'x')] = igradient_x(model, x, adj, labels, node, j, m)

    sorted_features = list(sorted(features.keys(), key=lambda x: features[x], reverse=True))

    x_mod, adj_mod = x.clone().detach(), adj.clone().detach()
    for i in range(int(n * delta)):
        i, j, tp = sorted_features[i]
        if tp == 'x':
            x_mod[i, j] = x_mod[i, j] == 0
        else:
            adj_mod[i, j] = adj_mod[i, j] == 0

    return adj_mod, x_mod