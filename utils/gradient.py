def gradient(model, x, adj):
    x.requires_grad = True
    adj.requires_grad = True
    _ = model(x, adj)
    model.zero_grad()
    return x.grad.data, adj.grad.data

def igradient_x(model, x, adj, i, j, m):
    x = x.copy()
    x[i, j] = 1
    x_igrad, _ = gradient(model, x, adj)
    for k in range(1, m):
        x[i, j] = k / m
        tmp, _ = gradient(model, x, adj)
        x_igrad += tmp
    return x_igrad

def igradient_adj(model, x, adj, i, j, m):
    adj = adj.copy()
    adj[i, j] = 1
    adj_igrad, _ = gradient(model, x, adj)
    for k in range(1, m):
        adj[i, j] = k / m
        _, tmp = gradient(model, x, adj)
        adj_igrad += tmp
    return adj_igrad