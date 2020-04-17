import torch.nn.functional as F
# import torch

def gradient(model, x, adj, labels, i, j, grad_adj=True):
    if grad_adj:
        adj = adj.clone().detach().requires_grad_(True)
    else:
        x = x.clone().detach().requires_grad_(True)
    model.zero_grad()
    output = model(x, adj)
    loss_train = F.nll_loss(output, labels)
    loss_train.backward()
    grad = adj.grad.data[i, j] if grad_adj else x.grad.data[i, j]
    del x, adj, loss_train, output
    return grad

def igradient_x(model, x, adj, labels, i, j, m):
    x = x.clone().detach()
    x[i, j] = 0
    x_igrad = 0
    for k in range(0, m+1):
        x[i, j] = k / m
        x_igrad += gradient(model, x, adj, labels, i, j, grad_adj=False)
    del x
    return x_igrad / (m+1)

def igradient_adj(model, x, adj, labels, i, j, m):
    adj = adj.clone().detach()
    adj[i, j] = 0
    adj_igrad = 0
    for k in range(0, m+1):
        adj[i, j] = k / m
        adj_igrad += gradient(model, x, adj, labels, i, j, grad_adj=True)
    del adj
    return adj_igrad / (m+1)