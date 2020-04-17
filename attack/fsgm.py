
from utils.gradient import gradient

def fsgm_attack(model, x, adj, epsilon):

    x_grad, adj_grad = gradient(model, x, adj)
    
    sign_data_grad = data_grad.sign()
    x_grad_sign = x_grad.sign()
    adj_grad_sign = adj_grad.sign()
    
    return x_mod, adj_mod