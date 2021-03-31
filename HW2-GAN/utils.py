import torch
from torch import nn


def compute_gradient_penalty(critic, real_samples, fake_samples):
    weight = torch.ones(real_samples.size()).to(next(critic.parameters()).device)
    dydx = torch.autograd.grad(outputs=real_samples,
                               inputs=fake_samples,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    gradient_penalty = torch.mean((dydx_l2norm-1)**2)
    
    return gradient_penalty

def permute_labels(labels):
    rand_idx = torch.randperm(labels.size(0))
    new_labels = labels[rand_idx]

    return new_labels