import torch
import torch.nn.functional as F


def compute_similar(x_centers, x_queries, method='cos', temp=10.):
    if method == 'cos':
        x_centers = F.normalize(x_centers, dim=-1)
        x_queries = F.normalize(x_queries, dim=-1)
        logits = torch.mm(x_centers, x_queries.t())

    return (logits * temp).t()
