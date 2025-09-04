import torch

def nmse(pred, target):
    return torch.sum((pred - target) ** 2) / torch.sum(target ** 2)
