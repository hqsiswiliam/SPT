import torch


def configure_optimizers(model, lr):
    adam = torch.optim.Adam(model.parameters(), lr=lr)
    return adam