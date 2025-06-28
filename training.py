import torch


class CustomLoss(torch.nn.Module):
    def __init__(self, loss_name, penalty: float):
        super().__init__()
        self.penalty = penalty
        self.relu = torch.nn.ReLU()
        self.loss_fn = None
        if loss_name == "mse":
            self.loss_fn = torch.nn.MSELoss(reduction="sum")
        elif loss_name == "l1":
            self.loss_fn = torch.nn.L1Loss(reduction="sum")
        elif loss_name == "huber":
            self.loss_fn = torch.nn.HuberLoss(reduction="sum", delta=1.0)
        else:
            raise ValueError(f"Unknown loss {loss_name}")

    def forward(self, input, target):
        loss = self.loss_fn(input, target)
        if self.penalty > 0:
            loss += self.penalty * torch.sum(torch.square(self.relu(-input)))
        return loss


def get_optimizer(model, optimizer_type: str, lr: float):
    if optimizer_type == "SGD":
        print("Using SGD as optimizer")
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        print("Using Adam as optimizer")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return optimizer
