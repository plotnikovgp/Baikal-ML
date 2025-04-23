import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder

class UncertaintyPredictor(nn.Module):
    def __init__(self, model: Encoder):
        super().__init__()
        self.device = "cuda"
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.return_hidden = True
        self.predictor = nn.Sequential(
            nn.Linear(model.hidden_size, model.hidden_size),
            nn.ReLU(),
            nn.Linear(model.hidden_size, model.hidden_size),
            nn.ReLU(),
            nn.Linear(model.hidden_size, 3),
        )
        
    def forward(self, x, mask):
        with torch.no_grad():
            x, hidden = self.model(x, mask)
        x = x / x.norm(dim=1, keepdim=True)
        log_sigma = self.predictor(hidden).mean(1)
        return torch.cat([x, log_sigma], dim=-1)


def uncertainty_loss(pred_and_log_sigma2, target):
    # pred_and_log_sigma2: B x (3 + 3)
    # target: B x 3
    pred, log_sigma2 = pred_and_log_sigma2[:, :3], pred_and_log_sigma2[:, 3:]
    pred_sigma2 = torch.exp(log_sigma2)
    # print(pred.shape, target.shape, log_sigma2.shape)

    squared_error = (pred - target) ** 2

    # NLL Loss for Gaussian: log(sigma^2) + [(y - mu)^2 / sigma^2]
    loss = torch.mean(log_sigma2 + squared_error / pred_sigma2)
    return loss
