import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder

class UncertaintyPredictor(nn.Module):
    def __init__(self, model: Encoder):
        super().__init__()
        self.device = "cuda"
        self.model = model.eval()
        self.model.return_hidden = True
        self.predictor = nn.Sequential(
            nn.Linear(model.hidden_size, model.hidden_size),
            nn.ReLU(),
            nn.Linear(model.hidden_size, 1),
        )
        
    def forward(self, x, mask):
        with torch.no_grad():
            x, hidden = self.model(x, mask)
        x = x / x.norm(dim=1, keepdim=True)
        sigma = self.predictor(hidden).mean(1)
        return torch.cat([x, sigma], dim=-1)


def cosine_similarity_uncertainty_loss(pred_and_sigma, target):
    # pred_and_sigma: B x (3 + 1)
    # target: B x 3
    pred = pred_and_sigma[:, :-1]
    sigma2 = pred_and_sigma[:, -1] ** 2
    cosine_similarity = F.cosine_similarity(pred, target, dim=1)
    cosine_loss = 1 - cosine_similarity.mean()
    loss = -torch.mean(torch.log(sigma2) + cosine_loss / sigma2)
    return loss
