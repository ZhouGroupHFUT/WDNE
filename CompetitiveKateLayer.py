
import torch
from torch import nn
import warnings


class KCompetitiveLayer(nn.Module):
    """
      dim_input :
      act : String, activation function "
    """

    def __init__(self, ktop, alpha_factor):

        super(KCompetitiveLayer, self).__init__()
        self.ktop = ktop
        self.alpha_factor = alpha_factor

    def forward(self, x):

        dim_input = x.size()[1]
        if self.ktop > dim_input:
            warnings.warn(f"Warning: KCompetitiveLayer's neurons ({self.ktop}) should not be larger than number of input features({dim_input}). It had been resized to {dim_input}.")
            self.ktop = dim_input

        # Positive neurons computation
        POS_ktop = int(self.ktop/2)
        POS_values = (x + torch.abs(x))/2
        POS_topk_values, POS_topk_indices = torch.topk(POS_values, k = POS_ktop)
        POS_topk_values = POS_topk_indices.cuda()
        POS_topk_indices = POS_topk_indices.cuda()
        POS_topk_range = torch.tile(torch.unsqueeze(torch.arange(0, POS_topk_indices.size()[0]), 1), [1, POS_ktop]).cuda()
        POS_full_indices = torch.reshape(torch.stack([POS_topk_range, POS_topk_indices], axis=2), [-1, 2])
        POS_sparse_values = torch.reshape(POS_topk_values, [-1])
        POS_reset = torch.sparse_coo_tensor(indices=POS_full_indices.t(), values=POS_sparse_values, size=x.size()).to_dense()
        POS_tmp = self.alpha_factor * torch.sum(POS_values - POS_reset, 1, keepdims=True)
        POS_reset = torch.sparse_coo_tensor(indices=POS_full_indices.t(), values=torch.reshape(POS_topk_values+POS_tmp, [-1]), size=x.size()).to_dense()

        # Negative neurons computation
        NEG_ktop = self.ktop - int(self.ktop/2)
        NEG_values = (x - torch.abs(x))/2
        NEG_topk_values, NEG_topk_indices = torch.topk(-NEG_values, largest=True, k = NEG_ktop)
        NEG_topk_range = torch.tile(torch.unsqueeze(torch.arange(0, NEG_topk_indices.size()[0]), 1), [1, NEG_ktop]).cuda()
        NEG_full_indices = torch.reshape(torch.stack([NEG_topk_range, NEG_topk_indices], axis=2), [-1, 2])
        NEG_sparse_values = torch.reshape(NEG_topk_values, [-1])
        NEG_reset = torch.sparse_coo_tensor(indices = NEG_full_indices.t(), values=NEG_sparse_values, size=x.size()).to_dense()
        NEG_tmp = self.alpha_factor * torch.sum(-NEG_values - NEG_reset, 1, keepdims=True)
        NEG_reset = torch.sparse_coo_tensor(indices = NEG_full_indices.t(), values=torch.reshape(NEG_topk_values+NEG_tmp, [-1]), size=x.size()).to_dense()

        total_reset = POS_reset - NEG_reset
        return total_reset
