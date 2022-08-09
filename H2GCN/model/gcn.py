import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats) -> None:
        super().__init__()
        self.gcn_conv1 = GCNConv(in_feats, hidden_size)
        self.gcn_conv2 = GCNConv(hidden_size, out_feats)

    def forward(self, x, adj):
        x = self.gcn_conv1(x, adj)
        x = F.relu(x)
        x = self.gcn_conv2(x, adj)
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    pass
