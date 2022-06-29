import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, add_remaining_self_loops


class GATConv(MessagePassing):
    def __init__(self, in_feats, out_feats, alpha, drop_prob, num_heads):
        super().__init__(aggr="add")
        self.drop_prob = drop_prob
        self.num_heads = num_heads
        self.out_feats = out_feats // num_heads
        self.lin = nn.Linear(in_feats, self.out_feats *
                             self.num_heads, bias=False)
        self.a = nn.Linear(2*self.out_feats, 1)
        self.leakrelu = nn.LeakyReLU(alpha)

    def forward(self, x, edge_index):
        edge_index, _ = add_remaining_self_loops(edge_index)
        h = self.lin(x)
        h_prime = self.propagate(edge_index, x=h)
        return h_prime

    def message(self, x_i, x_j, edge_index_i):
        x_i = x_i.view(-1, self.num_heads, self.out_feats)
        x_j = x_j.view(-1, self.num_heads, self.out_feats)
        e = self.a(torch.cat([x_i, x_j], dim=-1)).permute(1, 0, 2)
        e = self.leakrelu(e.permute(1, 0, 2))
        alpha = softmax(e, edge_index_i)
        alpha = F.dropout(alpha, self.drop_prob, self.training)
        return (x_j * alpha).view(x_j.size(0), -1)


class GAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, y_num,
                 alpha=0.2, drop_prob=0., num_heads=[1, 1]):
        super().__init__()
        self.drop_prob = drop_prob
        self.gatconv1 = GATConv(
            in_feats, hidden_feats, alpha, drop_prob, num_heads[0])
        self.gatconv2 = GATConv(
            hidden_feats, y_num, alpha, drop_prob, num_heads[1])

    def forward(self, x, edge_index):
        x = self.gatconv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, self.drop_prob, self.training)
        out = self.gatconv2(x, edge_index)
        return F.log_softmax(out, dim=1)


if __name__ == "__main__":
    conv = GATConv(in_feats=64, out_feats=64, alpha=0.2,
                   num_heads=8, drop_prob=0.2)
    x = torch.rand(4, 64)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 0, 2, 0, 3], [1, 0, 2, 1, 2, 0, 3, 0]], dtype=torch.long)
    x = conv(x, edge_index)
    print(x.shape)
