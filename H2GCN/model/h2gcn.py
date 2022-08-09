import torch
import torch.nn as nn
import torch.nn.functional as F


class H2GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, drop_prob,
                 round):
        super().__init__()
        self.round = round
        self.embed = nn.Sequential(nn.Linear(in_channels, hidden_channels),
                                   nn.ReLU())

        self.dropout = nn.Dropout(drop_prob)
        self.classification = nn.Linear(hidden_channels * (2**(round + 1) - 1),
                                        out_channels)

    def forward(self, x, adj, adj_2hop):
        hidden_reps = []
        x = self.embed(x)
        hidden_reps.append(x)
        for _ in range(self.round):
            r1 = adj.matmul(x)
            r2 = adj_2hop.matmul(x)
            x = torch.cat([r1, r2], dim=-1)
            hidden_reps.append(x)
        hf = self.dropout(torch.cat(hidden_reps, dim=-1))
        return F.log_softmax(self.classification(hf), dim=1)


if __name__ == "__main__":
    pass
