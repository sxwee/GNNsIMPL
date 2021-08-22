import torch.nn.functional as F
import torch.nn as nn
from gcn_layer import GCNLayer


class GCNModel(nn.Module):
    def __init__(self,in_feats,h_feats,num_classes,bias=True):
        super(GCNModel,self).__init__()
        self.conv1 = GCNLayer(in_feats,h_feats,bias)
        self.conv2 = GCNLayer(h_feats,num_classes,bias)
    
    def forward(self,g,in_feat):
        h = self.conv1(g,in_feat)
        h = F.relu(h)
        h = self.conv2(g,h)
        return h

if __name__ == "__main__":
    pass