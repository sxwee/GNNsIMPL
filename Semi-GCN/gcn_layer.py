import torch
import torch.nn as nn
import dgl.function as fn

class GCNLayer(nn.Module):
    def __init__(self,in_feats,out_feats,bias=True):
        super(GCNLayer,self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_feats,out_feats))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_feats))
        else:
            self.bias = None

        self.reset_parameter()
        
    def reset_parameter(self):
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self,g,h):
        with g.local_scope():
            h = torch.matmul(h,self.weight)
            g.ndata['h'] = h * g.ndata['norm']
            g.update_all(message_func = fn.copy_u('h','m'),
                            reduce_func=fn.sum('m','h'))
            h = g.ndata['h']
            h = h * g.ndata['norm']
            if self.bias is not None:
                h = h + self.bias
            return h

if __name__ == "__main__":
    pass