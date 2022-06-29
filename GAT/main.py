from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from gat import GAT
from copy import deepcopy
from gtrick import random_feature

dataset = Planetoid(root='../datasets/PYG/', name='Cora')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = dataset[0].to(device)
h = random_feature(data.x)
feature_dim = h.shape[1]
class_num = dataset.num_classes
model = GAT(in_feats=feature_dim,
            hidden_feats=64,
            y_num=class_num,
            drop_prob=0.6,
            num_heads=[8, 1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
best_acc, best_model = 0., None
model.train()
for epoch in range(600):
    optimizer.zero_grad()
    out = model(h, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    valid_acc = (out[data.val_mask].argmax(
        dim=1) == data.y[data.val_mask]).sum()
    if valid_acc > best_acc:
        best_acc = valid_acc
        best_model = deepcopy(model)
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}: loss: {loss.item()}")
    loss.backward()
    optimizer.step()

best_model.eval()
pred = best_model(h, data.edge_index).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Testset Accuracy: {acc:.4f}')
