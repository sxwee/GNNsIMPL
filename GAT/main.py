from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from gat import GAT
from copy import deepcopy
from sklearn.metrics import accuracy_score

# load dataset
dataset = Planetoid(root='../datasets/PYG/', name='Cora')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = dataset[0].to(device)
h = data.x
feature_dim = h.shape[1]
class_num = dataset.num_classes
# define model
model = GAT(in_feats=feature_dim,
            hidden_feats=64,
            y_num=class_num,
            drop_prob=0.6,
            num_heads=[8, 1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
best_acc, best_model = 0., None
# start training
model.train()
for epoch in range(300):
    out = model(h, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    valid_acc = accuracy_score(data.y[data.val_mask].cpu(),
                               out[data.val_mask].argmax(dim=1).cpu())
    if valid_acc > best_acc:
        best_acc = valid_acc
        best_model = deepcopy(model)
    if (epoch + 1) % 25 == 0:
        print(f"Epoch {epoch + 1}: train_loss: {loss.item():.8f}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
# evaluate on testset
best_model.eval()
pred = best_model(h, data.edge_index)
test_acc = accuracy_score(data.y[data.test_mask].cpu(),
                          pred[data.test_mask].argmax(dim=1).cpu())
print(f'Testset Accuracy: {test_acc:.4f}')
