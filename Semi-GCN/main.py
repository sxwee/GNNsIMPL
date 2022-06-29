import dgl
import torch
from gcn import GCNModel
import torch.optim as optim
import torch.nn as nn
from func import drawPlot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(g, model, optimizer, loss_fn, epochs):
    best_val_acc, best_test_acc = 0, 0
    # gain features and labels of datasets
    features, labels = g.ndata['feat'], g.ndata['label']
    train_mask, val_mask, test_mask = g.ndata['train_mask'], g.ndata[
        'val_mask'], g.ndata['test_mask']
    train_acc_list, val_acc_list, test_acc_list = [], [], []
    for e in range(epochs):
        logits = model(g, features)  # (N, label_nums)
        pred = logits.argmax(1)
        loss = loss_fn(logits[train_mask], labels[train_mask])
        # cal trian acc
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        # cal val acc
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        # cal test acc
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        train_acc_list.append(train_acc.item())
        val_acc_list.append(val_acc.item())
        test_acc_list.append(test_acc.item())

        # save result based on valid dataset
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (e + 1) % 5 == 0:
            print(
                'Epoch {}, loss: {:.3f}, train acc: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'
                .format(e + 1, loss, train_acc, val_acc, best_val_acc,
                        test_acc, best_test_acc))
    drawPlot([train_acc_list, val_acc_list, test_acc_list], "accuracy.png",
             "Acc", ["train", "val", "test"])


if __name__ == "__main__":
    epochs = 200
    hidden_size = 32
    lr = 0.01
    weight_decay = 5e-4
    # download and loading dataset
    dataset = dgl.data.CoraGraphDataset(raw_dir="../Datasets/DGL/")
    g = dataset[0]
    # add self-loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    # gain degree matrix
    degs = g.out_degrees().float()
    # cal D^{-1/2}
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    g.ndata['norm'] = norm.unsqueeze(1)

    model = GCNModel(in_feats=g.ndata['feat'].shape[1],
                     h_feats=hidden_size,
                     num_classes=dataset.num_classes)

    if torch.cuda.is_available():
        g = g.to(device)
        model = model.to(device)

    optimizer = optim.Adam(model.parameters(),
                           lr=lr,
                           weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    train(g, model, optimizer, loss_fn, epochs)
