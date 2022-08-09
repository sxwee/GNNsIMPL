import torch
import torch.nn as nn
from config import args
from model.h2gcn import H2GCN
from model.gcn import GCN
import torch.optim as optim
from torch_geometric.datasets import Planetoid, WebKB
import torch_geometric.transforms as T
from copy import deepcopy
from utils import hopNeighborhood, norm_adj


def train(g, model, optimizer, loss_fn):
    """
    g: graph
    """
    global adj
    global adj_2hop

    model.train()
    g = g.to(device)
    adj_2hop = adj_2hop.to(device)
    feats = g['x']
    labels = g['y']
    if args.dataset == "texas":
        train_mask = g['train_mask'][:, 0]
    else:
        train_mask = g['train_mask']
    if args.model == "h2gcn":
        logits = model(feats, adj, adj_2hop)
    elif args.model == "gcn":
        logits = model(feats, adj)
    pred = logits.argmax(dim=-1)
    train_loss = loss_fn(logits[train_mask], labels[train_mask])
    train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    return train_loss, train_acc


def evaluate(g, model, mask='val'):
    global adj
    global adj_2hop

    model.eval()
    g = g.to(device)
    feats = g['x']
    labels = g['y']
    with torch.no_grad():
        if args.model == "h2gcn":
            logits = model(feats, adj, adj_2hop)
        elif args.model == "gcn":
            logits = model(feats, adj)
    pred = logits.argmax(dim=-1)
    if mask == 'val':
        if args.dataset == "texas":
            eval_mask = g['val_mask'][:, 0]
        else:
            eval_mask = g['val_mask']
    else:
        if args.dataset == "texas":
            eval_mask = g['test_mask'][:, 0]
        else:
            eval_mask = g['test_mask']
    eval_acc = (pred[eval_mask] == labels[eval_mask]).float().mean()
    return eval_acc


if __name__ == "__main__":
    device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.
                          is_available() else "cpu")
    if args.dataset == "cora":

        dataset = Planetoid("dataset",
                            name="Cora",
                            transform=T.ToSparseTensor())
    elif args.dataset == "pubmed":
        dataset = Planetoid("dataset",
                            name="PubMed",
                            transform=T.ToSparseTensor())
    elif args.dataset == "citeseer":
        dataset = Planetoid("dataset",
                            name="CiteSeer",
                            transform=T.ToSparseTensor())
    elif args.dataset == "texas":
        dataset = WebKB("dataset", "Texas", transform=T.ToSparseTensor())
    args.input_size = dataset.num_features
    args.output_size = dataset.num_classes
    g = dataset[0]
    # print(g['train_mask'].sum(), g['val_mask'].sum(), g['test_mask'].sum())
    adj = norm_adj(g.adj_t, add_self_loops=False)
    adj_2hop = norm_adj(hopNeighborhood(g.adj_t), add_self_loops=False)
    adj = adj.to(device)
    adj_2hop = adj_2hop.to(device)
    print("loading dataset {} done".format(args.dataset))

    if args.model == "h2gcn":
        model = H2GCN(in_channels=args.input_size,
                      hidden_channels=args.hidden_size,
                      out_channels=args.output_size,
                      drop_prob=args.drop_prob,
                      round=args.round)
    elif args.model == "gcn":
        model = GCN(in_feats=args.input_size,
                    hidden_size=args.hidden_size,
                    out_feats=args.output_size)
    model = model.to(device)
    print("loading model {} done".format(args.model))

    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    loss_fn = nn.NLLLoss()

    best_val_acc, best_model = 0., None
    for i in range(args.epochs):
        train_loss, train_acc = train(g, model, optimizer, loss_fn)
        val_acc = evaluate(g, model, mask='val')
        if (i + 1) % 200 == 0:
            print("ep{}: train loss: {:.4f} train acc: {:.4f} val acc: {:.4f}".
                  format(i + 1, train_loss, train_acc, val_acc))

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_model = deepcopy(model)

    test_acc = evaluate(g, best_model, mask='test')
    print("test acc: {:.4f}".format(test_acc))
