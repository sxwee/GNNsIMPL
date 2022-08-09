import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--model",
                    type=str,
                    default="h2gcn",
                    choices=["h2gcn", "gcn"])
parser.add_argument("--dataset",
                    type=str,
                    default="cora",
                    choices=["cora", "pubmed", "citeseer", "actor", "texas"])
parser.add_argument("--input_size", type=int)
parser.add_argument("--hidden_size", type=int, default=64)
parser.add_argument("--output_size", type=int)
parser.add_argument("--epochs", type=int, default=2000)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--drop_prob", type=float, default=0.5)
parser.add_argument("--round", type=int, default=2)


args = parser.parse_args()

if __name__ == "__main__":
    pass
