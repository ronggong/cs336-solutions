import argparse
from cs336_basics.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data")
    parser.add_argument("valid_data")
    parser.add_argument("conf_yaml")
    parser.add_argument("outdir")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    train(args.train_data, args.valid_data, args.conf_yaml, args.outdir, args.checkpoint)