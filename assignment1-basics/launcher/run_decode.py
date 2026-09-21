import argparse
from cs336_basics.decode import decode

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("conf_yaml")
    parser.add_argument("checkpoint")
    parser.add_argument("vocab")
    parser.add_argument("merges")
    parser.add_argument("prompt")
    parser.add_argument("outdir")
    parser.add_argument("special_tokens", nargs="+")
    args = parser.parse_args()

    decode(args.conf_yaml, args.checkpoint, args.vocab, args.merges, args.special_tokens, args.prompt, args.outdir)