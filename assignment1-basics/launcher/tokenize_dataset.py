from cs336_basics.bpe_tokenizer import Tokenizer
import argparse
import numpy as np
import tempfile
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text_file")
    parser.add_argument("vocab")
    parser.add_argument("merges")
    parser.add_argument("id_file")
    parser.add_argument("special_tokens", nargs="+")
    args = parser.parse_args()

    tkn = Tokenizer.from_files(args.vocab, args.merges, args.special_tokens)


    with tempfile.NamedTemporaryFile(
        dir="/tmp",
        suffix=".bin",
        delete=False,
    ) as tmp:
        tmp_path = tmp.name
        
    try:
        with open(tmp_path, "ab") as f, open(args.text_file, "r", encoding="utf-8") as text:
            pieces = []
            for piece in tkn.encode_iterable(text):
                pieces.append(piece)
                if len(pieces) > 65535:
                    token_ids = np.asarray(pieces, dtype=np.uint16)
                    token_ids.tofile(f)
                    pieces = []

            if pieces:
                token_ids = np.asarray(pieces, dtype=np.uint16)
                token_ids.tofile(f)

        shutil.move(tmp_path, args.id_file)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

if __name__ == "__main__":
    main()   