import argparse
import pyarrow.parquet as pq
from pathlib import Path
import random

def concat_parquet(parquet_filepaths, output_train, output_valid):
    first_write_trn = False
    first_write_valid = False
    rng = random.Random(42)
    with open(output_train, "w", encoding="utf-8") as f_trn, open(output_valid, "w", encoding="utf-8") as f_valid:
        for i, parquet_fp in enumerate(parquet_filepaths):
            file = pq.ParquetFile(parquet_fp)
            for batch in file.iter_batches():
                for text in batch.column("text").to_pylist():
                    if text is not None:
                        if rng.random() < 0.1:
                            if not first_write_valid:
                                first_write_valid = True
                            else:
                                f_valid.write("<|endoftext|>")
                            f_valid.write(text)
                        else:
                            if not first_write_trn:
                                first_write_trn = True
                            else:
                                f_trn.write("<|endoftext|>")
                            f_trn.write(text) 


            print(f"{i+1}/{len(parquet_filepaths)} written")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_path")
    parser.add_argument("output_train")
    parser.add_argument("output_valid")
    args = parser.parse_args()

    parquet_files = sorted(Path(args.parquet_path).glob("*.parquet"))

    if len(parquet_files) == 0:
        print("No parquet file found")
        exit(1)

    concat_parquet(parquet_files, args.output_train, args.output_valid)