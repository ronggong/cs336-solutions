from cs336_basics.bpe_tokenizer import Tokenizer
import time

tkn_time = 0
num_all_b = 0

tkn_tiny = Tokenizer.from_files("_bpe_tiny/vocab.pkl", "_bpe_tiny/merges.pkl", ["<|endoftext|>"])

with open("tiny_2000.txt", "r") as f:
    num_b = len(f.read().encode("utf-8"))
    num_all_b += num_b

with open("tiny_2000.txt", "r") as f:
    start = time.perf_counter()
    pieces = tkn_tiny.encode_iterable(f)
    num_t = len(list(pieces))
    tkn_time += time.perf_counter() - start

print("tiny compress ratio: ", num_b / num_t)

tkn_owt = Tokenizer.from_files("_openwebtext/vocab.pkl", "_openwebtext/merges.pkl", ["<|endoftext|>"])

with open("data_2000.txt", "r") as f:
    num_b = len(f.read().encode("utf-8"))

with open("data_2000.txt", "r") as f:
    start = time.perf_counter()
    pieces = tkn_owt.encode_iterable(f)
    num_t = len(list(pieces))
    tkn_time += time.perf_counter() - start

with open("data_2000.txt", "r") as f:
    start = time.perf_counter()
    pieces = tkn_tiny.encode_iterable(f)
    num_t_tiny = len(list(pieces))
    tkn_time += time.perf_counter() - start

print("owt compress ratio: ", num_b / num_t, "owt tiny compress ratio: ", num_b / num_t_tiny)

num_all_b += 2*num_b

print("bytes / second", num_all_b / tkn_time)