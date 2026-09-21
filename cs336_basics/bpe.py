import os
import regex as re
import heapq
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from .pretokenization_example import find_chunk_boundaries
from itertools import repeat

NUM_PROCESSES = int(os.environ.get("NUM_PROCESS", 4))
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def init_vocab(special_tokens: list[str]):

    vocab: dict[int, bytes] = {}

    i = 0
    for token in special_tokens:
        vocab[i] = token.encode("utf-8")
        i += 1

    for j in range(256):
        vocab[i] = bytes([j])
        i += 1

    return vocab

def pretoken_process(input_path: str, special_tokens: list[str], start, end) -> dict[tuple[bytes, ...], int]:
    out : dict[tuple[bytes, ...], int] = defaultdict(int)

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        for sub_chunk in re.split("|".join(re.escape(token) for token in special_tokens), chunk):
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            for match in re.finditer(PAT, sub_chunk):
                k = tuple(bytes([byte]) for byte in match.group().encode('utf-8'))
                out[k] += 1

    return out

def pretokenization(input_path: str, special_tokens: list[str]) -> list[tuple[tuple[bytes, ...], int]]:
    out : dict[tuple[bytes, ...], int] = defaultdict(int)

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, NUM_PROCESSES * 16, b"<|endoftext|>")

    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        results = executor.map(pretoken_process, repeat(input_path), repeat(special_tokens), boundaries[:-1], boundaries[1:])

        for res in results:
            for k, v in res.items():
                out[k] += v

    return list(out.items())

""" def merge(pretok_dict: dict[tuple[bytes, ...], int], merges: list[tuple[bytes, bytes]]):
    # build frequency dict
    frequency: dict[tuple[bytes, bytes], int] = defaultdict(int)
    for tup in pretok_dict:
        if len(tup) < 2:
            continue
        mul = pretok_dict[tup]
        for i in range(len(tup) - 1):
            frequency[(tup[i], tup[i+1])] += mul
    if len(frequency) == 0:
        return pretok_dict, merges

    # best pair according to frequency and lexicographical order
    best_pair = max(frequency, key=lambda pair: (frequency[pair], pair))

    # new merges
    out_merges = merges.copy()
    out_merges.append(best_pair)

    # new vocab
    out_dict = {}
    for tup in pretok_dict:
        if len(tup) < 2:
            out_dict[tup] = pretok_dict[tup]
            continue
        out_tup = []
        i = 0
        while (i < len(tup) - 1):
            if (tup[i], tup[i+1]) == best_pair:
                out_tup.append(tup[i]+tup[i+1])
                i += 2
            else:
                out_tup.append(tup[i])
                i += 1
            if i == len(tup) - 1:
                out_tup.append(tup[i])
                
        out_dict[tuple(out_tup)] = pretok_dict[tup]

    return out_dict, out_merges """

def count_pairs(tup: tuple[bytes, ...]):
    out: dict[tuple[bytes, bytes], int] = defaultdict(int)
    for i in range(len(tup) - 1):
        out[(tup[i], tup[i+1])] += 1
    return out

class RevPair:
    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair

    def __lt__(self, other):
        return self.pair > other.pair

def merge(pretokens: list[tuple[tuple[bytes, ...], int]], 
          merges: list[tuple[bytes, bytes]],
          pair_pretoken: dict[tuple[bytes, bytes], set],
          frequency: dict[tuple[bytes, bytes], int],
          freq_heap: list[tuple[int, tuple[bytes, bytes]]]):

    # best pair according to frequency and lexicographical order
    while freq_heap:
        best_freq, rev_pair = heapq.heappop(freq_heap)
        best_pair = rev_pair.pair
        if best_pair in frequency and frequency[best_pair] == -best_freq:
            break
    else:
        return pretokens, merges, pair_pretoken, frequency

    # new merges
    merges.append(best_pair)

    best_pair_pretoken_ids = pair_pretoken[best_pair]
    for pretoken_id in list(best_pair_pretoken_ids):
        tup, mul = pretokens[pretoken_id]
        if len(tup) < 2:
            continue

        seq = []
        i = 0
        while (i < len(tup) - 1):
            if (tup[i], tup[i+1]) == best_pair:
                seq.append(tup[i] + tup[i+1])
                i += 2
            else:
                seq.append(tup[i])
                i += 1
            if i == len(tup) - 1:
                seq.append(tup[i])

        pretokens[pretoken_id] = (tuple(seq), mul)

        old_pairs = count_pairs(tup)
        new_pairs = count_pairs(tuple(seq))

        # removed pairs
        for pair in (old_pairs.keys() - new_pairs.keys()):
            pair_pretoken[pair].remove(pretoken_id)

        # added pairs
        for pair in (new_pairs.keys() - old_pairs.keys()):
            pair_pretoken[pair].add(pretoken_id)

        for pair in new_pairs.keys() | old_pairs.keys():
            frequency[pair] += (new_pairs.get(pair, 0) - old_pairs.get(pair, 0)) * mul
            if frequency[pair] == 0:
                del frequency[pair]
            else:
                heapq.heappush(freq_heap, (-frequency[pair], RevPair(pair)))

    return pretokens, merges, pair_pretoken, frequency

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) \
    -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    merges: list[tuple[bytes, bytes]] = []

    vocab = init_vocab(special_tokens)

    pretokens = pretokenization(input_path, special_tokens)

    pair_pretoken: dict[tuple[bytes, bytes], set] = defaultdict(set)
    frequency: dict[tuple[bytes, bytes], int] = defaultdict(int)
    freq_heap: list[tuple[int, tuple[bytes, bytes]]] = []

    # build frequency dict
    for pretoken_id, (tup, mul) in enumerate(pretokens):
        if len(tup) < 2:
            continue
        for i in range(len(tup) - 1):
            pair = (tup[i], tup[i+1])
            frequency[pair] += mul
            pair_pretoken[pair].add(pretoken_id)

    # No mergable pairs
    if not frequency:
        return vocab, merges

    # build heapq for best_pair
    for pair, count in frequency.items():
        heapq.heappush(freq_heap, (-count, RevPair(pair)))

    while (True):
        pretokens, merges, pair_pretoken, frequency = merge(pretokens, merges, pair_pretoken, frequency, freq_heap)

        # rebuild heapq for memroy
        if len(freq_heap) > 2 * len(frequency):
            freq_heap.clear()
            for pair, count in frequency.items():
                heapq.heappush(freq_heap, (-count, RevPair(pair)))

        if len(frequency) == 0 or len(vocab) + len(merges) == vocab_size:
            i = len(vocab)
            for pair in merges:
                vocab[i] = pair[0] + pair[1]
                i += 1
            return vocab, merges
