import pickle
from .bpe import PAT
import regex as re
from typing import Iterable, Iterator

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]] , special_tokens:list[str] | None=None):
        self.vocab_r: dict[bytes, int] = {}
        self.special_tokens: list[str] | None = special_tokens
        self.merges: dict[tuple[bytes, bytes], int] = {}
        max_id = 0
        for id, token in vocab.items():
            self.vocab_r[token] = id
            if id > max_id:
                max_id = id

        self.vocab = vocab

        if special_tokens is not None:
            st_id = max_id + 1
            for st in special_tokens:
                st_b = st.encode("utf-8")
                if st_b not in self.vocab_r:
                    self.vocab_r[st_b] = st_id
                    st_id += 1

        for i, merge in enumerate(merges):
            self.merges[merge] = i

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode_chunk(self, chunk: str, pretoken_cache: dict[bytes, list[int]]) -> list[int]:
        out: list[int] = []
        for match in re.finditer(PAT, chunk):
            pretoken = match.group().encode('utf-8')
            if pretoken in pretoken_cache:
                out.extend(pretoken_cache[pretoken])
                continue
            bytes_list = [bytes([byte]) for byte in pretoken]
            while True:
                least_rank = [len(self.merges), None]
                for i in range(len(bytes_list)-1):
                    tup = (bytes_list[i], bytes_list[i+1])
                    if tup in self.merges and self.merges[tup] < least_rank[0]:
                        least_rank = [self.merges[tup], tup]

                if least_rank[1] is None:
                    break

                seq = []
                i = 0
                while (i < len(bytes_list) - 1):
                    tup = (bytes_list[i], bytes_list[i+1])
                    if tup == least_rank[1]:
                        seq.append(tup[0]+tup[1])
                        i += 2
                    else:
                        seq.append(tup[0])
                        i += 1
                    if i == len(bytes_list) - 1:
                        seq.append(bytes_list[-1])
                bytes_list = seq

            cache = [self.vocab_r[b] for b in bytes_list]
            pretoken_cache[pretoken] = cache
            out.extend(cache)
        return out

    def encode(self, text: str, pretoken_cache: dict[bytes, list[int]] = {}) -> list[int]:
        if not self.special_tokens:
            return self.encode_chunk(text, pretoken_cache)
        
        out: list[int] = []
        ordered_tokens = sorted(self.special_tokens, key=len, reverse=True)
        st_pattern = "|".join(re.escape(token) for token in ordered_tokens)
        chunk_begin = 0
        for match in re.finditer(st_pattern, text):
            chunk_end = match.start()
            if chunk_end > chunk_begin:
                chunk = text[chunk_begin:chunk_end]
                out.extend(self.encode_chunk(chunk, pretoken_cache))

            st_id = self.vocab_r[match.group().encode("utf-8")]
            out.append(st_id)

            chunk_begin = match.end()

        if len(text) > chunk_begin:
            out.extend(self.encode_chunk(text[chunk_begin:], pretoken_cache))
        return out

    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[i] for i in ids]).decode('utf-8', errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pretoken_cache: dict[bytes, list[int]] = {}
        for line in iterable:
            yield from self.encode(line, pretoken_cache)