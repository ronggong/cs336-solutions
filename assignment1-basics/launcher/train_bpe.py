import argparse
import pickle
import time
import threading
import psutil
from pathlib import Path
from cs336_basics.bpe import train_bpe

def peak_mem(mem, stop_event, raise_error):
    try:
        while not stop_event.is_set():
            rss = 0
            parent = psutil.Process()
            rss += parent.memory_info().rss
            for p in parent.children(recursive=True):
                try:
                    rss += p.memory_info().rss
                except (psutil.NoSuchProcess, psutil.ZombieProcess) as error:
                    continue

            if rss > mem['rss']:
                mem['rss'] = rss

            stop_event.wait(0.1)
    except Exception as error:
        raise_error.append(error)
        stop_event.set()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output_path")
    parser.add_argument("vocab_size", type=int)
    parser.add_argument("special_token", nargs="+")
    args = parser.parse_args()

    stop_event = threading.Event()
    peak_m = {'rss': 0}
    raise_error = []

    t = threading.Thread(target=peak_mem, args=[peak_m, stop_event, raise_error])

    t.start()

    try:
        start_time = time.perf_counter()
        vocab, merges = train_bpe(args.input, args.vocab_size, args.special_token)
        end_time = time.perf_counter()
    finally:
        stop_event.set()
        t.join()

    if raise_error:
        raise RuntimeError("Memory monitoring failed") from raise_error[0]

    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    with open(Path(args.output_path) / "log", "w") as f:
        f.write(f"time: {end_time - start_time}\n")
        f.write(f"peak mem: {peak_m['rss'] / (1024**3)}\n")

    with open(Path(args.output_path) / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open(Path(args.output_path) / "merges.pkl", "wb") as f:
        pickle.dump(merges, f)

if __name__ == '__main__':
    main()