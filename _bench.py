import heapq, time
from collections import Counter


def train_naive(data, vocab_size):
    """Your current (corrected) logic — the ground truth to match."""
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    data = list(data)
    while len(vocab) < vocab_size:
        c = Counter()
        for pair in zip(data[:-1], data[1:]):
            c[pair] += 1
        if not c:
            break
        new_pair = min(c.items(), key=lambda x: (-x[1], x[0]))[0]
        new_id = len(vocab)
        vocab[new_id] = vocab[new_pair[0]] + vocab[new_pair[1]]
        merges.append(new_pair)
        i, n, out = 0, len(data), []
        while i < n:
            if i < n - 1 and (data[i], data[i + 1]) == new_pair:
                out.append(new_id); i += 2
            else:
                out.append(data[i]); i += 1
        data = out
    return vocab, merges


def train_fast(data, vocab_size):
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    tokens = list(data)
    n = len(tokens)
    if n < 2:
        return vocab, merges

    prev = [i - 1 for i in range(n)]      # -1 == none
    nxt = [i + 1 for i in range(n)]       # n == none
    alive = [True] * n

    counts = Counter()
    positions = {}                         # pair -> set of left indices
    for i in range(n - 1):
        p = (tokens[i], tokens[i + 1])
        counts[p] += 1
        positions.setdefault(p, set()).add(i)

    heap = [(-c, p) for p, c in counts.items()]
    heapq.heapify(heap)

    def bump(p, i, delta):
        counts[p] += delta
        s = positions.setdefault(p, set())
        if delta > 0:
            s.add(i)
        else:
            s.discard(i)
        heapq.heappush(heap, (-counts[p], p))   # always re-publish current count

    while len(vocab) < vocab_size and heap:
        negc, pair = heapq.heappop(heap)
        if -negc != counts.get(pair, 0) or -negc <= 0:
            continue                       # stale entry

        a, b = pair
        new_id = len(vocab)
        vocab[new_id] = vocab[a] + vocab[b]
        merges.append(pair)

        for i in sorted(positions[pair]):  # left-to-right, like the rebuild
            if not alive[i]:
                continue
            j = nxt[i]
            if j >= n or not alive[j] or tokens[i] != a or tokens[j] != b:
                continue
            pl, jr = prev[i], nxt[j]

            if pl >= 0:
                bump((tokens[pl], a), pl, -1)
            if jr < n:
                bump((b, tokens[jr]), j, -1)

            tokens[i] = new_id
            alive[j] = False
            nxt[i] = jr
            if jr < n:
                prev[jr] = i

            if pl >= 0:
                bump((tokens[pl], new_id), pl, +1)
            if jr < n:
                bump((new_id, tokens[jr]), i, +1)

        counts[pair] = 0
        positions[pair] = set()

    return vocab, merges


if __name__ == "__main__":
    with open("data/tinystories.txt", "rb") as f:
        data = f.read(300_000)            # slice so naive finishes quickly

    VS = 400
    t = time.perf_counter()
    v1, m1 = train_naive(data, VS)
    t1 = time.perf_counter() - t

    t = time.perf_counter()
    v2, m2 = train_fast(data, VS)
    t2 = time.perf_counter() - t

    print(f"naive: {t1:6.2f}s   fast: {t2:6.2f}s   speedup: {t1/t2:5.1f}x")
    print("merges identical:", m1 == m2)
    print("vocab  identical:", v1 == v2)
