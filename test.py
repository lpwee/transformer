# from transformer_lm.tokenizer import train_bpe
input_path = "data/tinystories.txt"
vocab_size = 512

from collections import Counter
# train_bpe(input_path, 512)

with open(input_path, "rb") as f:
    data = f.read()

# init vocab
vocab = {i: bytes([i]) for i in range(256)}
merges = []

while len(vocab) < vocab_size:
    c = Counter()
    for pair in zip(data[:-1], data[1:]): #(byte, byte)
        c[pair] += 1

    most_freq = min(c.items(), key=lambda x: (-x[1], x[0]))
    new_pair = most_freq[0] #(id, id)
    new_entry_id = len(vocab.keys())
    vocab[new_entry_id] = vocab[new_pair[0]] + vocab[new_pair[1]]
    merges.append(new_pair)

    i, n = 0, len(data)
    new_corpus = []
    while i < n:
        if i < n-1 and (data[i], data[i+1]) == new_pair:
            new_corpus.append(new_entry_id) #replace the corpus with the newly minted vocab byte
            i += 2
        else:
            new_corpus.append(data[i])
            i += 1

    data = new_corpus

print(vocab, merges)


# Returns:
# vocab: ``dict[int, bytes]`` mapping token ID to byte string.
# merges: ``list[tuple[int, int]]`` merge pairs in order.

# return vocab, merges
