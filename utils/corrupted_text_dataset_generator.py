import pandas as pd
import random
import string


class TextCorruptor:
    def __init__(self, charset=string.ascii_lowercase):
        self.charset = charset

    def corrupt(self, word):
        if len(word) <= 2:
            return word

        # Pick a random corruption strategy
        method = random.choice(["delete", "insert", "replace", "swap", "none"])
        chars = list(word)
        idx = random.randint(0, len(chars) - 1)

        if method == "delete":
            chars.pop(idx)
        elif method == "insert":
            chars.insert(idx, random.choice(self.charset))
        elif method == "replace":
            chars[idx] = random.choice(self.charset)
        elif method == "swap" and idx < len(chars) - 1:
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]

        return "".join(chars)


def generate_csv(word_list, filename="dataset.csv", num_copies=5):
    corruptor = TextCorruptor()
    data = []

    for word in word_list:
        # Generate multiple corrupted versions of the same word for variety
        for _ in range(num_copies):
            corrupted = corruptor.corrupt(word)
            data.append({"corrupted": corrupted, "clean": word})

    df = pd.DataFrame(data)
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename} with {len(df)} rows.")


# --- Example Usage ---
# You can use a local dictionary file or this list for testing
sample_words = [
    "algorithm",
    "architecture",
    "sequence",
    "encoder",
    "decoder",
    "attention",
    "network",
    "gradient",
    "tensor",
    "pytorch",
] * 100  # Expand the list

generate_csv(sample_words)
