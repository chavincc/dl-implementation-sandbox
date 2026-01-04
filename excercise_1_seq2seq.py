import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import string
from typing import List


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, padding_idx=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.rnn = nn.GRU(emb_dim, hid_dim, num_layers=n_layers, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        return hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, padding_idx=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.rnn = nn.GRU(emb_dim, hid_dim, num_layers=n_layers, batch_first=True)
        # Predicting output vocab for each timestep
        self.fc_out = nn.Linear(hid_dim, vocab_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output)
        return prediction, hidden


class Seq2SeqManager(nn.Module):
    def __init__(self, encoder, decoder, vocab_size):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size

    def forward(self, src, trg, teacher_forcing=True):
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        # Tensor to store all decoder predictions
        # (sequential and dynamic prediction)
        # Shape: [batch_size, trg_len, vocab_size]
        outputs = torch.zeros((batch_size, trg_len, self.vocab_size))
        # Setting the first column (1-hot) to <SOS>
        outputs[:, 0, SOS_IDX] = 1.0

        # Get context vector from input
        ctx = self.encoder(src)
        hidden = ctx

        # Get the "<SOS>" column.
        # input Shape: [batch_size, 1]
        input = trg[:, 0].unsqueeze(1)  # without unsqueeze(1) the shape is [batch_size]

        # Decoding loop
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden)

            # Save prediction
            outputs[:, t, :] = output.squeeze(1)

            top1 = output.argmax(2)
            input = trg[:, t].unsqueeze(1) if teacher_forcing else top1

        return outputs


SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
VOCAB_SET = list(string.ascii_lowercase) + SPECIAL_TOKENS
VOCAB_SIZE = len(VOCAB_SET)
UNKNOWN_IDX = VOCAB_SET.index("<UNK>")
PADDING_IDX = VOCAB_SET.index("<PAD>")
SOS_IDX = VOCAB_SET.index("<SOS>")
EOS_IDX = VOCAB_SET.index("<EOS>")


def char2idx(char: str) -> int:
    try:
        idx = VOCAB_SET.index(char)
    except:  # OOV
        idx = UNKNOWN_IDX
    return idx


def word2idx(word: str) -> List[int]:
    return [SOS_IDX, *list(map(char2idx, word)), EOS_IDX]


def prepare_model_input(words: List[str]) -> torch.Tensor:
    sequences = []
    for word in words:
        tensor = torch.tensor(word2idx(word))
        sequences.append(tensor)

    padded_sequences = torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=True, padding_value=PADDING_IDX
    )
    return padded_sequences


def outputs2words(outputs: torch.Tensor) -> List[str]:
    # outputs Shape: [batch_size, max_output_length]
    word_index_list = outputs.tolist()
    output_list = []
    for row in word_index_list:
        word = ""
        for idx in row:
            word += VOCAB_SET[idx]
        output_list.append(word)

    return output_list


# Training loop
if __name__ == "__main__":
    # FILL HERE
    pass
