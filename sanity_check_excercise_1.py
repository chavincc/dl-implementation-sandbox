import torch

from excercise_1_seq2seq import (
    Encoder,
    Decoder,
    word2idx,
    VOCAB_SIZE,
    PADDING_IDX,
    SOS_IDX,
    Seq2SeqManager,
    prepare_model_input,
    outputs2words,
)


# Debug preprocessing
test_word = "qython"
print(test_word)
print(word2idx(test_word))

print()
# Debug encoder
EMB_DIM = 64
HID_DIM = 64
N_LAYERS = 2
test_encoder = Encoder(
    VOCAB_SIZE, EMB_DIM, HID_DIM, n_layers=N_LAYERS, padding_idx=PADDING_IDX
)
debug_emb = test_encoder.embedding(torch.tensor([word2idx(test_word)]))
print("emb out size", debug_emb.size())
rnn_out, rnn_hidden = test_encoder.rnn(debug_emb)
print("rnn out size", rnn_out.size())
print("rnn hidden size", rnn_hidden.size())

print()
# Debug decoder
test_decoder = Decoder(
    VOCAB_SIZE, EMB_DIM, HID_DIM, n_layers=N_LAYERS, padding_idx=PADDING_IDX
)
test_unroll_input = torch.tensor([[SOS_IDX]])
dec_out, dec_hidden = test_decoder(test_unroll_input, rnn_hidden)
print("dec out size", dec_out.size())
print("dec hidden size", dec_hidden.size())

print()
# Debug Seq2SeqManager
sources = ["qython", "deocder"]
targets = ["python", "decoder"]

sources_tensor = prepare_model_input(sources)
targets_tensor = prepare_model_input(targets)
print("PADDING_IDX=", PADDING_IDX)
print("sources", sources_tensor)
print("targets", targets_tensor)

test_model = Seq2SeqManager(test_encoder, test_decoder, VOCAB_SIZE)
raw_outputs = test_model.forward(sources_tensor, targets_tensor, teacher_forcing=False)
print("raw_outputs", raw_outputs.shape)
maxxed_outputs = raw_outputs.argmax(2)
out_words = outputs2words(maxxed_outputs)
print(out_words)
