import tensorflow as tf
import pickle
from model import *


# reverses direction of encoding for a dict
# because current vocab dicts are token:ind
def fliperooni(inDict):
    return {v:k for (k,v) in inDict.items()}

def tokenize_input(inStr):
    new_str = inStr.lower()
    new_str = new_str.split()

    return new_str

def encode_string(inTokens):
    enc_seq = [2] # BOS

    for token in inTokens:
        if token in en_vocab:
            enc_seq.append(en_vocab[token])
        else:
            enc_seq.append(1)

    enc_seq.append(3) # EOS

    """
    if len(enc_seq) < MAX_LENGTH:
        diff = MAX_LENGTH - len(enc_seq)
        for _ in range(diff):
            enc_seq.append(0)

    assert len(enc_seq) == MAX_LENGTH
    """
    return enc_seq

# these are c/p-ed for now, will import from model.py later
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

"""
look ahead mask so that the model can only attend to prior tokens
otherwise it could cheat and look at future tokens
basically a 2d matrix of rows with progressively fewer masked entries, e.g.
[0, 1, 1]
[0, 0, 1]
[0, 0, 0]
"""
def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


def evaluate(enc_inp):
    encoder_input = tf.expand_dims(enc_inp, 0)

    # as the target is english, the first word to the transformer should be the
    # english start token.
    decoder_input = [2]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        print("EOS prob at pos {0} = {1}".format(i,predictions[:,:,3]))
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == 3:
            return tf.squeeze(output, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights


def translate(enc_sent):
    result, attention_weights = evaluate(enc_sent)

    with tf.compat.v1.Session() as sess:
        e = result.numpy()

    predicted_sentence = []
    for i in e:
        try:
            predicted_sentence.append(r_is_vocab[i])
        except KeyError:
            pass

    print('Input: {}'.format(enc_sent))
    print('Raw output: {}'.format(e))
    print('Predicted translation: {}'.format(predicted_sentence))


if __name__ == "__main__":
    saved_model_dir = "D:/train_test/saved_model/"
    tt_dir = "D:/train_test/"

    with open(tt_dir + "en_vocab", "rb") as en_v_file:
        en_vocab = pickle.load(en_v_file)

    with open(tt_dir + "is_vocab", "rb") as is_v_file:
        is_vocab = pickle.load(is_v_file)

    r_en_vocab = fliperooni(en_vocab)
    r_is_vocab = fliperooni(is_vocab)

    # hyperparameters
    BUFFER_SIZE = 20000
    BATCH_SIZE = 32

    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8

    input_vocab_size = len(en_vocab) + 4  # 4 for pad, unk and BOS/EOS.
    target_vocab_size = len(is_vocab) + 4
    dropout_rate = 0.1

    EPOCHS = 20

    MAX_LENGTH = 40


    transformer = Transformer(num_layers, d_model, num_heads, dff,
                              input_vocab_size, target_vocab_size,
                              pe_input=input_vocab_size,
                              pe_target=target_vocab_size,
                              rate=dropout_rate)

    # do a dummy push
    a = tf.expand_dims(tf.cast([2, 4, 5, 6, 3], dtype=tf.int32), 0)
    b = tf.expand_dims(tf.cast([2], dtype=tf.int32), 0)
    m1, m2, m3 = create_masks(a, b)

    dummy1, dummy2 = transformer(a, b, True, m1, m2, m3)

    x = transformer.trainable_weights

    transformer.load_weights(tt_dir)

    y = transformer.trainable_weights

    print(x == y)

    # transformer = tf.keras.models.load_model(saved_model_dir)

    to_translate = input("Type English sentence to translate: \n")

    enc = encode_string(tokenize_input(to_translate))

    translate(enc)
