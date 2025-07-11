import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Example parallel sentence pairs (Turkish → English)
pairs = [
    ("merhaba", "hello"),
    ("nasılsın", "how are you"),
    ("ben iyiyim", "i am fine"),
    ("seni seviyorum", "i love you"),
    ("teşekkür ederim", "thank you"),
    ("görüşürüz", "see you"),
    ("günaydın", "good morning"),
    ("iyi geceler", "good night"),
    ("ne yapıyorsun", "what are you doing"),
    ("nerelisin", "where are you from"),
    ("adın ne", "what is your name"),
    ("memnun oldum", "nice to meet you"),
    ("yardım eder misin", "can you help me"),
    ("özür dilerim", "i am sorry"),
    ("anlamıyorum", "i do not understand"),
    ("evet", "yes"),
    ("hayır", "no"),
    ("belki", "maybe"),
    ("ne zaman", "when"),
    ("nerede", "where"),
]

# Add start and end tokens to target sequences
input_texts = [src for src, tgt in pairs]
target_texts = ['startseq ' + tgt + ' endseq' for _, tgt in pairs]

# Tokenize source and target sequences
input_tokenizer = Tokenizer()
target_tokenizer = Tokenizer()

input_tokenizer.fit_on_texts(input_texts)
target_tokenizer.fit_on_texts(target_texts)

input_sequences = input_tokenizer.texts_to_sequences(input_texts)
target_sequences = target_tokenizer.texts_to_sequences(target_texts)

input_maxlen = max(len(seq) for seq in input_sequences)
target_maxlen = max(len(seq) for seq in target_sequences)

# Pad sequences
encoder_input_data = pad_sequences(input_sequences, maxlen=input_maxlen, padding='post')
decoder_input_data = pad_sequences([seq[:-1] for seq in target_sequences], maxlen=target_maxlen-1, padding='post')
decoder_target_data = pad_sequences([seq[1:] for seq in target_sequences], maxlen=target_maxlen-1, padding='post')

# Vocabulary sizes
num_encoder_tokens = len(input_tokenizer.word_index) + 1
num_decoder_tokens = len(target_tokenizer.word_index) + 1

# One-hot encoding for targets
decoder_target_onehot = tf.keras.utils.to_categorical(decoder_target_data, num_classes=num_decoder_tokens)

# Shared embedding layers
enc_emb_layer = Embedding(input_dim=num_encoder_tokens, output_dim=64)
dec_emb_layer = Embedding(input_dim=num_decoder_tokens, output_dim=64)

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb = enc_emb_layer(encoder_inputs)
_, state_h, state_c = LSTM(64, return_state=True)(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb = dec_emb_layer(decoder_inputs)
decoder_lstm = LSTM(64, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Combined training model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_onehot,
    batch_size=8,
    epochs=600,
    verbose=1
)

# Inference models
# Encoder inference model
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder inference model
decoder_state_input_h = Input(shape=(64,))
decoder_state_input_c = Input(shape=(64,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb2 = dec_emb_layer(decoder_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2
)

# Word index mappings
reverse_target_word_index = {v: k for k, v in target_tokenizer.word_index.items()}
target_word_index = target_tokenizer.word_index

# Translation function
def translate_sequence(input_sentence):
    seq = input_tokenizer.texts_to_sequences([input_sentence])
    seq = pad_sequences(seq, maxlen=input_maxlen, padding='post')
    states_value = encoder_model.predict(seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_word_index['startseq']

    decoded_sentence = []
    for _ in range(target_maxlen):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_target_word_index.get(sampled_token_index, '')
        if sampled_word == 'endseq':
            break
        decoded_sentence.append(sampled_word)
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return ' '.join(decoded_sentence)

# Test translations
print("merhaba →", translate_sequence("merhaba"))
print("nasılsın →", translate_sequence("nasılsın"))
print("teşekkür ederim →", translate_sequence("teşekkür ederim"))
print("yardım eder misin →", translate_sequence("yardım eder misin"))
print("adın ne →", translate_sequence("adın ne"))
