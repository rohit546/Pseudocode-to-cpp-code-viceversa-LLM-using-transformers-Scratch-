import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Custom Styling
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        color: white;
        background: linear-gradient(90deg, #4A90E2, #50E3C2);
        padding: 15px;
        border-radius: 10px;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    .convert-btn {
        background-color: #4A90E2;
        color: white;
        font-size: 16px;
        font-weight: bold;
        padding: 10px;
        border-radius: 8px;
    }
    .convert-btn:hover {
        background-color: #50E3C2;
    }
    .info-box {
        background: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)
# Define Transformer Components
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, q, k, v):
        batch_size = tf.shape(q)[0]
        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)
        attn_output = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.depth, tf.float32)))
        attn_output = tf.matmul(attn_output, v)
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])
        return self.dense(tf.reshape(attn_output, (batch_size, -1, self.d_model)))

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff):
        super(TransformerEncoder, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, x):
        attn_output = self.attention(x, x, x)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)

class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff):
        super(TransformerDecoder, self).__init__()
        self.attention1 = MultiHeadAttention(d_model, num_heads)
        self.attention2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, x, enc_output):
        attn1 = self.attention1(x, x, x)
        out1 = self.layernorm1(x + attn1)
        attn2 = self.attention2(out1, enc_output, enc_output)
        out2 = self.layernorm2(out1 + attn2)
        ffn_output = self.ffn(out2)
        return self.layernorm3(out2 + ffn_output)

@tf.keras.utils.register_keras_serializable()
class Transformer(tf.keras.Model):
    def __init__(self, vocab_size, d_model, num_heads, dff, max_len):
        super(Transformer, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, input_length=max_len)
        self.encoder = TransformerEncoder(d_model, num_heads, dff)
        self.decoder = TransformerDecoder(d_model, num_heads, dff)
        self.final_layer = tf.keras.layers.Dense(vocab_size, activation='softmax')
    
    def call(self, inputs):
        enc_output = self.encoder(self.embedding(inputs))
        dec_output = self.decoder(self.embedding(inputs), enc_output)
        return self.final_layer(dec_output)

# Load Tokenizers
with open('pseudocode_to_cpp_tokenizer.pkl', 'rb') as f:
    pseudocode_tokenizer = pickle.load(f)
with open('pseudo_to_cpp_tokenizer.pkl', 'rb') as f:
    cpp_tokenizer = pickle.load(f)

# Load Models
pseudo_to_cpp_model = tf.keras.models.load_model('psuedo_to_cpp_transformer_model.keras', custom_objects={"Transformer": Transformer}, compile=False)
cpp_to_pseudo_model = tf.keras.models.load_model('transformer_model.keras', custom_objects={"Transformer": Transformer}, compile=False)
def generate_output(input_text, tokenizer_input, tokenizer_output, model, max_len=150):
    input_seq = tokenizer_input.texts_to_sequences(["<sos> " + input_text + " <eos>"])
    input_seq = pad_sequences(input_seq, maxlen=max_len, padding='post')
    pred_seq = model.predict(input_seq)
    pred_indices = np.argmax(pred_seq, axis=-1)[0]
    output_tokens = [tokenizer_output.index_word.get(idx, '') for idx in pred_indices if idx > 0]
    return ' '.join(output_tokens).replace('<sos>', '').replace('<eos>', '').strip()

st.title("🔄 Pseudocode <> C++ Converter")
option = st.radio("Choose Conversion Type:", ("Pseudocode ➝ C++", "C++ ➝ Pseudocode"))
user_input = st.text_area("Enter Code:")
if st.button("Convert"):
    if option == "Pseudocode ➝ C++":
        output = generate_output(user_input, pseudocode_tokenizer, cpp_tokenizer, pseudo_to_cpp_model)
    else:
        output = generate_output(user_input, cpp_tokenizer, pseudocode_tokenizer, cpp_to_pseudo_model)
    st.text_area("Converted Code:", output, height=200)
    st.download_button("Download Output", output, file_name="converted_code.txt")


# Footer
st.markdown("<br><hr><p style='text-align: center;'>🚀 Built with ❤️ by Rohit Kumar</p>", unsafe_allow_html=True)
