#%%
import tensorflow as tf
from transformer.position import PositionalEmbedding
from transformer.attentions import CrossAttention, GlobalSelfAttention, CausalSelfAttention
from transformer.feed_forward import FeedForward
from transformer.encoder import EncoderLayer, Encoder
from transformer.decoder import DecoderLayer, Decoder
from transformer.transformer import Transformer

#%%
vocab_con = 5000 # vocabulary size of context language
vocab_tar = 7000  # vocabulary size of target language
batch = 64   # batch size
s = 55       # context size
t = 66       # target size  
units = 512  # units of embedding layer

#%% randomize tokens for context and target
token_con = tf.random.uniform(
    (batch, s),
    minval=0,
    maxval=vocab_con,
    dtype=tf.dtypes.int32,
)

token_tar = tf.random.uniform(
    (batch, t),
    minval=0,
    maxval=vocab_tar,
    dtype=tf.dtypes.int32,
)


# %% Embedding
embed_con = PositionalEmbedding(vocab_size=vocab_con, d_model=units)
embed_tar = PositionalEmbedding(vocab_size=vocab_tar, d_model=units)

con_emb = embed_con(token_con)
tar_emb = embed_tar(token_tar)

print(con_emb.shape)
print(tar_emb.shape)

#%%
con_emb._keras_mask


# %%
sample_ca = CrossAttention(num_heads=2, key_dim=units)
print(sample_ca(tar_emb, con_emb).shape)
print(sample_ca.last_attn_scores[:, 0, :, :].shape)

# %%
sample_gsa = GlobalSelfAttention(num_heads=2, key_dim=units)
print(sample_gsa(con_emb).shape)
print(sample_gsa(tar_emb).shape)

# %%
sample_csa = CausalSelfAttention(num_heads=2, key_dim=units)
print(sample_csa(con_emb).shape)
print(sample_csa(tar_emb).shape)

#%% The output for early sequence elements doesn't depend on later 
#   elements, so it shouldn't matter if you trim elements before or 
#   after applying the layer:
out1 = sample_csa(embed_tar(token_tar[:, :3])) 
out2 = sample_csa(embed_tar(token_tar))[:, :3]

tf.reduce_max(abs(out1 - out2)).numpy()

# %%
sample_ffn = FeedForward(units, 2048)
print(con_emb.shape)
print(sample_ffn(con_emb).shape)


# %%
sample_encoder_layer = EncoderLayer(d_model=units, num_heads=8, dff=2048)
print(con_emb.shape)
print(sample_encoder_layer(con_emb).shape)

# %%
sample_encoder = Encoder(num_layers=4,
                         d_model=units,
                         num_heads=8,
                         dff=2048,
                         vocab_size=vocab_con)

sample_encoder_output = sample_encoder(token_con, training=False)
print(token_con.shape)
print(sample_encoder_output.shape)  # Shape `(batch_size, input_seq_len, d_model)`.

# %%
sample_decoder_layer = DecoderLayer(d_model=units, num_heads=8, dff=2048)
sample_decoder_layer_output = sample_decoder_layer(
    x=tar_emb, context=con_emb)

print(con_emb.shape)
print(tar_emb.shape)
print(sample_decoder_layer_output.shape)  # `(batch_size, seq_len, d_model)`

# %%
sample_decoder = Decoder(num_layers=4,
                         d_model=units,
                         num_heads=8,
                         dff=2048,
                         vocab_size=vocab_tar)

output = sample_decoder(
    x=token_tar,
    context=con_emb)

# Print the shapes.
print(token_tar.shape)
print(con_emb.shape)
print(output.shape)


# %%
num_layers = 4
d_model = units
dff = 512
num_heads = 8
dropout_rate = 0.1

transformer = Transformer(
    num_layers = num_layers,
    d_model = d_model,
    num_heads = num_heads,
    dff = dff,
    input_vocab_size = vocab_con,
    target_vocab_size = vocab_tar,
    dropout_rate = dropout_rate)


# %%
output = transformer((token_con, token_tar))

print(token_con.shape)
print(token_tar.shape)
print(output.shape)

#%%
attn_scores = transformer.decoder.dec_layers[-1].last_attn_scores
print(attn_scores.shape)  # (batch, heads, target_seq, input_seq)

# %%
transformer.summary()