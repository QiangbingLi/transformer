#%%
import logging
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow as tf

import tensorflow_text

from transformer.transformer import Transformer
from transformer.utils.custom_schedule import CustomSchedule
from transformer.utils.masked_metrics import masked_accuracy, masked_loss
from transformer.translator import Translator
from transformer.export import ExportTranslator

#%% Download the dataset  
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                               with_info=True,
                               as_supervised=True)

train_examples, val_examples = examples['train'], examples['validation']

#%% Set up the tokenizer
#   The model has been already gernerated by running the 
#   data_handling script 
model_name = 'models/ted_hrlr_translate_pt_en_converter'
tokenizers = tf.saved_model.load(model_name)

#%%  list the model interface
[item for item in dir(tokenizers.en) if not item.startswith('_')]

#%% The distribution of tokens per example in the dataset
lengths = []  # count both languages

for pt_examples, en_examples in train_examples.batch(1024):
  pt_tokens = tokenizers.pt.tokenize(pt_examples)
  lengths.append(pt_tokens.row_lengths())

  en_tokens = tokenizers.en.tokenize(en_examples)
  lengths.append(en_tokens.row_lengths())
  print('.', end='', flush=True)

#%% plot it
all_lengths = np.concatenate(lengths)

plt.hist(all_lengths, np.linspace(0, 500, 101))
plt.ylim(plt.ylim())
max_length = max(all_lengths)
plt.plot([max_length, max_length], plt.ylim())
plt.title(f'Maximum tokens per example: {max_length}');

#%% converts dataset to a format suitable for training 
MAX_TOKENS=128
def prepare_batch(pt, en):
    pt = tokenizers.pt.tokenize(pt)      # Output is ragged.
    pt = pt[:, :MAX_TOKENS]    # Trim to MAX_TOKENS.
    pt = pt.to_tensor()  # Convert to 0-padded dense Tensor

    en = tokenizers.en.tokenize(en)
    en = en[:, :(MAX_TOKENS+1)]
    en_inputs = en[:, :-1].to_tensor()  # Drop the [END] tokens
    en_labels = en[:, 1:].to_tensor()   # Drop the [START] tokens

    return (pt, en_inputs), en_labels

# %%
BUFFER_SIZE = 20000
#BATCH_SIZE = 64
BATCH_SIZE = 16    #  decreased to fix OOM on GPU 
def make_batches(ds):
  return (
      ds
      .shuffle(BUFFER_SIZE)
      .batch(BATCH_SIZE)
      .map(prepare_batch, tf.data.AUTOTUNE)
      .prefetch(buffer_size=tf.data.AUTOTUNE))

#%% Create training and validation set batches.
train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)

# %% get th first of batches
for (pt, en), en_labels in train_batches.take(1):
  break

print(pt.shape)
print(en.shape)
print(en_labels.shape)


# %% Instantiate transformer
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1

transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
    target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
    dropout_rate=dropout_rate)

# %%
output = transformer((pt, en))

print(en.shape)
print(pt.shape)
print(output.shape)

#%%
attn_scores = transformer.decoder.dec_layers[-1].last_attn_scores
print(attn_scores.shape)  # (batch, heads, target_seq, input_seq)

# %%
transformer.summary()

# %% customize optimizer
learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
# %%
plt.plot(learning_rate(tf.range(40000, dtype=tf.float32)))
plt.ylabel('Learning Rate')
plt.xlabel('Train Step')

# %%
transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy])

# %%
try:
  transformer.fit(train_batches,
                  epochs=20,
                  initial_epoch=0,
                  validation_data=val_batches)
except:
  pass

# %%
translator = Translator(tokenizers, transformer)

#%%
translator = ExportTranslator(translator)

#%%
tf.saved_model.save(translator, export_dir='models/translator')

# %%
