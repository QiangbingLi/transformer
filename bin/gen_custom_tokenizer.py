#%%
import pathlib
import tensorflow_datasets as tfds
import tensorflow as tf

from transformer.data_processor.custom_tokenizer import CustomTokenizer 
from transformer.data_processor.vocab_generator import generate_vocabulary
from transformer.data_processor import RESERVED_TOKENS

tf.get_logger().setLevel('ERROR')
pwd = pathlib.Path.cwd()

#%% Download the dataset
# Fetch the Portuguese/English translation dataset from tfds:
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

# %%
for pt, en in train_examples.take(1):
  print("Portuguese: ", pt.numpy().decode('utf-8'))
  print("English:   ", en.numpy().decode('utf-8'))

# %%
train_en = train_examples.map(lambda pt, en: en)
train_pt = train_examples.map(lambda pt, en: pt)

#%%
pt_vocab_file = 'data/pt_vocab.txt'
en_vocab_file = 'data/en_vocab.txt'

#%%  Generate the vocabulary. This takes about 2 minutes.
%%time
pt_vocab = generate_vocabulary(train_pt, pt_vocab_file)

# %%  Here are some slices of the resulting vocabulary.
print(pt_vocab[:10])
print(pt_vocab[100:110])
print(pt_vocab[1000:1010])
print(pt_vocab[-10:])

# %% now english
%%time
en_vocab = generate_vocabulary(train_en, en_vocab_file)

#%%
print(en_vocab[:10])
print(en_vocab[100:110])
print(en_vocab[1000:1010])
print(en_vocab[-10:])

#%% too much data to unpack
# pt_examples, en_examples = tuple(zip(*train_examples)) 

#%% Take a batch of 3 examples from the data.
# Increase the number inside the take method to get more elements 
# from the data. Insert -1 to get all elemnts.
for pt_examples, en_examples in train_examples.batch(3).take(1):
  for ex in en_examples:
    print(ex.numpy())

# %% Export
#    Build a CustomTokenizer for each language:
tokenizers = tf.Module()
tokenizers.pt = CustomTokenizer(RESERVED_TOKENS, pt_vocab_file)
tokenizers.en = CustomTokenizer(RESERVED_TOKENS, en_vocab_file)

# %%
model_name = 'models/ted_hrlr_translate_pt_en_converter'

#%% Export the tokenizers as a saved_model
tf.saved_model.save(tokenizers, model_name)

# %% Reload the saved_model and list the methods, which are the same for both languages
reloaded_tokenizers = tf.saved_model.load(model_name)
[item for item in dir(reloaded_tokenizers.en) if not item.startswith('_')]

#%% test the methods:
reloaded_tokenizers.en.get_vocab_size().numpy()

#%%
en_example_tokens = reloaded_tokenizers.en.tokenize(en_examples)
en_example_tokens.numpy()

#%%
tokens = reloaded_tokenizers.en.tokenize(['Hello TensorFlow!'])
tokens.numpy()

# %%
text_tokens = reloaded_tokenizers.en.lookup(tokens)
text_tokens

#%%
round_trip = reloaded_tokenizers.en.detokenize(tokens)
print(round_trip.numpy()[0].decode('utf-8'))

