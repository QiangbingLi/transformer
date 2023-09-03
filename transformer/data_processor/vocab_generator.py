from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
from . import RESERVED_TOKENS, MAX_VOCAB_SIZE


# Here mostly use the default parameters
bert_tokenizer_params=dict(lower_case=True)

bert_vocab_args = dict(
    # The target vocabulary size
    vocab_size = MAX_VOCAB_SIZE,
    # Reserved tokens that must be included in the vocabulary
    reserved_tokens = RESERVED_TOKENS,
    # Arguments for `text.BertTokenizer`
    bert_tokenizer_params=bert_tokenizer_params,
    # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
    learn_params={},
)


def write_vocab_file(filepath, vocab):
  with open(filepath, 'w') as f:
    for token in vocab:
      print(token, file=f)


def generate_vocabulary(dataset, file_path):
    """
    This takes about 2 minutes to run.

    Generate vocabulary from dataset and write to file.

    """
    vocab = bert_vocab.bert_vocab_from_dataset(
      dataset.batch(1000).prefetch(2),
      **bert_vocab_args
    )
    write_vocab_file(file_path, vocab)
    return vocab
