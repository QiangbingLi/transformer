#%%
import tensorflow as tf
import tensorflow_text # without this import the model 
                       # loading will not work

#%%
translate = tf.saved_model.load('models/translator')

#%%
def print_translation(sentence, tokens, ground_truth):
  print()
  print(f'{"Input:":15s}: {sentence}')
  print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
  print(f'{"Ground truth":15s}: {ground_truth}')
  print()

#%%
sentence = 'este é o primeiro livro que eu fiz.'
ground_truth = "this is the first book i've ever done."

translated_text = translate(sentence)
print_translation(sentence, translated_text, ground_truth)

# %%
sentence = 'os meus vizinhos ouviram sobre esta ideia.'
ground_truth = 'and my neighboring homes heard about this idea .'

translated_text  = translate(sentence)
print_translation(sentence, translated_text, ground_truth)

# %%
sentence = 'vou então muito rapidamente partilhar convosco algumas histórias de algumas coisas mágicas que aconteceram.'
ground_truth = "so i'll just share with you some stories very quickly of some magical things that have happened."

translated_text = translate(sentence)
print_translation(sentence, translated_text, ground_truth)

# %%
sentence = 'Terei que extrair 7 dentes de acordo com os conselhos do dentista.'
ground_truth = "I will have to extract 7 teeth according to the dentist's advice."

translated_text = translate(sentence)
print_translation(sentence, translated_text, ground_truth)
# %%
