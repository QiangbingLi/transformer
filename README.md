# A basic transformer model

### create the virtual environment
    $ virtualenv .venv
    $ source .venv/bin/activate

### install dependencies
    $ pip install requirement.txt

### install to virtual env in editable mode
    $ pip install -e .

### run it
    $ cd bin

    $ python gen_custom_tokenizer.py  
      # To test the data processor, which generates the custom tokenizer model. 
      # Can be skipped if the custom tokenizer is downloaded from 
      # google storage in training.

    $ python train.py    
      # train, export and save the transformer model

    $ python inference.py  
      # translation examples

[tutorial](https://www.tensorflow.org/text/tutorials/transformer)