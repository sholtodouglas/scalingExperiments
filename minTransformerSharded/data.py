import os
import tensorflow as tf

class Dataloader():
    '''
    A super simple dataloader for a tiny dataset. Better implementations would 
    - Pre-process the data into tf-records, and stream this from 
        a GCP bucket. 
    - Eliminate unncessary copies (e.g. as_numpy_iterator)
    '''
    def __init__(self, config):
        super().__init__()
        
        if not os.path.exists('input.txt'):
            os.system('wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')
            
        text = open('input.txt', 'r').read() 
        self.vocab = sorted(list(set(text)))
        self.vocab_len = len(self.vocab)
        self.stoi = { ch:i for i,ch in enumerate(self.vocab) }
        self.itos = { i:ch for i,ch in enumerate(self.vocab) }
        tokens = [self.stoi[c] for c in text]
        d = tf.data.Dataset.from_tensor_slices(tokens)
        d = d.batch(config['block_size']+1, drop_remainder=True) # +1 because [:-1] will be x, and [1:] will be y
        self.d = iter(d.batch(config['batch_size_per_parallel']*config['devices'], drop_remainder=True).repeat().as_numpy_iterator())
        
    def next_batch(self):
        b = self.d.next()
        return b[:, :-1], b[:, 1:] # x, y