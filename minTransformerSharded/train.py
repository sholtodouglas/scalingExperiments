from data import Dataloader
from model import CausalTransformer
import jax.numpy as jnp
import jax

GPTConfig = {
    'n_vocab': 66,
    'block_size': 32,
    'n_layer' : 3,
    'n_head' : 8,
    'd_model' : 768,
    'shards': 2,
    'devices': 4,
    'batch_size_per_parallel': 256,
    'ckpt_dir': 'test'}

# A downside of using the more memory efficient method of embedding sharding is that it requires equal shard size across devices
# or a 'check which device I'm on, lookup desired shard size'. For the moment - easier to just have a few empty spots for tokens.

assert GPTConfig['n_vocab'] % GPTConfig['shards'] == 0


ds = Dataloader(GPTConfig)
model = CausalTransformer(GPTConfig)


x,y = ds.next_batch() # [B,T], [B,T]

with jax.experimental.maps.mesh(*model.mesh_def):
    state = model.init(jnp.array(model.key.take(GPTConfig['shards'])), x)



from tqdm import tqdm

losses = []
with jax.experimental.maps.mesh(*model.mesh_def):
    steps = [t for t in range(0, 10000)]
    pbar = tqdm(steps)
    for t in pbar:
        x,y = ds.next_batch()
        loss, state = model.train(state, x,y)
        if t % 100 == 0:
            pbar.set_description(f"Loss: {loss.mean()}")
            losses.append(loss.mean())

# Non auto-regressive sampling (works faster so you can see if it broadly making sense after 15 minutes)
with jax.experimental.maps.mesh(*model.mesh_def):
    x,y = ds.next_batch()
    y_pred = model.forward(state['params'], x)
    y_pred_logit = jnp.argmax(y_pred, -1)
    
    for i in range(0,100):
        print(''.join([ds.itos[c] for c in list(y_pred_logit[i])]))
        print('--------------------------')