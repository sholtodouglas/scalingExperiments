
import haiku as hk
import jax
import optax
from layers import CausalTransformerShard
from jax import value_and_grad
import os
import pickle
import numpy as np


class CausalTransformer():
    def __init__(self, config):
        super().__init__()
        
        axis_names = ('batch', 'shard')
        mesh_devices = np.array(jax.devices()).reshape((config['devices'], config['shards']))
        self.mesh_def = (mesh_devices, axis_names)

        self.config = config
        self.optimizer = optax.adam(1e-5)
        
        self.key = hk.PRNGSequence(42)
        
        self.init = jax.experimental.maps.xmap(fun=self.init_state,
                                  in_axes=(["shard", ...], # rngs
                                           ["batch", ...]), # x
                                  out_axes=["shard", ...],
                                  axis_resources={'shard': 'shard', 'batch': 'batch'})

        self.forward = jax.experimental.maps.xmap(fun=self.eval_step,
                                             in_axes=(["shard", ...], # params
                                                      ["batch", ...]), # x 
                                             out_axes=(["batch", ...]), 
                                             axis_resources={'shard': 'shard', 'batch': 'batch'})


        self.train = jax.experimental.maps.xmap(fun=self.train_step,
                                             in_axes=(["shard", ...], # state
                                                      ["batch", ...], # x
                                                      ["batch", ...]),# y
                                             out_axes=([['batch'],           # loss
                                                       ['shard',...]]), # state
                                             axis_resources={'shard': 'shard', 'batch': 'batch'})


    # Haiku pure functions for convenience 
    def eval_fn(self, x):
        model = CausalTransformerShard(self.config)
        return model(x)

    def train_fn(self, x,y):
        model = CausalTransformerShard(self.config)
        return model.loss(x,y)


    def init_state(self, key, x):
        '''
        A parallelised init function that ensures optimiser params are stored on the respective devices. 
        '''
        params = hk.transform(self.eval_fn).init(key, x)

        return {
            "params": params,
            "step": np.array(0),
            "opt_state": self.optimizer.init(params)
        }

    def eval_step(self, params, x):

        forward_fn = hk.without_apply_rng(hk.transform(self.eval_fn))
        out = forward_fn.apply(params, x)
        return out

    def train_step(self, state, x,y):

        l_fn = hk.without_apply_rng(hk.transform(self.train_fn))
        loss, grads = value_and_grad(l_fn.apply)(state['params'], x,y)
        grads = jax.lax.pmean(grads, "batch")
        updates, new_opt_state = self.optimizer.update(grads, state['opt_state'], state['params'])

        return loss, {
            "params": optax.apply_updates(state['params'], updates),
            "step": state['step'] + 1,
            "opt_state": new_opt_state
        }
    
    def save(self, state):
        os.makedirs(self.config['ckpt_dir'], exist_ok=True)
        with open(os.path.join(self.config['ckpt_dir'], "arrays.npy"), "wb") as f:
            for x in jax.tree_leaves(state):
                np.save(f, x, allow_pickle=False)

        tree_struct = jax.tree_map(lambda t: 0, state)
        with open(os.path.join(self.config['ckpt_dir'], "tree.pkl"), "wb") as f:
            pickle.dump(tree_struct, f)

    def restore(self):
        '''
        Usage: set state = model.restore() after initialising the model. 
        '''
        with open(os.path.join(self.config['ckpt_dir'], "tree.pkl"), "rb") as f:
            tree_struct = pickle.load(f)

        leaves, treedef = jax.tree_flatten(tree_struct)
        with open(os.path.join(self.config['ckpt_dir'], "arrays.npy"), "rb") as f:
            flat_state = [np.load(f) for _ in leaves]

        return jax.tree_unflatten(treedef, flat_state)