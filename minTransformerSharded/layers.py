    
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class TransformerLayerShard(hk.Module):
        '''
        # A simple transformer layer shard that exists on one of the devices. 
        '''
        def __init__(self, config, name=None, init_scale=1.):
            super().__init__(name=name)
            heads = config["n_head"]
            dim = config["d_model"]
            self.shards  = config["shards"]
            
            assert dim % heads == 0 
            assert heads % self.shards == 0 
            
            self.dim_per_head = dim // heads
            self.dim_per_shard = dim // self.shards
            self.heads_per_shard = heads // self.shards
            
            # GPT-J uses a common layer norm between the mlp and the self attention, minGPT uses different layers - lets go with one for now. Much of a muchness.
            self.ln = hk.LayerNorm(-1, True, True)
            
            # key, query and value projections for all heads on this shard
            self.q = hk.Linear(self.dim_per_shard, with_bias=False)
            self.k = hk.Linear(self.dim_per_shard, with_bias=False)
            self.v = hk.Linear(self.dim_per_shard, with_bias=False)
            
            # self att output projection
            self.att_proj = hk.Linear(dim, with_bias=False, w_init=hk.initializers.TruncatedNormal(stddev=init_scale / np.sqrt(dim)))
            
            # feedforward layers
            self.dense_proj = hk.Linear(self.dim_per_shard * 4)
            self.dense_proj_out = hk.Linear(dim,
                                      w_init=hk.initializers.TruncatedNormal(stddev=init_scale / np.sqrt(dim)))
            
            
        def self_attention(self, q, k ,v, bias):
            '''
            k,q,v: [T, heads_per_shard, dim_per_head]
            '''
            T, _, _ = k.shape
            
            # No batch dimension needed in the einsum because it is abstracted away by the xmap 
            attention_logits = jnp.einsum('thd,Thd->htT', q, k) # [heads_per_shard, T,T]
            sqrt_key_size = np.sqrt(self.dim_per_head).astype(k.dtype) # [1,]
            
            attention_logits = attention_logits/sqrt_key_size # [heads_per_shard, T,T]
            
            attention_logits += bias # [B, heads_per_shard, T,T]
            
            attention_weights = jax.nn.softmax(attention_logits)  # [heads_per_shard, T,T]
            
            weighted_values = jnp.einsum('htT, Thd->thd', attention_weights, v).reshape((T, self.dim_per_shard)) # [T, dim_per_shard]
            
            return self.att_proj(weighted_values)
        
        def feed_forward(self, x):
            '''
            x: [T,embed_dim]
            '''
            dense_proj = self.dense_proj(x)
            dense_proj = jax.nn.gelu(dense_proj)
            return self.dense_proj_out(dense_proj)
            
            
        def qkv_proj(self, x): 
            '''
            x: [T, embed_dim]
            '''
            q = self.q(x).reshape(x.shape[:-1] + (self.heads_per_shard, self.dim_per_head)) # [T, heads_per_shard, dim_per_head]
            v = self.v(x).reshape(x.shape[:-1] + (self.heads_per_shard, self.dim_per_head)) # ""
            k = self.k(x).reshape(x.shape[:-1] + (self.heads_per_shard, self.dim_per_head)) # "" 
            
            return q,k,v
        
        
        def __call__(self, x):
            '''
            x: [T, embed_dim]
            '''
            # preliminaries    
#             print(x.shape)
            T,C = x.shape
            x = self.ln(x) # [T,embed_dim]
            
            # causal self attention
            q,k,v = self.qkv_proj(x)
            causal_mask = np.tril(np.ones((T,T))) # [T,T]
            bias = -1e10 * (1. - causal_mask) # [T,T]
            attn = self.self_attention(q, k ,v, bias)  # [T,embed_dim]
            
            # feedforward
            ff  = self.feed_forward(x) # [B,T,embed_dim]
            
            # block
            x = x + attn # [T,embed_dim]
            x = x + ff # [T,embed_dim]
            
            # We finally need to sum across shards to collect the information from each head into the new embedding. 
            # The idea of heads 'adding to the information stream' is a nice way of thinking about transformers from Anthropic's 
            # latest work. 
            # In the full GPT-J implementation, they've defined a custom operator which does the psum on the forward pass but
            # is the identity function on the backward pass - currently testing how necessary that is.
            return jax.lax.psum(x, "shard")

        
                
class EmbeddingShard(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        in_dim = config["n_vocab"]
        out_dim = config["d_model"]
        shards = config["shards"]

        assert in_dim % shards == 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_dim_per_shard = in_dim // shards
        self.out_dim_per_shard = out_dim // shards

        embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
        self.positional_embeddings = hk.get_parameter('pos_embs', [config["block_size"], self.out_dim_per_shard], init=embed_init)

        # notice unlike the ff transformer layer, this linear layer has the full output dimension because we are partitioning across vocab. 
        self.proj = hk.Linear(self.out_dim, w_init=hk.initializers.TruncatedNormal(stddev=1 / np.sqrt(in_dim)))

    def __call__(self, x, dtype=jnp.bfloat16):
        
        # work out which shard we are on, and the start token index
        shard_start_index = jax.lax.axis_index('shard') * self.in_dim_per_shard
        
        # subtract the shard_start_index from the input indices. This means anything below it will be zero-d (as it will be a negative number)
        # at the same time, anything above 'in_dim_per_shard' will also be zero-d. This means that each shard gets a window of in_dim_per_shard indices
        # which it will expand to a one-hot representation - saving lots of space!
        input_onehot = jax.nn.one_hot(x - shard_start_index, self.in_dim_per_shard)
        proj_out = self.proj(input_onehot)
        # sum across shards to create a full embedding
        proj_out = jax.lax.psum(proj_out, "shard")
        # gets all of the positional embeddings as split across each shard (column wise split of positional embeddings)
        all_pos_embed = jax.lax.all_gather(self.positional_embeddings, 'shard')
        # flattens them, so now there are identical, complete positional embeddings on each device
        all_pos_embed = hk.Flatten()(jnp.transpose(all_pos_embed, (1, 0, 2)))

        proj_out += all_pos_embed[:proj_out.shape[0]] # only do the embeddings up to the length of the input sequence, to allow for variable input size

        return proj_out


class ProjectionShard(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        out_dim = config["n_vocab"]
        shards = config["shards"]

        assert out_dim % shards == 0

        self.dim = out_dim
        self.dim_per_shard = out_dim // shards

        self.norm = hk.LayerNorm(-1, True, True)

        self.proj = hk.Linear(self.dim_per_shard)

    def __call__(self, x):
        x = self.norm(x)
        proj = self.proj(x)

        all_proj = jax.lax.all_gather(proj, 'shard')

        return hk.Flatten()(jnp.transpose(all_proj, (1, 0, 2)))

    def loss(self, x, targets, z_loss=1):
        
        '''
        x: [T, dim_per_shard]
        targets: [T]
        '''
        x = self.norm(x)
        # calculate logits on a per shard basis
        logits = self.proj(x) # [T, dim_per_shard]
        # get the max of logits per dimension across the shards. Use this to prevent both under and overflow by subtracting it from every logit.
        # For an explaination on why you need this - see the opening pages of chapter 4 'Numerical computation' in goodfellow's deep learning book. 
        global_max = jax.lax.pmax(jax.lax.stop_gradient(logits.max(-1, keepdims=True)), "shard")
        logits -= jax.lax.stop_gradient(global_max) # [T, dim_per_shard]
        
        # As we are computing the output vocab matrix in a sharded fashion, only get the targets corresponding to that shard
        # using the same trick as used in the embedding matrix.
        shard_start_index = jax.lax.axis_index('shard') * self.dim_per_shard
        gt_onehot = jax.nn.one_hot(targets - shard_start_index, self.dim_per_shard) # [T, dim_per_shard]
        # this is a point multiplication, so it zeros out anything which isn't a 1 in the one-hot representation. 
        # then sums along the embedding axis. See above code snippet for explaination for the next few lines.
        predicted_logits = jnp.sum(jnp.multiply(gt_onehot, logits), axis=-1) # [T]
        # subtract the summed logit from the summed 'predicted_logit'
        # Any entry but the correct one is 0 in 'predicted logit' - and due to the max used for stabilisation
        # the entry of the highest index will be 0. Therefore, the subtraction of the two will draw the highest index to the correct one.
        # By only working with sums when we are using the sharded version we minimise communication.
        predicted_logits = jax.lax.psum(predicted_logits, 'shard') 
        exp_logits = jnp.exp(logits)
        sum_exp_logits = exp_logits.sum(axis=-1)
        sum_exp_logits = jax.lax.psum(sum_exp_logits, 'shard')
        loss = jnp.log(sum_exp_logits) - predicted_logits
        # An additional loss which keeps the logits small - avoiding roundoff errors in bfloat16 (according to the mesh tensorflow repo).
        loss += (1e-4 * jnp.square(jnp.log(sum_exp_logits)) * z_loss).mean()

        # Due to the above, it is easy to correctly predict accuracy. 
        correct = (0.0 == predicted_logits)
        return loss.sum(), correct
    

class CausalTransformerShard(hk.Module):
    def __init__(self, config):
        super().__init__()
        heads = config["n_head"]
        shards = config["shards"]
        layer_count = config["n_layer"]

        self.layers = []
        self.heads = heads

        self.heads_per_shard = heads // shards

        self.embed = EmbeddingShard(config)

        init_scale = 2. / layer_count

        for i in range(layer_count):
            self.layers.append(TransformerLayerShard(config, name=f"layer_{i}", init_scale=init_scale))

        self.proj = ProjectionShard(config)
        
        
    def trunk(self, tokens):
        x = self.embed(tokens)

        for l in self.layers:
            x = x + l(x)
        return x
    
    def loss(self, tokens, targets):
        x = self.trunk(tokens)
        l, acc = self.proj.loss(x, targets)
        return l

    def __call__(self, tokens):
        
        x = self.trunk(tokens)
        return self.proj(x)
    
    