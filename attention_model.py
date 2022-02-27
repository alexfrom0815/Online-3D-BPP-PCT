import torch
from torch import nn
import math
from typing import NamedTuple
from graph_encoder import GraphAttentionEncoder
from distributions import FixedCategorical
from tools import observation_decode_leaf_node, init

class AttentionModelFixed(NamedTuple):
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],
                glimpse_val=self.glimpse_val[:, key],
                logit_key=self.logit_key[key]
            )
        return super(AttentionModelFixed, self).__getitem__(key)

class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=False,
                 mask_logits=False,
                 n_heads=1,
                 internal_node_holder = None,
                 internal_node_length = None,
                 leaf_node_holder = None,
                 ):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0

        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.n_heads = n_heads
        self.internal_node_holder = internal_node_holder
        self.internal_node_length = internal_node_length
        self.next_holder = 1
        self.leaf_node_holder = leaf_node_holder

        graph_size = internal_node_holder + leaf_node_holder + self.next_holder

        activate, ini = nn.LeakyReLU, 'leaky_relu'
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain(ini))

        self.init_internal_node_embed = nn.Sequential(
            init_(nn.Linear(self.internal_node_length, 32)),
            activate(),
            init_(nn.Linear(32, embedding_dim)))

        self.init_leaf_node_embed  = nn.Sequential(
            init_(nn.Linear(8, 32)),
            activate(),
            init_(nn.Linear(32, embedding_dim)))

        self.init_next_embed = nn.Sequential(
            init_(nn.Linear(6, 32)),
            activate(),
            init_(nn.Linear(32, embedding_dim)))

        # Graph attention model
        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            graph_size = graph_size,
        )

        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0

    def forward(self, input, deterministic = False, evaluate_action = False, normFactor = 1, evaluate = False):

        internal_nodes, leaf_nodes, next_item, invalid_leaf_nodes, full_mask = observation_decode_leaf_node(input,
                                                                                                            self.internal_node_holder,
                                                                                                            self.internal_node_length,
                                                                                                            self.leaf_node_holder)
        leaf_node_mask = 1 - invalid_leaf_nodes
        valid_length = full_mask.sum(1)
        full_mask = 1 - full_mask

        batch_size = input.size(0)
        graph_size = input.size(1)
        internal_nodes_size = internal_nodes.size(1)
        leaf_node_size = leaf_nodes.size(1)
        next_size = next_item.size(1)

        internal_inputs = internal_nodes.contiguous().view(batch_size * internal_nodes_size, self.internal_node_length)*normFactor
        leaf_inputs = leaf_nodes.contiguous().view(batch_size * leaf_node_size, 8)*normFactor
        current_inputs = next_item.contiguous().view(batch_size * next_size, 6)*normFactor

        # We use three independent node-wise Multi-Layer Perceptron (MLP) blocks to project these raw space configuration nodes
        # presented by descriptors in different formats into the homogeneous node features.
        internal_embedded_inputs = self.init_internal_node_embed(internal_inputs).reshape((batch_size, -1, self.embedding_dim))
        leaf_embedded_inputs = self.init_leaf_node_embed(leaf_inputs).reshape((batch_size, -1, self.embedding_dim))
        next_embedded_inputs = self.init_next_embed(current_inputs.squeeze()).reshape(batch_size, -1, self.embedding_dim)
        init_embedding = torch.cat((internal_embedded_inputs, leaf_embedded_inputs, next_embedded_inputs), dim=1).view(batch_size * graph_size, self.embedding_dim)

        # transform init_embedding into high-level node features.
        embeddings, _ = self.embedder(init_embedding, mask = full_mask, evaluate = evaluate)
        embedding_shape = (batch_size, graph_size, embeddings.shape[-1])
        
        # Decide the leaf node indices for accommodating the current item
        log_p, action_log_prob, pointers, dist_entropy, dist, hidden = self._inner(embeddings,
                                                          deterministic=deterministic,
                                                          evaluate_action=evaluate_action,
                                                          shape = embedding_shape,
                                                          mask = leaf_node_mask,
                                                          full_mask = full_mask,
                                                          valid_length = valid_length)
        return action_log_prob, pointers, dist_entropy, hidden, dist

    def _inner(self, embeddings, mask = None, deterministic = False, evaluate_action = False, shape = None, full_mask = None, valid_length =None): # 元素齐了
        # The aggregation of global feature
        fixed = self._precompute(embeddings, shape = shape, full_mask = full_mask, valid_length = valid_length)
        # Calculate probabilities of selecting leaf nodes
        log_p, mask = self._get_log_p(fixed, mask)

        # The leaf node which is not feasible will be masked in a soft way.
        if deterministic:
            masked_outs = log_p * (1 - mask)
            if torch.sum(masked_outs) == 0:
                masked_outs += 1e-20
        else:
            masked_outs = log_p * (1 - mask) + 1e-20
        log_p = torch.div(masked_outs, torch.sum(masked_outs, dim=1).unsqueeze(1))

        dist = FixedCategorical(probs=log_p)
        dist_entropy = dist.entropy()

        # Get maximum probabilities and indices
        if deterministic:
            # We take the argmax of the policy for the test.
            selected = dist.mode()
        else:
            # The action at is sampled from the distribution for training
            selected = dist.sample()

        if not evaluate_action:
            action_log_probs = dist.log_probs(selected)
        else:
            action_log_probs = None

        # Collected lists, return Tensor
        return log_p, action_log_probs, selected, dist_entropy, dist, fixed.context_node_projected

    def _precompute(self, embeddings, num_steps=1, shape = None, full_mask = None, valid_length = None):
        # The aggregation of global feature, only happens on the eligible nodes.
        transEmbedding = embeddings.view(shape)
        full_mask = full_mask.view(shape[0], shape[1],1).expand(shape).bool()
        transEmbedding[full_mask]  = 0
        graph_embed = transEmbedding.view(shape).sum(1)
        transEmbedding = transEmbedding.view(embeddings.shape)

        graph_embed = graph_embed / valid_length.reshape((-1,1))
        fixed_context = self.project_fixed_context(graph_embed)

        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(transEmbedding).view((shape[0], 1, shape[1],-1)).chunk(3, dim=-1)

        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(transEmbedding, fixed_context, *fixed_attention_node_data)

    def _get_log_p(self, fixed, mask = None, normalize=True):
        # Compute query = context node embedding
        query = fixed.context_node_projected[:, None, :]

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed)

        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)
        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)
        assert not torch.isnan(log_p).any()
        return log_p.exp(), mask

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        logits = compatibility.reshape([-1,1,compatibility.shape[-1]])

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping

        logits = logits[:, 0, self.internal_node_holder: self.internal_node_holder + self.leaf_node_holder]
        if self.mask_logits:
            logits[mask.bool()] = -math.inf

        return logits, None

    def _get_attention_node_data(self, fixed):
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
                .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
                .permute(3, 0, 1, 2, 4)
        )