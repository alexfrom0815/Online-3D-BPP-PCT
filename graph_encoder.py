import torch
from torch import nn
import math

class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return {'data':input['data'] + self.module(input), 'mask': input['mask'], 'graph_size':input['graph_size']}

class SkipConnection_Linear(nn.Module):
    def __init__(self, module):
        super(SkipConnection_Linear, self).__init__()
        self.module = module

    def forward(self, input):
        return {'data':input['data'] + self.module(input['data']), 'mask': input['mask'], 'graph_size': input['graph_size']}

class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None,
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Linear(input_dim, key_dim, bias=False)
        self.W_key = nn.Linear(input_dim, key_dim, bias=False)
        self.W_val = nn.Linear(input_dim, val_dim, bias=False)

        if embed_dim is not None:
            # self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))
            self.W_out = nn.Linear(key_dim, embed_dim)

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, data, h=None):
        """
        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        q = data['data']
        mask = data['mask']
        graph_size = data['graph_size']
        if h is None:
            h = q

        batch_size = int(q.size()[0] / graph_size)
        graph_size = graph_size
        input_dim = h.size()[-1]
        n_query = graph_size
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)
        Q = self.W_query(qflat).view(shp_q)
        K = self.W_key(hflat).view(shp)
        V = self.W_val(hflat).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        mask = mask.unsqueeze(1).repeat((1, graph_size, 1)).bool()
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            if data['evaluate']:
                compatibility[mask] = -math.inf
            else:
                compatibility[mask] = -30
        attn = torch.softmax(compatibility, dim=-1) #

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)
        out = self.W_out(heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim)).view(batch_size * n_query, self.embed_dim)
        return out

class MultiHeadAttentionLayer(nn.Sequential):
    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=128):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim,
                )
            ),
            SkipConnection_Linear(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
        )

class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            feed_forward_hidden=128,
            graph_size=None,
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None
        self.graph_size = graph_size
        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden)
            for _ in range(n_layers)
        ))

    def forward(self, x, mask=None, limited=False, evaluate = False):

        # Batch multiply to get initial embeddings of nodes
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x

        data = {'data':h, 'mask': mask, 'graph_size': self.graph_size, 'evaluate': evaluate}
        h = self.layers(data)['data']
        return (h, h.view(int(h.size()[0] / self.graph_size), self.graph_size, -1).mean(dim=1),)

