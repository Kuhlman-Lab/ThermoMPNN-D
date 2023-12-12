import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

from thermompnn.protein_mpnn_utils import ProteinMPNN, PositionWiseFeedForward


def get_protein_mpnn(cfg, version='v_48_020.pt'):
    """Loading Pre-trained ProteinMPNN model for structure embeddings"""
    hidden_dim = 128
    num_layers = 3

    if 'version' in cfg.model:
        version = cfg.model.version
    else:
        version = 'v_48_020.pt'
    
    if 'IPMP' in version:
        use_IPMP = True
    else:
        use_IPMP = False

    model_weight_dir = os.path.join(cfg.platform.thermompnn_dir, 'vanilla_model_weights')
    checkpoint_path = os.path.join(model_weight_dir, version)
    print('Loading model %s', checkpoint_path)
    # checkpoint_path = "vanilla_model_weights/v_48_020.pt"
    checkpoint = torch.load(checkpoint_path, map_location='cpu') 
    model = ProteinMPNN(use_ipmp=use_IPMP, num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, 
                        num_encoder_layers=num_layers, num_decoder_layers=num_layers, k_neighbors=checkpoint['num_edges'], augment_eps=0.0)
    if cfg.model.load_pretrained:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if cfg.model.freeze_weights:
        model.eval()
        # freeze these weights for transfer learning
        for param in model.parameters():
            param.requires_grad = False

    return model


class LightAttention(nn.Module):
    """Source:
    Hannes Stark et al. 2022
    https://github.com/HannesStark/protein-localization/blob/master/models/light_attention.py
    """
    def __init__(self, embeddings_dim=1024, output_dim=11, dropout=0.25, kernel_size=9, conv_dropout: float = 0.25):
        super(LightAttention, self).__init__()

        self.feature_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                             padding=kernel_size // 2)
        self.attention_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                               padding=kernel_size // 2)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(conv_dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor
        Returns:
            LightAttention reweighted vector of the same dimensions
        """
        o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        o = self.dropout(o)  # [batch_gsize, embeddings_dim, sequence_length]
        attention = self.attention_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        o1 = o * self.softmax(attention)  # [batch_size, embeddings_dim]
        return torch.squeeze(o1, -1)


class MultHeadAttn(nn.Module):
    """Implementation of multi-head attention
    Based on https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch
    """
    def __init__(self, embed_dim=512, n_heads=8):
        super(MultHeadAttn, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.single_head_dim = int(self.embed_dim / self.n_heads) # each qkv will be of value (embed_dim / n_heads)

        # make qkv matrices
        self.query_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)  # single key matrix for all 8 keys #512x512
        self.key_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(self.single_head_dim,self.single_head_dim, bias=False)
        # final output layer
        self.out = nn.Linear(self.n_heads * self.single_head_dim, self.embed_dim) 
    
    def forward(self, query, key, value, mask=None):

        # [batch_size, embed_size, seq_length] is query/value/key size

        batch_size = key.size(0)
        seq_length = key.size(-1)
        
        # query dimension can change in decoder during inference, so we cant take general seq_length
        seq_length_query = query.size(-1)

        # reshape to fit individual attention heads 
        key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim)  #batch_size x sequence_length x n_heads x single_head_dim = (32x10x8x64)
        query = query.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim) #(32x10x8x64)
        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim) #(32x10x8x64)
       
        k = self.key_matrix(key)       # (32x10x8x64)
        q = self.query_matrix(query)
        v = self.value_matrix(value)

        q = q.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)    # (32 x 8 x 10 x 64)
        k = k.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)
        v = v.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)
              
        # computes attention; adjust key for matrix multiplication
        k_adjusted = k.transpose(-1,-2)  #(batch_size, n_heads, single_head_dim, seq_ken)  #(32 x 8 x 64 x 10)
        product = torch.matmul(q, k_adjusted)  #(32 x 8 x 10 x 64) x (32 x 8 x 64 x 10) = #(32x8x10x10)
        # fill those positions of product matrix as (-1e20) where mask positions are 0
        if mask is not None:
             product = product.masked_fill(mask == 0, float("-1e20"))

        #divising by square root of key dimension
        product = product / math.sqrt(self.single_head_dim) # / sqrt(64)

        #applying softmax
        scores = F.softmax(product, dim=-1)
        #mutiply with value matrix
        scores = torch.matmul(scores, v)  ##(32x8x 10x 10) x (32 x 8 x 10 x 64) = (32 x 8 x 10 x 64) 
        
        #concatenated output
        concat = scores.transpose(1,2).contiguous().view(batch_size, seq_length_query, self.single_head_dim*self.n_heads)  # (32x8x10x64) -> (32x10x8x64)  -> (32,10,512)
        
        output = self.out(concat) #(32,10,512) -> (32,10,512)
        return output 


class MPNNLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30):
        super(MPNNLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward_new(self, emb1, emb2):
        """ Message passing b/w two embeddings"""
        # concatenate emb1 + emb2 and emb2 + emb1

        # pass both through shared MLP to construct message

        # aggregate message with original emb1 or emb2 to get new rep

        # aggregate emb1 with emb2 to get final rep

        # Concatenate h_V_i to h_E_ij
        # h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        # h_EV = torch.cat([h_V_expand, h_E], -1)

        # # use MLP to construct message
        # h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))

        # # optional: attend to message
        # if mask_attend is not None:
        #     h_message = mask_attend.unsqueeze(-1) * h_message

        # # aggregate message?
        # dh = torch.sum(h_message, -2) / self.scale

        # h_V = self.norm1(h_V + self.dropout1(dh))

        # # Position-wise feedforward
        # dh = self.dense(h_V)
        # h_V = self.norm2(h_V + self.dropout2(dh))

        # # mask again?
        # if mask_V is not None:
        #     mask_V = mask_V.unsqueeze(-1)
        #     h_V = mask_V * h_V
        # return h_V

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        # use MLP to construct message
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))

        # optional: attend to message
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message

        # aggregate message?
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        # mask again?
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V

