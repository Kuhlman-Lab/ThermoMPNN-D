import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import numpy as np

from thermompnn.protein_mpnn_utils import ProteinMPNN
from protein_mpnn_utils import gather_nodes, gather_edges, DecLayer, cat_neighbors_nodes, PositionWiseFeedForward


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


class SideChainPositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, period_range=[2,1000], max_relative_feature=32, af2_relpos=False):
        super(SideChainPositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.period_range = period_range
        self.max_relative_feature = max_relative_feature 
        self.af2_relpos = af2_relpos
        
    def _transformer_encoding(self, E_idx):
        # i-j
        N_nodes = E_idx.size(1)
        ii = torch.arange(N_nodes, dtype=torch.float32, device=E_idx.device).view((1, -1, 1))
        d = (E_idx.float() - ii).unsqueeze(-1)
        
        # Original Transformer frequencies
        frequency = torch.exp(
            torch.arange(0, self.num_embeddings, 2, dtype=torch.float32, device=E_idx.device)
            * -(np.log(10000.0) / self.num_embeddings)
        )
        
        angles = d * frequency.view((1,1,1,-1))
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        
        return E
    
    def _af2_encoding(self, E_idx, residue_index=None):
        # i-j
        if residue_index is not None:
            offset = residue_index[..., None] - residue_index[..., None, :]
            offset = torch.gather(offset, -1, E_idx)
        else:
            N_nodes = E_idx.size(1)
            ii = torch.arange(N_nodes, dtype=torch.float32, device=E_idx.device).view((1, -1, 1))
            offset = (E_idx.float() - ii)
        
        relpos = torch.clip(offset.long() + self.max_relative_feature, 0, 2 * self.max_relative_feature)
        relpos = F.one_hot(relpos, 2 * self.max_relative_feature + 1)
        
        return relpos

    def forward(self, E_idx, residue_index=None):

        if self.af2_relpos:
            E = self._af2_encoding(E_idx, residue_index)
        else:
            E = self._transformer_encoding(E_idx)

        return E
    

class SideChainProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
                 num_rbf=16, top_k=30, augment_eps=0., num_chain_embeddings=16, action_centers='none'):
        """ Extract protein features """
        super(SideChainProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.action_centers = action_centers

        self.embeddings = SideChainPositionalEncodings(num_positional_embeddings)
        if action_centers != 'none':
            edge_in = num_positional_embeddings + num_rbf * (5 ** 2)
        else:
            edge_in = num_positional_embeddings + num_rbf * (14 ** 2)
        print('edge in:', edge_in)

        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps=1E-6):
        """ Pairwise euclidean distances """
        # Convolutional network on NCHW
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

        # Identify k nearest neighbors (including self)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + 2 * (1. - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(D_adjust, min(self.top_k, X.shape[-2]), dim=-1, largest=False)
        mask_neighbors = gather_edges(mask_2D.unsqueeze(-1), E_idx)

        return D_neighbors, E_idx, mask_neighbors

    def _rbf(self, D):
        # Distance radial basis function
        D_min, D_max, D_count = 0., 20., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)

        return RBF

    def _get_rbf(self, A, B, E_idx, A_mask, B_mask):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]

        # make pairwise atom mask
        combo = torch.unsqueeze(torch.unsqueeze(A_mask, 2) * torch.unsqueeze(B_mask, 1), -1) # [B, L, L, 1]

        # gather K neighbors out of overall mask        
        combo = torch.unsqueeze(gather_edges(combo, E_idx)[..., 0], -1) # [B, L, K, 1]
        
        # mask RBFs directly using paired atom mask
        # RBF_A_B = self._rbf(D_A_B_neighbors)
        RBF_A_B = self._rbf(D_A_B_neighbors) * combo.expand(combo.shape[0], combo.shape[1], combo.shape[2], self.num_rbf) # [B, L, K, N_RBF]
        return RBF_A_B

    def _impute_CB(self, N, CA, C):
        b = CA - N
        c = C - CA
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + CA
        return Cb

    def _atomic_distances(self, X, E_idx, atom_mask):
        RBF_all = []
        
        if self.action_centers != 'none':
            X, atom_mask = self._action_centers(X, atom_mask)
        
        for i in range(X.shape[-2]):
            for j in range(X.shape[-2]):
                # pass specific  atom masks to RBF fxn
                RBF_all.append(self._get_rbf(X[..., i, :], X[..., j, :], E_idx, atom_mask[..., i], atom_mask[..., j]))

        RBF_all = torch.cat(tuple(RBF_all), dim=-1)    
        return RBF_all
    
    def _action_centers(self, X, atom_mask):
        """Takes full coord set [B, L, 14, 3] and atom mask [B, L, 14]
        returns X with appended action center [B, L, 5, 3] and updated atom mask [B, L, 5]        
        """
        if self.action_centers == 'com':
            
            # taking mean position of valid side chain atoms
            X_m = X[:, :, 4:, :] * atom_mask[:, :, 4:, None].repeat(1, 1, 1, 3)
            X_m[X_m == 0.] = torch.nan
            X_ac = torch.nanmean(X_m, dim=-2)
            
            X_ac[X_ac.isnan()] = X[:, :, 4, :][X_ac.isnan()] # if missing side chain, just use virtual Cb
            X_ac = torch.nan_to_num(X_ac) # just-in-case fill with zeros
            # add action center back onto the X info
            X = torch.cat([X[:, :, :4, :], X_ac[:, :, None, :]], dim=-2)
            atom_mask = atom_mask[:, :, :5]
            return X, atom_mask

        elif self.action_centers == 'eoc':
            
            # retrieve last valid atom from each residue
            num_atoms = torch.sum(atom_mask, dim=-1) # [B, L]
            num_atoms = num_atoms[:, :, None, None].repeat(1, 1, 1, 3) # [B, L, 1, 3]
            num_atoms[num_atoms <= 0] = 1 # if no atoms exist, need to keep from throwing index error
            X_ac = torch.gather(X, dim=-2, index=num_atoms - 1) # [B, L, 1, 3]
            X = torch.cat([X[:, :, :4, :], X_ac], dim=-2) # [B, L, 5, 3]
            atom_mask = atom_mask[:, :, :5] # [B, L, 5]
            return X, atom_mask
        
        else:
            raise ValueError("Invalid action center setting!")

    def forward(self, X, mask, residue_idx, chain_labels, atom_mask):
        
        if self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        # Build k-Nearest Neighbors graph
        X_ca = X[:,:,1,:]
        _, E_idx, _ = self._dist(X_ca, mask)
        
        # Pairwise embeddings
        E_positional = self.embeddings(E_idx, residue_idx)
        
        # Pairwise bb atomic distances
        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]
        Cb = self._impute_CB(N, Ca, C)
        sc_atoms = X[..., 5:, :]
        X2 = torch.stack((N, Ca, C, O, Cb), dim=-2)
        X2 = torch.cat((X2, sc_atoms), dim=-2) 
        RBF_all = self._atomic_distances(X2, E_idx, 1 - atom_mask)
        E = torch.cat((E_positional, RBF_all), -1)

        # Embed edges
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx


class SideChainModule(nn.Module):
    def __init__(self, num_positional_embeddings=16, num_rbf=16, 
                 node_features=128, edge_features=128, top_k=30, augment_eps=0., encoder_layers=1, hidden_dim=128, 
                 thru=False, action_centers='none'):
        super(SideChainModule, self).__init__()
                
        vocab=21
        dropout=0.1
        self.augment_eps = augment_eps

        self.features = SideChainProteinFeatures(node_features, edge_features, top_k=top_k, augment_eps=augment_eps, 
                                                 num_rbf=num_rbf, num_positional_embeddings=num_positional_embeddings, 
                                                 action_centers=action_centers)
        
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        
        # self.W_s = nn.Embedding(vocab, hidden_dim)
        self.thru = thru
        
        if self.thru:
            num_in = 128 + 128
        else:
            num_in = 128     
        self.sc_agg = SimpleMPNNAgg(num_hidden=128, num_in=num_in, dropout=0.1, scale=top_k)
        
        # self.sc_layers = nn.ModuleList([
        #     DecLayer(hidden_dim, hidden_dim*3, dropout=dropout)
        #     for _ in range(encoder_layers)
        # ])
    
    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, h_V, atom_mask):
        # generate features (RBFs and positional encodings)
        if self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
        
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all, atom_mask)
        
        # make hidden dim encoding
        h_E = self.W_e(E) # [B, L, K, H]
        
        # embed sequence nodes
        # h_S = self.W_s(S) # [B, L, H]
        
        # grab neighbor sequence nodes
        # h_ES = cat_neighbors_nodes(h_S, h_E, E_idx) # [B, L, K, H * 2]

        # mask keeping all residues except padding
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend

        # TODO add handling for using existing h_V
        if self.thru:
            h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
            h_EV = torch.cat([h_V_expand, h_E], -1)
            h_V = self.sc_agg(h_EV, mask_attend)
        else:  
            h_V = self.sc_agg(h_E, mask_attend) # [B, L, H]
        
        # for layer in self.sc_layers:
        #     # decoder layers do node updates only, using node + seq + edge information
        #     h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
        #     print(h_ESV.shape , mask.attend.shape)
        #     h_ESV = mask_attend * h_ESV
        #     h_V = layer(h_V, h_ESV, mask)
        
        return h_V


class SimpleMPNNAgg(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30):
        super(SimpleMPNNAgg, self).__init__()

        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        # self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        # self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

        self.act = torch.nn.GELU()

    def forward(self, edges, mask=None):
        """ 
        Parallel computation of full transformer layer 
        Message passing to make nodes out of edges without any prior nodes
        """

        # use MLP to construct message
        h_message = self.W3(self.act(self.W2(self.act(self.W1(edges)))))

        # optional: attend to message
        if mask is not None:
            h_message = mask.unsqueeze(-1) * h_message

        # aggregate message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(self.dropout1(dh))

        # Position-wise feedforward
        # dh = self.dense(h_V)
        
        # do self-update
        # h_V = self.norm2(h_V + self.dropout2(dh))

        return h_V



class LightAttention(nn.Module):
    """Source:
    Hannes Stark et al. 2022
    https://github.com/HannesStark/protein-localization/blob/master/models/light_attention.py
    """
    def __init__(self, embeddings_dim=1024, output_dim=11, dropout=0.25, kernel_size=9, conv_dropout=0.25, linear=False):
        super(LightAttention, self).__init__()

        self.feature_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                            padding=kernel_size // 2)
        self.attention_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                            padding=kernel_size // 2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(conv_dropout)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor
        Returns:
            LightAttention reweighted vector of the same dimensions
        """
        o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        o = self.dropout(o)  # [batch_size, embeddings_dim, sequence_length]
        attention = self.attention_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        o1 = o * self.softmax(attention)  # [batch_size, embeddings_dim, sequence_length]
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

        self.num_in = num_in
        print('Agg drop:', dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_in // 2)
        self.W1 = nn.Linear(num_in, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, emb1, emb2, mask=None):
        """ Message passing b/w two embeddings
        emb1 and emb2: embeddings with the same shape [BATCH, EMBED]
        mask: mask of whether to use a message or not (if single mutation, don't use message) [BATCH, EMBED]
        """
        # concatenate emb1 to emb2
        emb_both = torch.cat([emb1, emb2], dim=-1)

        # pass through linear layer or MLP to construct message using BOTH emb1 and emb2
        message = self.act(self.W1(emb_both))
        if mask is not None:
            message = message * mask
            # print(mask.sum() / mask.nelement())

        # aggregate message with original emb1 to get new emb1
        emb1 = self.norm1(emb1 + self.dropout1(message))

        return emb1

    def forwardOLD(self, h_V, h_E, mask_V=None, mask_attend=None):
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

