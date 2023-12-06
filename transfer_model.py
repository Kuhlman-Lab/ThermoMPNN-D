import torch
import torch.nn as nn
from protein_mpnn_utils import ProteinMPNN, tied_featurize, PositionWiseFeedForward
from model_utils import featurize
import os
import torch.nn.functional as F
import math
from random import shuffle
from copy import deepcopy


HIDDEN_DIM = 128
EMBED_DIM = 128
VOCAB_DIM = 21
ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'

MLP = True
SUBTRACT_MUT = True


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


class TransferModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_dims = list(cfg.model.hidden_dims)
        self.subtract_mut = cfg.model.subtract_mut
        self.num_final_layers = cfg.model.num_final_layers
        self.lightattn = cfg.model.lightattn if 'lightattn' in cfg.model else False
        self.multiheadattn = cfg.model.multiheadattn if 'multiheadattn' in cfg.model else False
        self.mutant_embedding = cfg.model.mutant_embedding if 'mutant_embedding' in cfg.model else False
        self.mlp_first = cfg.model.mlp_first if 'mlp_first' in cfg.model else False
        self.single_target = cfg.model.single_target if 'single_target' in cfg.model else False

        if 'decoding_order' not in self.cfg:
            self.cfg.decoding_order = 'left-to-right'
        
        self.prot_mpnn = get_protein_mpnn(cfg)
        EMBED_DIM = 128 if not self.mutant_embedding else 256
        EMBED_DIM = EMBED_DIM if 'double' not in self.cfg.data.mut_types else EMBED_DIM * 2
        EMBED_DIM = EMBED_DIM if not self.mlp_first else EMBED_DIM // 2

        HIDDEN_DIM = 128 if 'double' not in self.cfg.data.mut_types else 256
        HIDDEN_DIM = HIDDEN_DIM if not self.mlp_first else HIDDEN_DIM // 2

        if 'double' in self.cfg.data.mut_types:
            VOCAB_DIM = 441
        else:
            VOCAB_DIM = 21
        
        if self.single_target:
            print('Enabled single-target training...')
            VOCAB_DIM = 1

        # modify input size if multi mutations used
        hid_sizes = [(HIDDEN_DIM*self.num_final_layers + EMBED_DIM)]
        hid_sizes += self.hidden_dims
        hid_sizes += [ VOCAB_DIM ]

        print('MLP HIDDEN SIZES:', hid_sizes)

        if self.lightattn:
            print('Enabled LightAttention')
            self.light_attention = LightAttention(embeddings_dim=(HIDDEN_DIM*self.num_final_layers + EMBED_DIM))
        elif self.multiheadattn:
            print('Enabled Multi-Head Attention')
            self.light_attention = MultHeadAttn(embed_dim = (HIDDEN_DIM * self.num_final_layers + EMBED_DIM), n_heads=8)

        if self.mlp_first:
            print('Enabled INIT MLP transform layer')
            self.init_mlp = nn.Sequential()
            self.init_mlp.append(nn.ReLU())
            self.init_mlp.append(nn.Linear(hid_sizes[0], hid_sizes[0]))
        else:
            self.init_mlp = None

        self.both_out = nn.Sequential()

        for sz1, sz2 in zip(hid_sizes, hid_sizes[1:]):
            self.both_out.append(nn.ReLU())
            self.both_out.append(nn.Linear(sz1, sz2))

        self.ddg_out = nn.Linear(1, 1)
        # self.dtm_out = nn.Linear(1, 1)  # TODO remove

    def forward(self, pdb, mutations, tied_feat=True):        
        device = next(self.parameters()).device

        X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize([pdb[0]], device, None, None, None, None, None, None, ca_only=False)

        # getting ProteinMPNN structure embeddings
        all_mpnn_hid, mpnn_embed, _ = self.prot_mpnn(X, S, mask, chain_M, residue_idx, chain_encoding_all, None)
        if self.num_final_layers > 0:
            mpnn_hid = torch.cat(all_mpnn_hid[:self.num_final_layers], -1)

        out = []

        if 'double' in self.cfg.data.mut_types:
            mutations = self._single_to_double_mut(mutations)  # stochastic sampling changes every epoch
            if 'flip-double' in self.cfg.data.mut_types:
                mutations = self._flip_double_mut(mutations)

        for mut in mutations:
            if mut is None:
                out.append(None)
                continue

            if 'double' in self.cfg.data.mut_types:
                # double mutant embeddings are aggregated to preserve input/model size
                multi_embeddings = []

                for n, pos in enumerate(mut.position):
                    emb_mut = mut.mutation[n] if self.mutant_embedding else None
                    multi_embeddings.append(self._get_embedding(mpnn_hid[0], mpnn_embed[0], pos, device, emb_mut))

                if self.mlp_first:
                    # instead of concat, use an MLP to disambiguate your vectors, then aggregate them
                    multi_embeddings[0] = self.init_mlp(torch.unsqueeze(torch.squeeze(multi_embeddings[0]), 0))
                    multi_embeddings[1] = self.init_mlp(torch.unsqueeze(torch.squeeze(multi_embeddings[1]), 0))
                    lin_input = torch.mean(torch.stack(multi_embeddings), dim=0)
                    lin_input = torch.flatten(lin_input)
                    both_input = torch.unsqueeze(self.both_out(lin_input), -1)
                    ddg_out = self.ddg_out(both_input)
                    # single ddg out
                    ddg = ddg_out[0][0]

                else:
                    # concatenate, NOT aggregate, to preserve order equivariance
                    lin_input = torch.cat(multi_embeddings, 0)

                    mut_lookup_list = []
                    for m, wt in zip(mut.mutation, mut.wildtype):
                        aa_index = ALPHABET.index(m)
                        wt_aa_index = ALPHABET.index(wt)
                        mut_lookup_list.append(aa_index)
                        # passing vector through lightattn
                    if self.lightattn:
                        lin_input = self.light_attention(lin_input, mask)
                    elif self.multiheadattn:
                        lin_input = torch.unsqueeze(lin_input, 0)
                        lin_input = self.light_attention(lin_input, lin_input, lin_input, mask)
                    else:
                        lin_input = torch.flatten(lin_input)

                    both_input = torch.unsqueeze(self.both_out(lin_input), -1)
                    ddg_out = self.ddg_out(both_input)
                
                    # final index is ORDER-DEPENDENT b/c F1P:R7Q is different from R1P:F7Q
                    final_idx = (mut_lookup_list[0] * 21) + mut_lookup_list[1]
                    ddg = ddg_out[final_idx][0]
            
            else:
                emb_mut = mut.mutation[0] if self.mutant_embedding else None
                lin_input = self._get_embedding(mpnn_hid[0], mpnn_embed[0], mut.position[0], device, emb_mut)

                aa_index = ALPHABET.index(mut.mutation[0])
                wt_aa_index = ALPHABET.index(mut.wildtype[0])
                # passing vector through lightattn
                if self.lightattn:
                    lin_input = self.light_attention(lin_input, mask)
                elif self.multiheadattn:
                    lin_input = self.light_attention(lin_input, lin_input, lin_input, mask)

                both_input = torch.unsqueeze(self.both_out(lin_input), -1)
                ddg_out = self.ddg_out(both_input)
                ddg = self._get_ddG(ddg_out, aa_index, wt_aa_index)

            out.append({
                "ddG": torch.unsqueeze(ddg, 0),
            })
        return out, None
    
    def _flip_double_mut(self, all_mutations):
        """Make flipped double mutations for training"""

        singles, doubles = [], []
        for a in all_mutations:
            if len(a.position) > 1:
                doubles.append(a)
            else:
                singles.append(a)

        if len(doubles) == 0:
            return singles

        flip_doubles = []
        for n, doub in enumerate(doubles):
            flip_doub = deepcopy(doub)
            flip_doub.position.reverse()
            flip_doub.wildtype.reverse()
            flip_doub.mutation.reverse()
            flip_doubles.append(flip_doub)
        
        all_mutations = singles + doubles + flip_doubles
        shuffle(all_mutations)
        return all_mutations
    
    def _single_to_double_mut(self, all_mutations):
        """Convert a single mutation to a pseudo-double mutation with stochastic sampling"""
        # split into single/double mutants
        singles, doubles = [], []
        for a in all_mutations:
            if len(a.position) > 1:
                doubles.append(a)
            else:
                singles.append(a)

        if len(singles) == 0:
            return doubles
        
        # select random second single mutant
        rand_idx = torch.randint(len(singles), (len(singles), ))

        for n, sing in enumerate(singles):
            rand_mut = singles[rand_idx[n]]
            idx = rand_idx[n] % 2
            sing.position.insert(idx, rand_mut.position[0])
            sing.wildtype.insert(idx, rand_mut.wildtype[0])
            sing.mutation.insert(idx, rand_mut.wildtype[0])
        all_mutations = singles + doubles
        shuffle(all_mutations)
        return all_mutations
    
    def _get_ddG(self, ddg_out, mut_idx, wt_idx):
        """Retrieve ddG from output layer depending on model configuration"""
        if self.single_target:
            return torch.squeeze(ddg_out)
        elif self.subtract_mut:
            return ddg_out[mut_idx][0] - ddg_out[wt_idx][0]
        else:
            return ddg_out[mut_idx][0]
    
    def _get_embedding(self, hid_embed, seq_embed, idx, device, mutant=None):
        """Retrieve ProteinMPNN embeddings for a given position"""
        inputs = []
        if self.num_final_layers > 0:
            hid = hid_embed[idx]  # MPNN hidden decoder embeddings
            inputs.append(hid)

        embed = seq_embed[idx]  # MPNN seq embeddings
        inputs.append(embed)
        # grab mutant embedding too if mutant is not None
        if mutant is not None:
            S_mut = torch.tensor([ALPHABET.index(mutant)], dtype=torch.long, device=device)
            inputs.append(torch.flatten(self.prot_mpnn.W_s(S_mut)))

        return torch.unsqueeze(torch.cat(inputs, -1), -1)  # returns [N, 1] vector


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



class TransferModelAUG(TransferModel):

    def forward(self, pdb, mutations, tied_feat=True):
        device = next(self.parameters()).device

        X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize([pdb[0]], device, None, None, None, None, None, None, ca_only=False)
        out = []
        batch_size = 5
        S_all = torch.zeros((batch_size, S[0].shape[0]), dtype=torch.long, device=device)
        # need to batchify forward calls for speedup
        for n in range(0, len(mutations), batch_size):
            mut_batch = mutations[n:n + batch_size]
            for b, mut in enumerate(mut_batch):
                S_wildtype = ALPHABET[S[0][mut.position]]

                if mut is None:
                    out.append(None)
                    continue
                # if sequence and mutation WT are mis-matched, override sequence (for augmentation)
                if S_wildtype != mut.wildtype:
                    S_new = torch.concat([S[0][:mut.position],
                                          torch.tensor([ALPHABET.index(mut.wildtype)], device=device),
                                          S[0][mut.position + 1:]], dim=0)
                    S_wildtype = ALPHABET[S_new[mut.position]]
                    assert S_wildtype == mut.wildtype
                else:
                    S_new = S[0]
                S_all[b, :] = S_new

            S_all = S_all.long()
            # need to duplicate out all masks etc
            X_all = X.repeat(batch_size, 1, 1, 1)
            mask = mask.repeat(batch_size, 1)
            chain_M = chain_M.repeat(batch_size, 1)
            chain_M_pos = chain_M_pos.repeat(batch_size, 1)
            residue_idx = residue_idx.repeat(batch_size, 1)
            chain_encoding_all = chain_encoding_all.repeat(batch_size, 1)
            # run batched fwd call
            all_mpnn_hid, mpnn_embed, _ = self.prot_mpnn(X_all, S_all, mask, chain_M * chain_M_pos, residue_idx,
                                                         chain_encoding_all, None)
            # Run ProteinMPNN fwd fxn for each individual mutation
            # all_mpnn_hid, mpnn_embed, _ = self.prot_mpnn(X, S, mask, chain_M * chain_M_pos, residue_idx,
            #                                              chain_encoding_all, None)
            if self.num_final_layers > 0:
                mpnn_hid = torch.cat(all_mpnn_hid[:self.num_final_layers], -1)

            inputs = []

            aa_index = ALPHABET.index(mut.mutation)
            wt_aa_index = ALPHABET.index(mut.wildtype)

            if self.num_final_layers > 0:
                hid = mpnn_hid[0][mut.position]  # MPNN hidden embeddings at mut position
                inputs.append(hid)

            embed = mpnn_embed[0][mut.position]  # MPNN seq embeddings at mut position
            inputs.append(embed)

            # concatenating hidden layers and embeddings
            lin_input = torch.cat(inputs, -1)

            # passing vector through light attn
            if self.lightattn:
                lin_input = torch.unsqueeze(torch.unsqueeze(lin_input, -1), 0)
                lin_input = self.light_attention(lin_input, mask)

            both_input = torch.unsqueeze(self.both_out(lin_input), -1)
            ddg_out = self.ddg_out(both_input)

            if self.subtract_mut:
                ddg = ddg_out[aa_index][0] - ddg_out[wt_aa_index][0]
            else:
                ddg = ddg_out[aa_index][0]

            out.append({
                "ddG": torch.unsqueeze(ddg, 0),
            })

        return out, None


class MultHeadAttn(nn.Module):
    """Implementation of multi-head attention with query/key/value generation
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

        # [1, embed_size, seq_length] is query/value/key size

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

    def forward(self, x: torch.Tensor, mask, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.
        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        
        o = self.dropout(o)  # [batch_gsize, embeddings_dim, sequence_length]

        attention = self.attention_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        
        o1 = o * self.softmax(attention)  # [batch_size, embeddings_dim]
        return torch.squeeze(o1, -1)
