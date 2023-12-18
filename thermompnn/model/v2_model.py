import torch
import torch.nn as nn

from thermompnn.model.modules import get_protein_mpnn, LightAttention, MultHeadAttn


class TransferModelv2(nn.Module):
    """Rewritten TransferModel class using Batched datasets for faster training"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_dims = list(cfg.model.hidden_dims)
        self.subtract_mut = cfg.model.subtract_mut
        if self.subtract_mut:
            print('Enabled wt mutation subtraction!')
            
        self.num_final_layers = cfg.model.num_final_layers

        if 'decoding_order' not in self.cfg:
            self.cfg.decoding_order = 'left-to-right'
        
        self.prot_mpnn = get_protein_mpnn(cfg)
        
        EMBED_DIM = 128
        HIDDEN_DIM = 128
        VOCAB_DIM = 21

        # modify input size if multi mutations used
        hid_sizes = [(HIDDEN_DIM*self.num_final_layers + EMBED_DIM)]
        hid_sizes += self.hidden_dims
        hid_sizes += [ VOCAB_DIM ]

        print('MLP HIDDEN SIZES:', hid_sizes)
        
        self.lightattn = cfg.model.lightattn if 'lightattn' in cfg.model else False
        
        if self.lightattn:
            self.light_attention = LightAttention(embeddings_dim=(HIDDEN_DIM*self.num_final_layers + EMBED_DIM), kernel_size=1)
        
        self.ddg_out = nn.Sequential()

        for sz1, sz2 in zip(hid_sizes, hid_sizes[1:]):
            self.ddg_out.append(nn.ReLU())
            self.ddg_out.append(nn.Linear(sz1, sz2))

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs):
        """Vectorized fwd function for arbitrary batches of mutations"""

        # getting ProteinMPNN structure embeddings
        all_mpnn_hid, mpnn_embed, _ = self.prot_mpnn(X, S, mask, chain_M, residue_idx, chain_encoding_all, None)
        if self.num_final_layers > 0:
            all_mpnn_hid = torch.cat(all_mpnn_hid[:self.num_final_layers], -1)
            mpnn_embed = torch.cat([all_mpnn_hid, mpnn_embed], -1)
        
        # vectorized indexing of the embeddings (this is very ugly but the best I can do for now)
        # unsqueeze gets mut_pos to shape (batch, 1, 1), then this is copied with expand to be shape (batch, 1, embed_dim) for gather
        mpnn_embed = torch.gather(mpnn_embed, 1, mut_positions.unsqueeze(-1).expand(mut_positions.size(0), mut_positions.size(1), mpnn_embed.size(2)))
        mpnn_embed = torch.squeeze(mpnn_embed, 1) # final shape: (batch, embed_dim)
        
        if self.lightattn:
            mpnn_embed = torch.unsqueeze(mpnn_embed, -1)  # shape for LA input: (batch, embed_dim, seq_length=1)
            mpnn_embed = self.light_attention(mpnn_embed)  # shape for LA output: (batch, embed_dim)

        ddg = self.ddg_out(mpnn_embed)  # shape: (batch, 21)
        
        # index ddg outputs based on mutant AA indices
        if self.cfg.model.subtract_mut:
            ddg = torch.gather(ddg, 1, mut_mutant_AAs) - torch.gather(ddg, 1, mut_wildtype_AAs)
        else:
           ddg = torch.gather(ddg, 1, mut_mutant_AAs)
           
        return ddg
