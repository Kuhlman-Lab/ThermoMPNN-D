import torch
import torch.nn as nn

from thermompnn.model.modules import get_protein_mpnn, LightAttention


class TransferModelSiamese(nn.Module):
    """Rewritten TransferModel class using Siamese training and Batched datasets"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_dims = list(cfg.model.hidden_dims)
        print('Siamese Model Enabled!')

        self.num_final_layers = cfg.model.num_final_layers

        if 'decoding_order' not in self.cfg:
            self.cfg.decoding_order = 'left-to-right'
        
        self.prot_mpnn = get_protein_mpnn(cfg)
        
        EMBED_DIM = 128 * 2
        HIDDEN_DIM = 128
        VOCAB_DIM = 1

        # modify input size if multi mutations used
        hid_sizes = [(HIDDEN_DIM*self.num_final_layers + EMBED_DIM)]
        hid_sizes += self.hidden_dims
        hid_sizes += [ VOCAB_DIM ]

        print('MLP HIDDEN SIZES:', hid_sizes)
        
        self.lightattn = cfg.model.lightattn if 'lightattn' in cfg.model else False
        
        if self.lightattn:
            print('Enabled LightAttention')
            self.light_attention = LightAttention(embeddings_dim=(HIDDEN_DIM*self.num_final_layers + EMBED_DIM))

        self.ddg_out = nn.Sequential()

        for sz1, sz2 in zip(hid_sizes, hid_sizes[1:]):
            self.ddg_out.append(nn.ReLU())
            self.ddg_out.append(nn.Linear(sz1, sz2))

    def forward(self, wt_features, mut_features):
        """Vectorized fwd function for arbitrary batches of mutations"""
        
        features = [(wt_features, mut_features), (mut_features, wt_features)]
        ddg_both = []
        
        for feature_pair in features:
            # feat1 is the "main" set, feat2 gets used for mutant residue embedding retrieval
            feat1, feat2 = feature_pair
            # grab current features
            X, S, mask, lengths, chain_M, residue_idx, chain_encoding_all, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs = feat1
            # grab mutant sequence for embedding generation
            S2 = feat2[1]
            
            # getting ProteinMPNN structure embeddings
            all_mpnn_hid, seq_embed, _ = self.prot_mpnn(X, S, mask, chain_M, residue_idx, chain_encoding_all, None)
        
            if self.num_final_layers > 0:
                all_mpnn_hid = torch.cat(all_mpnn_hid[:self.num_final_layers], -1)
                # grab mutant sequence embedding
                if S2.shape != S.shape:
                    S2 = self._fix_mutant_seq(S, S2, mut_wildtype_AAs, mut_mutant_AAs, mut_positions)
                mutant_seq_embed = self.prot_mpnn.W_s(S2)
                mpnn_embed = torch.cat([all_mpnn_hid, seq_embed, mutant_seq_embed], -1)
            
            # vectorized indexing of the embeddings (this is very ugly but the best I can do for now)
            # unsqueeze gets mut_pos to shape (batch, 1, 1), then this is copied with expand to be shape (batch, 1, embed_dim) for gather
            mpnn_embed = torch.gather(mpnn_embed, 1, mut_positions.unsqueeze(-1).expand(mut_positions.size(0), mut_positions.size(1), mpnn_embed.size(2)))
            mpnn_embed = torch.squeeze(mpnn_embed, 1) # final shape: (batch, embed_dim)
            
            # pass through lightattn
            if self.lightattn:
                mpnn_embed = torch.unsqueeze(mpnn_embed, -1)  # shape for LA input: (batch, embed_dim, seq_length=1)
                mpnn_embed = self.light_attention(mpnn_embed, mask)  # shape for LA output: (batch, embed_dim)

            ddg = self.ddg_out(mpnn_embed)  # shape: (batch, 1)
            ddg_both.append(ddg)

        # first pred is the direct mutation, second is the reverse mutations            
        ddg1, ddg2 = ddg_both
        return ddg1, ddg2

    def _fix_mutant_seq(self, S, S2, wtAA, mutAA, positions):
        """
        If mutant seq isn't a match, make it synthetically
        Only works for batch size = 1 for now
        """
        new_S2 = torch.clone(S)
        assert new_S2[0, positions[0, 0]] == wtAA[0, 0]  # check you've got the right spot
        new_S2[0, positions[0, 0]] = mutAA[0, 0]  # overwrite AA embedding manually
        return new_S2
