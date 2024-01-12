import torch
import torch.nn as nn
from itertools import permutations

from thermompnn.model.modules import get_protein_mpnn, LightAttention, MPNNLayer


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
        self.mutant_embedding = cfg.model.mutant_embedding if 'mutant_embedding' in cfg.model else False
        self.single_target = self.cfg.model.single_target if 'single_target' in cfg.model else False
        self.double_mutations = True if 'double' in self.cfg.data.mut_types else False

        self.aggregation = self.cfg.model.aggregation if 'aggregation' in self.cfg.model else 'na'
        self.double_mutations = True if self.aggregation != 'na' else False
        self.conf = cfg.training.confidence if 'confidence' in cfg.training else None
        
        self.prot_mpnn = get_protein_mpnn(cfg)
                
        EMBED_DIM = 128
        if self.mutant_embedding:
            EMBED_DIM *= 2

        HIDDEN_DIM = 128
        VOCAB_DIM = 21 if not self.single_target else 1

        # modify input size if multi mutations used
        hid_sizes = [(HIDDEN_DIM*self.num_final_layers + EMBED_DIM)]
        hid_sizes += self.hidden_dims
        hid_sizes += [ VOCAB_DIM ]

        print('MLP HIDDEN SIZES:', hid_sizes)
        
        self.lightattn = cfg.model.lightattn if 'lightattn' in cfg.model else False
        
        if self.aggregation == 'bias':

            # need to scramble/self-learn BEFORE aggregation to allow disambiguation
            self.message_size = 128            
            self.light_attention = nn.Sequential()
            self.light_attention.append(nn.LayerNorm(hid_sizes[0]))
            self.light_attention.append(nn.ReLU())
            self.light_attention.append(nn.Linear(hid_sizes[0], self.message_size))
            print('Initial representation mechanism', self.light_attention)

            # override all other settings for now
            self.aggregator = nn.Sequential()
            hid_sizes[0] = self.message_size
            # auxiliary MLP to predict residual
            for sz1, sz2 in zip(hid_sizes, hid_sizes[1:]):
                self.aggregator.append(nn.LayerNorm(sz1))
                self.aggregator.append(nn.ReLU())
                self.aggregator.append(nn.Linear(sz1, sz2))
            print('Thermodynamic coupling model:', self.aggregator)
            
            # set up ddG prediction MLP
            hid_sizes[0] = (HIDDEN_DIM*self.num_final_layers + EMBED_DIM)
            self.ddg_out = nn.Sequential()
            for sz1, sz2 in zip(hid_sizes, hid_sizes[1:]):
                self.ddg_out.append(nn.LayerNorm(sz1))
                self.ddg_out.append(nn.ReLU())
                self.ddg_out.append(nn.Linear(sz1, sz2))
            print('Main ddG model:', self.ddg_out)
            return
        
        if self.aggregation == 'mpnn':
            self.message_size = 128            
            self.agg_drop = self.cfg.model.agg_drop
            self.aggregator = nn.Sequential()
            self.aggregator.append(nn.ReLU())
            self.aggregator.append(nn.Linear(hid_sizes[0], self.message_size))  # Linear (512, 128)
            self.aggregator.append(MPNNLayer(num_hidden = self.message_size, num_in = self.message_size * 2, dropout = self.agg_drop))
            hid_sizes[0] = self.message_size

        if self.lightattn:
            print('Enabled light attn')
            self.light_attention = LightAttention(embeddings_dim=(HIDDEN_DIM*self.num_final_layers + EMBED_DIM), kernel_size=1)
        
        self.ddg_out = nn.Sequential()

        if self.double_mutations and self.aggregation != 'mpnn':
            self.ddg_out.append(nn.LayerNorm(HIDDEN_DIM * self.num_final_layers + EMBED_DIM))  # do layer norm before MLP

        for sz1, sz2 in zip(hid_sizes, hid_sizes[1:]):
            self.ddg_out.append(nn.ReLU())
            self.ddg_out.append(nn.Linear(sz1, sz2))
            
        # TODO add error prediction head here (alongside ddg out)
        if self.conf:
            self.conf_model = nn.Sequential()

            # conf model is exact same as ddg out module in total params etc
            for sz1, sz2 in zip(hid_sizes, hid_sizes[1:]):
                self.conf_model.append(nn.LayerNorm(sz1))
                self.conf_model.append(nn.ReLU())
                self.conf_model.append(nn.Linear(sz1, sz2))


    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs):
        """Vectorized fwd function for arbitrary batches of mutations"""

        # getting ProteinMPNN embeddings
        all_mpnn_hid, mpnn_embed, _ = self.prot_mpnn(X, S, mask, chain_M, residue_idx, chain_encoding_all)

        if self.double_mutations:

            if self.num_final_layers > 0:
                all_mpnn_hid = torch.cat(all_mpnn_hid[:self.num_final_layers], -1)
                mpnn_embed = torch.cat([all_mpnn_hid, mpnn_embed], -1)  # WT seq and structure

                if self.mutant_embedding:
                    # there are actually two sets of mutant sequences, so we need to run this TWICE
                    mut_embed_list = []
                    for m in range(mut_mutant_AAs.shape[-1]):
                        mut_embed = self.prot_mpnn.W_s(mut_mutant_AAs[:, m])
                        mut_embed_list.append(mut_embed)
                    mut_embed = torch.cat([m.unsqueeze(-1) for m in mut_embed_list], -1) # shape: (Batch, Embed, N_muts)
            
            all_mpnn_embed = [] 
            for i in range(mut_mutant_AAs.shape[-1]):
                # gather embedding for a specific position
                current_positions = mut_positions[:, i:i+1] # shape: (B, 1])
                gathered_embed = torch.gather(mpnn_embed, 1, current_positions.unsqueeze(-1).expand(current_positions.size(0), current_positions.size(1), mpnn_embed.size(2)))
                gathered_embed = torch.squeeze(gathered_embed, 1) # final shape: (batch, embed_dim)
                # add specific mutant embedding to gathered embed based on which mutation is being gathered
                gathered_embed = torch.cat([gathered_embed, mut_embed[:, :, i]], -1)
                all_mpnn_embed.append(gathered_embed)  # list with length N_mutations - used to make permutations

            if self.aggregation == 'bias':  # altered network setup to emphasize DM bias term
                
                # get mask of which embeds are empty second halves of single mutations
                mask = (mut_mutant_AAs + mut_wildtype_AAs + mut_positions) == 0
                # mask and embeds can be [B, E, 1] or [B, E, 2]
                assert(torch.sum(mask[:, 0]) == 0)  # check that first mutation is ALWAYS visible
                mask = mask.unsqueeze(1).repeat(1, all_mpnn_embed[0].shape[1], 1)  # expand along embedding dimension

                if mask.shape[-1] == 1: # single mutant batches
                    # only run through ddg head
                    ddg_total = self.ddg_out(all_mpnn_embed[0])
                    return ddg_total
                
                else:
                    # run each embedding through MLP to get purely additive ddG predictions
                    ddg_list = []
                    for emb in all_mpnn_embed:
                        ddg_list.append(self.ddg_out(emb))
                      
                    # add them together, using mask as needed
                    ddg_list = torch.stack(ddg_list, dim=-1)

                    ddg_list[mask[:, 0:1, :]] = torch.nan
                    ddg_total = torch.nansum(ddg_list, dim=-1)

                    # process embeddings for bias prediction network
                    for n, emb in enumerate(all_mpnn_embed):
                        emb = self.light_attention(emb)  # shape for LA output: (batch, embed_dim)
                        all_mpnn_embed[n] = emb  # update list of embs
                    
                    # aggregate with max strategy
                    all_mpnn_embed = torch.stack(all_mpnn_embed, dim=-1)
                    all_mpnn_embed[mask[:, :all_mpnn_embed.shape[1], :]] = -float("inf")
                    mpnn_embed, _ = torch.max(all_mpnn_embed, dim=-1)
                    
                    # gather and add bias term, zero out for single mutants
                    bias = self.aggregator(mpnn_embed)
                    bias = bias * ~mask[:, 0:1, 1]  # mask bias based on whether single or double mutant
                    ddg_total += bias
                    return ddg_total

            elif self.aggregation == 'mpnn':
                # get mask of which embeds are empty second halves of single mutations
                mask = (mut_mutant_AAs + mut_wildtype_AAs + mut_positions) == 0
                # mask and embeds can be [B, E, 1] or [B, E, 2]
                assert(torch.sum(mask[:, 0]) == 0)  # check that first mutation is ALWAYS visible
                mask = mask.unsqueeze(1).repeat(1, self.message_size, 1)  # expand along embedding dimension

                if mask.shape[-1] == 1: # single mutant batches
                    # run through for norm, but don't do any updates
                    # print('Singles only batch!')
                    # run embed through initial MLP, then aggregator
                    all_mpnn_embed[0] = self.aggregator[0:2](all_mpnn_embed[0])
                    all_mpnn_embed[0] = self.aggregator[-1](all_mpnn_embed[0], all_mpnn_embed[0], mask.squeeze(-1) * 0.)
                else:
                    # run both embeds through aggregator - use second half of mask to decide where to update
                    # print('Doubles/mixed batch!')
                    # run embeds through initial MLP, then aggregator
                    all_mpnn_embed[0] = self.aggregator[0:2](all_mpnn_embed[0])
                    all_mpnn_embed[1] = self.aggregator[0:2](all_mpnn_embed[1])

                    all_mpnn_embed[0] = self.aggregator[-1](all_mpnn_embed[0], all_mpnn_embed[1], mask[:, :, 1])
                    all_mpnn_embed[1] = self.aggregator[-1](all_mpnn_embed[1], all_mpnn_embed[0], mask[:, :, 1])
                # run each permutation through update/aggregator module)
                all_mpnn_embed = torch.stack(all_mpnn_embed, dim=-1)
                all_mpnn_embed[mask] = torch.nan
                # aggregate the embeddings 
                mpnn_embed = torch.nanmean(all_mpnn_embed, dim=-1)

            else:  # non-learned aggregations
                # run each embedding through LA / MLP layer, even if masked out
                for n, emb in enumerate(all_mpnn_embed):
                    emb = torch.unsqueeze(emb, -1)  # shape for LA input: (batch, embed_dim, seq_length=1)
                    emb = self.light_attention(emb)  # shape for LA output: (batch, embed_dim)
                    all_mpnn_embed[n] = emb  # update list of embs

                all_mpnn_embed = torch.stack(all_mpnn_embed, dim=-1)  # shape: (batch, embed_dim, n_mutations)

                # get mask of which embeds are empty second halves of single mutations
                mask = (mut_mutant_AAs + mut_wildtype_AAs + mut_positions) == 0
                assert(torch.sum(mask[:, 0]) == 0)  # check that first mutation is ALWAYS visible
                mask = mask.unsqueeze(1).repeat(1, all_mpnn_embed.shape[1], 1)  # expand along embedding dimension
                
                # depending on aggregation fxn, different masking needs to be done
                if self.aggregation == 'mean':
                    all_mpnn_embed[mask] = torch.nan
                    mpnn_embed = torch.nanmean(all_mpnn_embed, dim=-1)
                elif self.aggregation == 'sum':
                    all_mpnn_embed[mask] = 0
                    mpnn_embed = torch.sum(all_mpnn_embed, dim=-1)
                elif self.aggregation == 'prod':
                    all_mpnn_embed[mask] = 1
                    mpnn_embed = torch.prod(all_mpnn_embed, dim=-1)
                elif self.aggregation == 'max':
                    all_mpnn_embed[mask] = -float("inf")
                    mpnn_embed, _ = torch.max(all_mpnn_embed, dim=-1)
                else:
                    raise ValueError("Invalid aggregation function selected")

        else:  # standard (single-mutation) indexing
            if self.num_final_layers > 0:
                all_mpnn_hid = torch.cat(all_mpnn_hid[:self.num_final_layers], -1)
                if self.mutant_embedding:
                    mut_embed = self.prot_mpnn.W_s(mut_mutant_AAs[:, 0])
            mpnn_embed = torch.cat([all_mpnn_hid, mpnn_embed], -1)

            # vectorized indexing of the embeddings (this is very ugly but the best I can do for now)
            # unsqueeze gets mut_pos to shape (batch, 1, 1), then this is copied with expand to be shape (batch, 1, embed_dim) for gather
            mpnn_embed = torch.gather(mpnn_embed, 1, mut_positions.unsqueeze(-1).expand(mut_positions.size(0), mut_positions.size(1), mpnn_embed.size(2)))
            mpnn_embed = torch.squeeze(mpnn_embed, 1) # final shape: (batch, embed_dim)
            if self.mutant_embedding:
                mpnn_embed = torch.cat([mpnn_embed, mut_embed], -1)

            if self.lightattn:
                mpnn_embed = torch.unsqueeze(mpnn_embed, -1)  # shape for LA input: (batch, embed_dim, seq_length=1)
                mpnn_embed = self.light_attention(mpnn_embed)  # shape for LA output: (batch, embed_dim)

        ddg = self.ddg_out(mpnn_embed)  # shape: (batch, 21)
        
        if self.conf:
            conf = self.conf_model(mpnn_embed)
        else:
            conf = None
        
        # index ddg outputs based on mutant AA indices
        if self.cfg.model.subtract_mut:
            ddg = torch.gather(ddg, 1, mut_mutant_AAs) - torch.gather(ddg, 1, mut_wildtype_AAs)
            
        elif self.single_target:
            return ddg, conf
        else:
           ddg = torch.gather(ddg, 1, mut_mutant_AAs)
           
        return ddg, conf
