import torch
import torch.nn as nn
from itertools import permutations
import numpy as np

from thermompnn.model.modules import get_protein_mpnn, LightAttention, MPNNLayer, SideChainModule


def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

def _dist(X, mask, eps=1E-6, top_k=48):
    mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
    dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
    D = mask_2D * torch.sqrt(torch.sum(dX ** 2, 3) + eps)
    D_max, _ = torch.max(D, -1, keepdim=True)
    D_adjust = D + (1. - mask_2D) * D_max
    D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(top_k, X.shape[1]), dim=-1, largest=False)
    return D_neighbors, E_idx


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
        self.mpnn_edges = True if 'edges' in cfg.model else False
        self.dist = True if 'dist' in cfg.model else False
        self.separate_heads = True if 'separate_heads' in cfg.model else False
        self.side_chains = self.cfg.data.side_chains if 'side_chains' in self.cfg.data else False
        self.action_centers = self.cfg.data.action_centers if 'action_centers' in self.cfg.data else 'none'

        self.prot_mpnn = get_protein_mpnn(cfg)
                
        EMBED_DIM = 128
        if self.mutant_embedding:
            EMBED_DIM *= 2

        if self.mpnn_edges:  # add edge input size
            print('Enabling edge features')
            EMBED_DIM += 128
        elif self.dist:
            print('Enabling pairwise dist features')
            EMBED_DIM += 25
            
        if self.side_chains:
            print('Enabling side chains!')
            EMBED_DIM += 128

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
            self.message_size = hid_sizes[0]  # was 128 before
            # self.agg_drop = self.cfg.model.agg_drop
            self.agg_drop = 0.1
            self.aggregator = nn.Sequential()
            # do initial LA-style reweighting before MPNN layer
            self.aggregator.append(nn.ReLU())
            self.aggregator.append(nn.Linear(hid_sizes[0], self.message_size))  # Linear (512, 128)
            self.aggregator.append(MPNNLayer(num_hidden = self.message_size, num_in = self.message_size * 2, dropout = self.agg_drop))
            hid_sizes[0] = self.message_size

        if self.lightattn:
            print('Enabled light attn')
            self.light_attention = LightAttention(embeddings_dim=(HIDDEN_DIM*self.num_final_layers + EMBED_DIM), kernel_size=1)
        
        if self.dist:
            self.dist_norm = nn.LayerNorm(25)  # do normalization of raw dist values

        if self.side_chains:
            self.sc_rbfs = self.cfg.data.side_chain_rbfs if 'side_chain_rbfs' in self.cfg.data else 16
            self.sc_augment_eps = self.cfg.data.side_chain_augment_eps if 'side_chain_augment_eps' in self.cfg.data else 0.0
            self.sc_topk = self.cfg.data.side_chain_topk if 'side_chain_topk' in self.cfg.data else 30
            print('Side chain params:', '\n', 'RBFs:', self.sc_rbfs, '\nAugment_eps:', self.sc_augment_eps)
            self.sc_thru = self.cfg.data.thru if 'thru' in self.cfg.data else False
            self.side_chain_features = SideChainModule(num_positional_embeddings=16, num_rbf=self.sc_rbfs, 
                                                       node_features=128, edge_features=128, 
                                                       top_k=self.sc_topk, augment_eps=self.sc_augment_eps, encoder_layers=1, thru=self.sc_thru, 
                                                       action_centers=self.action_centers)

        self.ddg_out = nn.Sequential()

        if self.double_mutations and self.aggregation != 'mpnn':
            self.ddg_out.append(nn.LayerNorm(HIDDEN_DIM * self.num_final_layers + EMBED_DIM))  # do layer norm before MLP
        
        for sz1, sz2 in zip(hid_sizes, hid_sizes[1:]):
            self.ddg_out.append(nn.ReLU())
            self.ddg_out.append(nn.Linear(sz1, sz2))
            
        # use for either conf model or separate heads
        if self.conf or self.separate_heads:
            print('Loading conf/separate head model!')
            self.conf_model = nn.Sequential()

            # conf model is exact same as ddg out module in total params etc
            for sz1, sz2 in zip(hid_sizes, hid_sizes[1:]):
                self.conf_model.append(nn.LayerNorm(sz1))
                self.conf_model.append(nn.ReLU())
                self.conf_model.append(nn.Linear(sz1, sz2))

    def _get_cbeta(self, X):
        """ProteinMPNN virtual Cb calculation"""
        b = X[:, :, 1, :] - X[:, :, 0, :]
        c = X[:, :, 2, :] - X[:, :, 1, :]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
        Cb = Cb.unsqueeze(2)
        return torch.cat([Cb, X], axis=2)

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask):
        """Vectorized fwd function for arbitrary batches of mutations"""

        # getting ProteinMPNN embeddings (use only backbone atoms)
        if self.side_chains:
            all_mpnn_hid, mpnn_embed, _, mpnn_edges = self.prot_mpnn(X[:, :, :4, :], S, mask, chain_M, residue_idx, chain_encoding_all)
        else:
            all_mpnn_hid, mpnn_embed, _, mpnn_edges = self.prot_mpnn(X, S, mask, chain_M, residue_idx, chain_encoding_all)
        
        if self.dist:
            X = self._get_cbeta(X)

        if self.double_mutations:

            if self.num_final_layers > 0:
                all_mpnn_hid = torch.cat(all_mpnn_hid[:self.num_final_layers], -1)
                mpnn_embed = torch.cat([all_mpnn_hid, mpnn_embed], -1)  # WT seq and structure

                if self.mutant_embedding:
                    # there are actually N sets of mutant sequences, so we need to run this N times
                    mut_embed_list = []
                    for m in range(mut_mutant_AAs.shape[-1]):
                        mut_embed = self.prot_mpnn.W_s(mut_mutant_AAs[:, m])
                        mut_embed_list.append(mut_embed)
                    mut_embed = torch.cat([m.unsqueeze(-1) for m in mut_embed_list], -1) # shape: (Batch, Embed, N_muts)
            
                if self.mpnn_edges:  # add edges to input for gathering
                    # retrieve paired residue edges based on mut_position values

                     # E_idx is [B, K, L] and is a tensor of indices in X that should match neighbors
                    D_n, E_idx = _dist(X[:, :, 1, :], mask)

                    all_mpnn_edges = []
                    n_mutations = [a for a in range(mut_positions.shape[-1])]
                    for n_current in n_mutations:  # iterate over N-order mutations

                        # select the edges at the current mutated positions
                        mpnn_edges_tmp = torch.squeeze(batched_index_select(mpnn_edges, 1, mut_positions[:, n_current:n_current+1]), 1)
                        E_idx_tmp = torch.squeeze(batched_index_select(E_idx, 1, mut_positions[:, n_current:n_current+1]), 1)

                        # find matches for each position in the array of neighbors, grab edges, and add to list
                        edges = []
                        for b in range(E_idx_tmp.shape[0]):
                            # iterate over all neighbors for each sample
                            n_other = [a for a in n_mutations if a != n_current]
                            tmp_edges = []
                            for n_o in n_other:
                                idx = torch.where(E_idx_tmp[b, :] == mut_positions[b, n_o:n_o+1].expand(1, E_idx_tmp.shape[-1]))
                                if len(idx[0]) == 0: # if no edge exists, fill with empty edge for now
                                    edge = torch.full([mpnn_edges_tmp.shape[-1]], torch.nan, device=E_idx.device)
                                else:
                                    edge = mpnn_edges_tmp[b, idx[1][0], :]
                                tmp_edges.append(edge)

                            # aggregate when multiple edges are returned (take mean of valid edges)
                            tmp_edges = torch.stack(tmp_edges, dim=-1)
                            edge = torch.nanmean(tmp_edges, dim=-1)
                            edge = torch.nan_to_num(edge, nan=0)
                            edges.append(edge)

                        edges_compiled = torch.stack(edges, dim=0)
                        all_mpnn_edges.append(edges_compiled)

                    mpnn_edges = torch.stack(all_mpnn_edges, dim=-1) # shape: (Batch, Embed, N_muts)
                
                elif self.dist:
                    # X is coord matrix of size [B, L, 5, 3]
                    eps = 1e-6
                    n_mutations = [a for a in range(mut_positions.shape[-1])]
                    dX_all_agg = []
                    for n_current in n_mutations:
                        # select target coordinates
                        target = batched_index_select(X, 1, mut_positions[:, n_current: n_current + 1])

                        n_other = [a for a in n_mutations if a != n_current]
                        for n_o in n_other:
                            # select each match and calculate distances
                            match = batched_index_select(X, 1, mut_positions[:, n_o: n_o + 1])
                            # get distance calc for every pair of match, target dim 2
                            dX_all = []
                            for a in range(target.shape[2]):
                                for b in range(match.shape[2]):
                                    # do distance calc with target[:, :, a, :] and match[:, ]
                                    dX = torch.sqrt(torch.sum((target[:, :, a, :] - match[:, :, b, :]) ** 2, -1) + 1e-6)
                                    dX_all.append(dX)
                            # gathered distances for all atom combos [B, A ** 2]
                            dX_all = torch.stack(dX_all, dim=-1)
                            dX_all = torch.mean(dX_all, dim=1) # take mean dist for each mut - naive aggregation
                            dX_all_agg.append(dX_all)

                    # dist output should be [B, A ** 2, N_mut] where A is num atoms being used
                    dX_all_agg = torch.stack(dX_all_agg, dim=-1)

            all_mpnn_embed = [] 
            for i in range(mut_mutant_AAs.shape[-1]):
                # gather embedding for a specific position
                current_positions = mut_positions[:, i:i+1] # shape: (B, 1])
                gathered_embed = torch.gather(mpnn_embed, 1, current_positions.unsqueeze(-1).expand(current_positions.size(0), current_positions.size(1), mpnn_embed.size(2)))
                gathered_embed = torch.squeeze(gathered_embed, 1) # final shape: (batch, embed_dim)
                # add specific mutant embedding to gathered embed based on which mutation is being gathered
                gathered_embed = torch.cat([gathered_embed, mut_embed[:, :, i]], -1)
                
                # cat to mpnn edges here if enabled
                if self.mpnn_edges:
                    gathered_embed = torch.cat([gathered_embed, mpnn_edges[:, :, i]], -1)

                # cat to pairwise distances here if enabled
                elif self.dist:
                    gathered_embed = torch.cat([gathered_embed, self.dist_norm(dX_all_agg[:, :, i])], -1)

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


                    # TODO rewrite this to work on N-order mutations!
                    
                    # convert each embedding to learned form
                    for i, emb in enumerate(all_mpnn_embed):
                        all_mpnn_embed[i] = self.aggregator[0:2](emb)
                    
                    new_embs = []
                    # run single aggregated update for each mutation
                    for i, emb in enumerate(all_mpnn_embed):
                        other_embs = [a for ia, a in enumerate(all_mpnn_embed) if ia != i]
                        new_emb = self.aggregator[-1](emb, other_embs)
                        new_embs.append(new_emb)
                    all_mpnn_embed = new_embs
                    # run embeds through initial MLP, then aggregator
                    # all_mpnn_embed[0] = self.aggregator[0:2](all_mpnn_embed[0])
                    # all_mpnn_embed[1] = self.aggregator[0:2](all_mpnn_embed[1])

                    # a = self.aggregator[-1](all_mpnn_embed[0], all_mpnn_embed[1], mask[:, :, 1])
                    # b = self.aggregator[-1](all_mpnn_embed[1], all_mpnn_embed[0], mask[:, :, 1])
                # aggregate the embeddings 
                all_mpnn_embed = torch.stack(all_mpnn_embed, dim=-1)
                all_mpnn_embed[mask] = -float("inf")
                mpnn_embed, _ = torch.max(all_mpnn_embed, dim=-1)

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
            
            if self.side_chains:
                side_chain_embeds = self.side_chain_features(X, S, mask, chain_M, residue_idx, chain_encoding_all, all_mpnn_hid[0], atom_mask)
                
            if self.num_final_layers > 0:
                all_mpnn_hid = torch.cat(all_mpnn_hid[:self.num_final_layers], -1)
                embeds_all = [all_mpnn_hid, mpnn_embed]
                if self.mutant_embedding:
                    mut_embed = self.prot_mpnn.W_s(mut_mutant_AAs[:, 0])
                    embeds_all.append(mut_embed)
                if self.mpnn_edges:  # add edges to input for gathering
                    # the self-edge is the edge with index ZERO for each position L
                    mpnn_edges = mpnn_edges[:, :, 0, :]  # index 2 is the K neighbors index
                    # E_idx is [B, L, K] and is a tensor of indices in X that should match neighbors
                    embeds_all.append(mpnn_edges)
                
                if self.side_chains:
                    embeds_all.append(side_chain_embeds)
                    
            mpnn_embed = torch.cat(embeds_all, -1)
            # vectorized indexing of the embeddings (this is very ugly but the best I can do for now)
            # unsqueeze gets mut_pos to shape (batch, 1, 1), then this is copied with expand to be shape (batch, 1, embed_dim) for gather
            mpnn_embed = torch.gather(mpnn_embed, 1, mut_positions.unsqueeze(-1).expand(mut_positions.size(0), mut_positions.size(1), mpnn_embed.size(2)))
            mpnn_embed = torch.squeeze(mpnn_embed, 1) # final shape: (batch, embed_dim)

            if self.lightattn:
                mpnn_embed = torch.unsqueeze(mpnn_embed, -1)  # shape for LA input: (batch, embed_dim, seq_length=1)
                mpnn_embed = self.light_attention(mpnn_embed)  # shape for LA output: (batch, embed_dim)

        ddg = self.ddg_out(mpnn_embed)  # shape: (batch, 21)
        
        if self.conf or self.separate_heads:
            # single mutant predictions HERE if co-training with separate heads
            conf = self.conf_model(mpnn_embed)
        else:
            conf = None
        
        # index ddg outputs based on mutant AA indices
        if self.cfg.model.subtract_mut:
            ddg = torch.gather(ddg, 1, mut_mutant_AAs) - torch.gather(ddg, 1, mut_wildtype_AAs)
        elif self.single_target:
            pass
        else:
           ddg = torch.gather(ddg, 1, mut_mutant_AAs)
           
        if self.separate_heads:
            # combine single and double mutant ddg batches here using mask
            mask = (mut_mutant_AAs + mut_wildtype_AAs + mut_positions) == 0
            if mask.shape[-1] == 1: # if only single mutations are present, just use only conf
                return conf, None
            # mask -1 row is True if single mutant - replace all these with conf outputs
            ddg[mask[:, -1]] = conf[mask[:, -1]]
            
        return ddg, conf
