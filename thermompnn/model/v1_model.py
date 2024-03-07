import torch
import torch.nn as nn
from random import shuffle
from copy import deepcopy

from thermompnn.model.modules import get_protein_mpnn, LightAttention
from thermompnn.protein_mpnn_utils import tied_featurize

HIDDEN_DIM = 128
EMBED_DIM = 128
VOCAB_DIM = 21
ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'


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
