from __future__ import print_function
import torch

import torch.utils
import torch.nn as nn
import torch.nn.functional as F

from proteinmpnn.model_utils import ProteinFeatures, EncLayer, DecLayer, IPMPDecoder, IPMPEncoder
from proteinmpnn.model_utils import gather_nodes, cat_neighbors_nodes

"""
Copied model class from proteinmpnn.model_utils 
only changes are decoding order/visibility and returned arguments
"""


class ProteinMPNN(nn.Module):
    def __init__(self, num_letters=21, node_features=128, edge_features=128,
        hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
        vocab=21, k_neighbors=32, augment_eps=0.1, dropout=0.1,
        use_ipmp=False, n_points=8, side_chains=False, single_res_rec=False, 
        decoding_order='id', nfl=1):
        super(ProteinMPNN, self).__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.use_ipmp = use_ipmp
        self.side_chains = side_chains
        self.single_res_rec = single_res_rec
        self.decoding_order = decoding_order
        if self.single_res_rec:
            print('Running single residue recovery ProteinMPNN!')
        
        self.nfl = nfl
        print('ProteinMPNN NFL value:', self.nfl)
        print('ProteinMPNN Dropout:', dropout)

        self.features = ProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps, side_chains=False)

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

        print('Encoder and Decoder Layers:', num_encoder_layers, num_decoder_layers)
        # Encoder layers
        if not use_ipmp:
            self.encoder_layers = nn.ModuleList([
                EncLayer(hidden_dim, hidden_dim*2, dropout=dropout)
                for _ in range(num_encoder_layers)
            ])
        else:
            self.encoder_layers = nn.ModuleList([
                IPMPEncoder(hidden_dim, hidden_dim*2, dropout=dropout, n_points=n_points)
                for _ in range(num_encoder_layers)
            ])

        # If side chains are enabled, add them in right before the decoder
        if self.side_chains:
            print('Side chains enabled!')
            self.sca_features = ProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps, side_chains=True)
            self.sca_W_e = nn.Linear(edge_features, hidden_dim, bias=True)

            # add additional hidden_dim to accomodate injection of side chain features
            self.decoder_layers = nn.ModuleList([
                DecLayer(hidden_dim, hidden_dim*4, dropout=dropout)
                for _ in range(num_decoder_layers)
            ])
            
        else:
            # Decoder layers
            if not use_ipmp:
                self.decoder_layers = nn.ModuleList([
                    DecLayer(hidden_dim, hidden_dim*3, dropout=dropout)
                    for _ in range(num_decoder_layers)
                ])
            else:
                self.decoder_layers = nn.ModuleList([
                    IPMPDecoder(hidden_dim, hidden_dim*3, dropout=dropout, n_points=n_points)
                    for _ in range(num_decoder_layers)
                ])
                
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        """ Graph-conditioned sequence model """
        device=X.device
        # Prepare node and edge embeddings
        all_hidden = []

        if not self.side_chains:
            X = torch.nan_to_num(X, nan=0.0)
            E, E_idx, X = self.features(X, mask, residue_idx, chain_encoding_all)
        else:
            mask_per_atom = (~torch.isnan(X)[:, :, :, 0]).long()  # different side chain atoms exist for different residues
            # mask_per_atom is shape [B, L_max, 14] - use this for RBF masking
            X = torch.nan_to_num(X, nan=0.0)

            # only pass backbone to main ProteinFeatures
            E, E_idx, _ = self.features(X[..., :4, :], mask, residue_idx, chain_encoding_all)
            
            # pass full side chain set to separate SideChainFeatures for use in DECODER ONLY
            E_sc, E_idx_sc, X = self.sca_features(X, mask, residue_idx, chain_encoding_all, mask_per_atom)
            h_E_sc = self.sca_W_e(E_sc) # project down to hidden dim

        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            if not self.use_ipmp:
                h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
            else:
                h_V, h_E = layer(h_V, h_E, E_idx, X, mask, mask_attend)

        # TODO henry - if NFL=2, save h_V after encoder too
        if (self.nfl != 1) and self.single_res_rec:
            all_hidden.append(h_V)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        # 0 for visible chains, 1 for masked chains
        chain_M = chain_M*mask  # update chain_M to include missing regions

        decoding_order = torch.argsort((chain_M+0.0001)*(torch.abs(torch.randn(chain_M.shape, device=device)))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp', (1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)

        if self.decoding_order == 'srr': # set ALL residues EXCEPT target to be visible
            order_mask_backward = torch.ones_like(order_mask_backward, device=device) - torch.eye(order_mask_backward.shape[-1], device=device).unsqueeze(0).repeat(order_mask_backward.shape[0], 1, 1)
        elif self.decoding_order == 'id': # set ALL residues to be visible
            order_mask_backward = torch.ones_like(order_mask_backward, device=device)
        else: # autoregressive decoding (not used here)
            pass

        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        # mask contains info about missing residues etc
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)
        
        # Create neighbors of X
        E_idx_flat = E_idx.view((*E_idx.shape[:-2], -1))
        E_idx_flat = E_idx_flat[..., None, None].expand(-1, -1, *X.shape[-2:])
        X_neighbors = torch.gather(X, -3, E_idx_flat)
        X_neighbors = X_neighbors.view((*E_idx.shape, -1, 3))

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            # insert E_sc (side chain edges) into decoder inputs - no masking needed
            if self.side_chains:
                h_V = layer(h_V, h_ESV, mask, None, h_E_sc)
            else:
                if not self.use_ipmp:
                    h_V = layer(h_V, h_ESV, mask)
                else:
                    h_V = layer(h_V, h_ESV, E_idx, X_neighbors, mask)
            all_hidden.append(h_V)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)

        # return list(all_hidden), h_S, log_probs, h_E # TODO use if running nfl reversal mode
        return list(reversed(all_hidden)), h_S, log_probs, h_E


def get_protein_mpnn_sca(cfg):
    """Loading Pre-trained ProteinMPNN model for structure embeddings"""
    hidden_dim = cfg.model.proteinmpnn.hidden_dim
    enc_layers = cfg.model.proteinmpnn.num_encoder_layers
    dec_layers = cfg.model.proteinmpnn.num_decoder_layers

    side_chains = cfg.model.proteinmpnn.side_chains
    single_res_rec = cfg.model.proteinmpnn.single_res_rec
    decoding_order = cfg.model.proteinmpnn.decoding_order
    nfl = cfg.model.num_final_layers
    dropout = cfg.model.proteinmpnn.dropout if 'dropout' in cfg.model.proteinmpnn else 0.1

    checkpoint_path = cfg.model.proteinmpnn.ckpt_path

    print('Loading model %s', checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu') 

    model = ProteinMPNN(use_ipmp=False, num_letters=21, node_features=hidden_dim, 
                        edge_features=hidden_dim, hidden_dim=hidden_dim, num_encoder_layers=enc_layers, 
                        num_decoder_layers=dec_layers, k_neighbors=checkpoint['num_edges'], augment_eps=0.0, 
                        vocab=21, n_points=8, dropout=dropout,
                        side_chains=side_chains, single_res_rec=single_res_rec, decoding_order=decoding_order, nfl=nfl)

    if cfg.model.load_pretrained:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Skipping model weight overwrites!")
    
    if cfg.model.freeze_weights:
        model.eval()
        # freeze these weights for transfer learning
        for param in model.parameters():
            param.requires_grad = False

    return model