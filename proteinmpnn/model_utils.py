from __future__ import print_function
import json, time, os, sys, glob
import shutil
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import torch.utils
import torch.utils.checkpoint

import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import itertools

import sys
sys.path.append('/proj/kuhl_lab/users/dieckhau/ThermoMPNN/proteinmpnn')
from rigid_utils import Rigid


def featurize(batch, device, side_chains=-1):
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32) #sum of chain seq lengths
    L_max = max([len(b['seq']) for b in batch])

    # DONE henry set dimension vector to hold all atom types
    if side_chains == -1:
        atom_names = ["N", "CA", "C", "O"]
    else:
        atom_names = ["N", "CA", "C", "O", "SC1", "SC2", "SC3", "SC4", "SC5", "SC6", "SC7", "SC8", "SC9", "SC10"]
        atom_names = atom_names[:side_chains]

    X = np.zeros([B, L_max, len(atom_names), 3])

    residue_idx = -100*np.ones([B, L_max], dtype=np.int32) #residue idx with jumps across chains
    chain_M = np.zeros([B, L_max], dtype=np.int32) #1.0 for the bits that need to be predicted, 0.0 for the bits that are given
    mask_self = np.ones([B, L_max, L_max], dtype=np.int32) #for interface loss calculation - 0.0 for self interaction, 1.0 for other
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32) #integer encoding for chains 0, 0, 0,...0, 1, 1,..., 1, 2, 2, 2...
    S = np.zeros([B, L_max], dtype=np.int32) #sequence AAs integers
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_letters = init_alphabet + extra_alphabet
    for i, b in enumerate(batch):
        masked_chains = b['masked_list']
        visible_chains = b['visible_list']
        all_chains = masked_chains + visible_chains
        visible_temp_dict = {}
        masked_temp_dict = {}
        for step, letter in enumerate(all_chains):
            chain_seq = b[f'seq_chain_{letter}']
            if letter in visible_chains:
                visible_temp_dict[letter] = chain_seq
            elif letter in masked_chains:
                masked_temp_dict[letter] = chain_seq
        for km, vm in masked_temp_dict.items():
            for kv, vv in visible_temp_dict.items():
                if vm == vv:
                    if kv not in masked_chains:
                        masked_chains.append(kv)
                    if kv in visible_chains:
                        visible_chains.remove(kv)
        all_chains = masked_chains + visible_chains
        random.shuffle(all_chains) #randomly shuffle chain order
        num_chains = b['num_of_chains']
        mask_dict = {}
        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        c = 1
        l0 = 0
        l1 = 0
        for step, letter in enumerate(all_chains):
            if letter in visible_chains:
                chain_seq = b[f'seq_chain_{letter}']
                chain_length = len(chain_seq)
                chain_coords = b[f'coords_chain_{letter}'] #this is a dictionary
                chain_mask = np.zeros(chain_length) #0.0 for visible chains

                # shape: [chain_length, num_atoms, 3]
                x_chain = np.stack([chain_coords[c] for c in [f'{a}_chain_{letter}' for a in atom_names]], 1)

                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                residue_idx[i, l0:l1] = 100*(c-1)+np.arange(l0, l1)
                l0 += chain_length
                c+=1
            elif letter in masked_chains: 
                chain_seq = b[f'seq_chain_{letter}']
                chain_length = len(chain_seq)
                chain_coords = b[f'coords_chain_{letter}'] #this is a dictionary
                chain_mask = np.ones(chain_length) #0.0 for visible chains
                
                # shape: [chain_length, num_atoms, 3]
                x_chain = np.stack([chain_coords[c] for c in [f'{a}_chain_{letter}' for a in atom_names]], 1)

                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                residue_idx[i, l0:l1] = 100*(c-1)+np.arange(l0, l1)
                l0 += chain_length
                c+=1
        x = np.concatenate(x_chain_list,0) #[L, 4, 3]
        all_sequence = "".join(chain_seq_list)
        m = np.concatenate(chain_mask_list,0) #[L,], 1.0 for places that need to be predicted
        chain_encoding = np.concatenate(chain_encoding_list,0)

        l = len(all_sequence)
        x_pad = np.pad(x, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
        X[i,:,:,:] = x_pad

        m_pad = np.pad(m, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        chain_M[i,:] = m_pad

        chain_encoding_pad = np.pad(chain_encoding, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        chain_encoding_all[i,:] = chain_encoding_pad

        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in all_sequence], dtype=np.int32)
        S[i, :l] = indices

    # isnan = np.isnan(X)

    if side_chains == -1:
        mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    else:
        # need to specify ONLY backbone atoms or else no residue will exist (except TRP)
        mask = np.isfinite(np.sum(X[:, :, 0:4, :], (2, 3))).astype(np.float32)

    # removed so that mask_per_atom can be gathered in forward fxn instead
    # X[isnan] = 0.

    # Conversion
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long,device=device)
    S = torch.from_numpy(S).to(dtype=torch.long,device=device)
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    mask_self = torch.from_numpy(mask_self).to(dtype=torch.float32, device=device)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32, device=device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(dtype=torch.long, device=device)
    return X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all


def loss_nll(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    S_argmaxed = torch.argmax(log_probs,-1) #[B, L]
    true_false = (S == S_argmaxed).float()
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av, true_false


def loss_smoothed(S, log_probs, mask, weight=0.1):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(S, 21).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / 2000.0 #fixed 
    return loss, loss_av


def get_bb_frames(N, CA, C):
    return Rigid.from_3_points(N, CA, C, fixed=True)


# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


def get_virtual_cbeta(X):
    """Calculate virtual Cb from ideal bond lengths and angles."""
    b = X[:, :, 1, :] - X[:, :, 0, :]
    c = X[:, :, 2, :] - X[:, :, 1, :]
    a = torch.cross(b, c, dim=-1)
    Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
    return Cb


def gather_mask(E_idx, mask1, mask2):
    """Gather atom-wise mask using neighbor indices to get 2D mask."""
    # make pairwise 2D mask
    mask_2D = torch.unsqueeze(mask1, 1) * torch.unsqueeze(mask2, 2)
    # torch gather - take values from specific dim(s) of E_idx as index for condensing mask_2D
    gathered = torch.gather(mask_2D, dim=1, index=E_idx)
    return gathered


class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


class IPMPEncoder(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, n_points=8):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.n_points = n_points
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.points_fn_node = nn.Linear(num_hidden, n_points * 3)
        self.points_fn_edge = nn.Linear(num_hidden, n_points * 3)
        self.W1 = nn.Linear(num_hidden + num_in + 9 * n_points, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in + 9 * n_points, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def _get_message_input(self, h_V, h_E, E_idx, X, edge=False):
        # Get backbone global frames from N, CA, and C
        bb_to_global = get_bb_frames(X[..., 0, :], X[..., 1, :], X[..., 2, :])

        # Generate points in local frame of each node
        if not edge:
            p_local = self.points_fn_node(h_V).reshape((*h_V.shape[:-1], self.n_points, 3))
        else:
            p_local = self.points_fn_edge(h_V).reshape((*h_V.shape[:-1], self.n_points, 3))

        # Project points into global frame
        p_global = bb_to_global[..., None].apply(p_local)
        p_global_expand = p_global.unsqueeze(-3).expand(*E_idx.shape, *p_global.shape[-2:])

        # Get neighbor points in global frame for each node
        neighbor_idx = E_idx.view((*E_idx.shape[:-2], -1))
        neighbor_p_global = torch.gather(p_global, -3, neighbor_idx[..., None, None].expand(*neighbor_idx.shape, self.n_points, 3))
        neighbor_p_global = neighbor_p_global.view(*E_idx.shape, self.n_points, 3)

        # Form message components:
        # 1. Node i's local points
        p_local_expand = p_local.unsqueeze(-3).expand(*E_idx.shape, *p_local.shape[-2:])

        # 2. Distance between node i's local points and its CA
        p_local_norm = torch.sqrt(torch.sum(p_local_expand ** 2, dim=-1) + 1e-8)

        # 3. Node j's points in i's local frame
        neighbor_p_local = bb_to_global[..., None, None].invert_apply(neighbor_p_global)

        # 4. Distance between node j's points in i's local frame and i's CA
        neighbor_p_local_norm = torch.sqrt(torch.sum(neighbor_p_local ** 2, dim=-1) + 1e-8)

        # 5. Distance between node i's global points and node j's global points
        neighbor_p_global_norm = torch.sqrt(
            torch.sum(
                (p_global_expand - neighbor_p_global) ** 2,
                dim=-1) + 1e-8)

        # Node message
        node_expand = h_V.unsqueeze(-2).expand(*E_idx.shape, h_V.shape[-1])
        neighbor_edge = cat_neighbors_nodes(h_V, h_E, E_idx)
        message_in = torch.cat(
            (node_expand,
             neighbor_edge,
             p_local_expand.view((*E_idx.shape, -1)),
             p_local_norm,
             neighbor_p_local.view((*E_idx.shape, -1)),
             neighbor_p_local_norm,
             neighbor_p_global_norm), dim=-1)

        return message_in

    def forward(self, h_V, h_E, E_idx, X, mask_V=None, mask_attend=None):
        # Get message fn input for node message
        message_in = self._get_message_input(h_V, h_E, E_idx, X)

        # Update nodes
        node_m = self.W3(self.act(self.W2(self.act(self.W1(message_in)))))
        if mask_attend is not None:
            node_m = mask_attend.unsqueeze(-1) * node_m
        node_m = torch.mean(node_m, dim=-2)
        h_V = self.norm1(h_V + self.dropout1(node_m))
        node_m = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(node_m))
        if mask_V is not None:
            h_V = mask_V.unsqueeze(-1) * h_V

        # Get message fn input for edge message
        message_in = self._get_message_input(h_V, h_E, E_idx, X, edge=True)
        edge_m = self.W13(self.act(self.W12(self.act(self.W11(message_in)))))
        h_E = self.norm3(h_E + self.dropout3(edge_m))

        return h_V, h_E
        

class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayer, self).__init__()
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

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class IPMPDecoder(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, n_points=8):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.n_points = n_points
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.points_fn = nn.Linear(num_hidden, n_points * 3)
        self.W1 = nn.Linear(num_hidden + num_in + 9 * n_points, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def _get_message_input(self, h_V, h_E, E_idx, X_neighbors):
        # Get backbone global frames from N, CA, and C
        bb_to_global = get_bb_frames(X_neighbors[..., 0, :], X_neighbors[..., 1, :], X_neighbors[..., 2, :])

        # Generate points in local frame of each node as its updated
        # This means getting the node from the back of h_E concatenation
        h_V_prev = h_E[..., -self.num_hidden:]
        p_local_neighbor = self.points_fn(h_V_prev).reshape((*h_V_prev.shape[:-1], self.n_points, 3))
        p_local = p_local_neighbor[..., 0, :, :]

        # Project points into global frame
        neighbor_p_global = bb_to_global[..., None].apply(p_local_neighbor)
        p_global = neighbor_p_global[..., 0, :, :]
        p_global_expand = p_global.unsqueeze(-3).expand(*E_idx.shape, *p_global.shape[-2:])

        # Form message components:
        # 1. Node i's local points
        p_local_expand = p_local.unsqueeze(-3).expand(*E_idx.shape, *p_local.shape[-2:])

        # 2. Distance between node i's local points and its CA
        p_local_norm = torch.sqrt(torch.sum(p_local_expand ** 2, dim=-1) + 1e-8)

        # 3. Node j's points in i's local frame
        neighbor_p_local = bb_to_global[..., 0][..., None, None].invert_apply(neighbor_p_global)

        # 4. Distance between node j's points in i's local frame and i's CA
        neighbor_p_local_norm = torch.sqrt(torch.sum(neighbor_p_local ** 2, dim=-1) + 1e-8)

        # 5. Distance between node i's global points and node j's global points
        neighbor_p_global_norm = torch.sqrt(
            torch.sum(
                (p_global_expand - neighbor_p_global) ** 2,
                dim=-1) + 1e-8)

        # Node message
        node_expand = h_V.unsqueeze(-2).expand(*E_idx.shape, h_V.shape[-1])
        message_in = torch.cat(
            (node_expand,
             h_E,
             p_local_expand.view((*E_idx.shape, -1)),
             p_local_norm,
             neighbor_p_local.view((*E_idx.shape, -1)),
             neighbor_p_local_norm,
             neighbor_p_global_norm), dim=-1)

        return message_in
        
        
    def forward(self, h_V, h_E, E_idx, X_neighbors, mask_V=None, mask_attend=None):
        # Get message fn input for node message
        message_in = self._get_message_input(h_V, h_E, E_idx, X_neighbors)

        # Update nodes
        node_m = self.W3(self.act(self.W2(self.act(self.W1(message_in)))))
        if mask_attend is not None:
            node_m = mask_attend.unsqueeze(-1) * node_m
        node_m = torch.mean(node_m, dim=-2)
        h_V = self.norm1(h_V + self.dropout1(node_m))
        node_m = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(node_m))
        if mask_V is not None:
            h_V = mask_V.unsqueeze(-1) * h_V

        return h_V
    

class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()
    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h

class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2*max_relative_feature+1+1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0, 2*self.max_relative_feature)*mask + (1-mask)*(2*self.max_relative_feature+1)
        d_onehot = torch.nn.functional.one_hot(d, 2*self.max_relative_feature+1+1)
        E = self.linear(d_onehot.float())
        return E


class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30, augment_eps=0., num_chain_embeddings=16, side_chains=-1):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.side_chains = side_chains

        num_atom_combos = 25
        if self.side_chains != -1:
            num_atom_combos = self.side_chains ** 2
        
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        node_in, edge_in = 6, num_positional_embeddings + num_rbf * num_atom_combos
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def _get_rbf_sca(self, A, B, E_idx, gathered_mask):
        """Calculate RBF after masking out non-existent atoms in side chains"""
        D_A_B = torch.sqrt(torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6)  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[:, :, :, 0]  # [B,L,K]
        # mask distances by does-exist mask of neighbors
        D_A_B_neighbors = D_A_B_neighbors * gathered_mask.int()
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, X, mask, residue_idx, chain_labels):
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
        RBF_all = []
        num_atoms = X.shape[2]

        if self.side_chains == -1:
            # get Cb and add to X array
            Cb = torch.unsqueeze(get_virtual_cbeta(X), dim=-2)
            X = torch.concatenate([X, Cb], dim=2)
            # we only need to find neighbors once
            D_neighbors, E_idx = self._dist(X[:, :, 1, :], mask)
            # make RBF of every possible atom combo
            for c1 in range(num_atoms + 1):
                for c2 in range(num_atoms + 1):
                    RBF_all.append(self._get_rbf(X[:, :, c1, :], X[:, :, c2, :], E_idx))
            X = X[:, :, :-1, :]  # drop Cb for IPMP calc, if needed

        else:  # side-chains enabled
            D_neighbors, E_idx = self._dist(X[:, :, 1, :], mask[:, :, 1])
            for c1 in range(num_atoms):
                for c2 in range(num_atoms):
                    gathered_mask = gather_mask(E_idx, mask[:, :, c1], mask[:, :, c2])
                    print(c1, c2, '&&&')
                    print(torch.sum(mask[:, :, c1]), mask[:, :, c1].shape)
                    print(torch.sum(mask[:, :, c2]), mask[:, :, c2].shape)
                    print(torch.sum(gathered_mask), gathered_mask.shape)

                    RBF_all.append(self._get_rbf_sca(X[:, :, c1, :], X[:, :, c2, :], E_idx, gathered_mask))

        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        # print('RBF:', RBF_all.shape)
        offset = residue_idx[:,:,None]-residue_idx[:,None,:]
        offset = gather_edges(offset[:,:,:,None], E_idx)[:,:,:,0] #[B, L, K]

        d_chains = ((chain_labels[:, :, None] - chain_labels[:,None,:])==0).long() #find self vs non-self interaction
        E_chains = gather_edges(d_chains[:,:,:,None], E_idx)[:,:,:,0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx, X


class ProteinMPNN(nn.Module):
    def __init__(self, num_letters=21, node_features=128, edge_features=128,
        hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
        vocab=21, k_neighbors=32, augment_eps=0.1, dropout=0.1,
        use_ipmp=False, n_points=8, side_chains=-1):
        super(ProteinMPNN, self).__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.use_ipmp = use_ipmp
        self.side_chains = side_chains

        self.features = ProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps, side_chains=self.side_chains)

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

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
        if not self.side_chains:
            X = torch.nan_to_num(X, nan=0.0)
            E, E_idx, X = self.features(X, mask, residue_idx, chain_encoding_all)
        else:
            mask_per_atom = torch.isnan(X)[:, :, :, 0].long()  # different num. of side chains exist for different residues
            # mask_per_atom is shape [B, L_max, 14]
            X = torch.nan_to_num(X, nan=0.0)
            E, E_idx, X = self.features(X, mask_per_atom, residue_idx, chain_encoding_all)

        # print('**', E.shape, E_idx.shape, X.shape)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            if not self.use_ipmp:
                h_V, h_E = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend)
            else:
                h_V, h_E = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, X, mask, mask_attend)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        chain_M = chain_M*mask  # update chain_M to include missing regions

        decoding_order = torch.argsort((chain_M+0.0001)*(torch.abs(torch.randn(chain_M.shape, device=device)))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp', (1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)

        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
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
            if not self.use_ipmp:
                h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_ESV, mask)
            else:
                h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_ESV, E_idx, X_neighbors, mask)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

    def sample(self, X, randn, S_true, chain_mask, chain_encoding_all, residue_idx, mask=None, temperature=1.0):
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx, X = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            if not self.use_ipmp:
                h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
            else:
                h_V, h_E = layer(h_V, h_E, E_idx, X, mask, mask_attend)

        # Decoder uses masked self-attention
        chain_mask = chain_mask*mask #update chain_M to include missing regions

        # if self.single_res_rec:  # new decoding order (all except current AA known)
        #     # need to handle batched inputs - multiple proteins of varied sizes
        #     decoding_order = torch.tensor([list(range(X.size(1)))] * X.shape[0], device=device)
        #     mask_size = E_idx.shape[1]
        #     permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        #     order_mask_backward = torch.einsum('ij, biq, bjp->bqp',
        #                                        (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
        #                                        permutation_matrix_reverse, permutation_matrix_reverse)  # [1, L_max, L_max] array of visibility ordered backward
        #     # set ALL residues visible EXCEPT current one
        #     order_mask_backward = torch.ones_like(order_mask_backward, device=device) - torch.eye(order_mask_backward.shape[1], order_mask_backward.shape[1], device=device).repeat(X.shape[0], 1, 1)

        # else:
        decoding_order = torch.argsort((chain_mask+0.0001)*(torch.abs(randn))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)

        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        N_batch, N_nodes = X.size(0), X.size(1)
        log_probs = torch.zeros((N_batch, N_nodes, 21), device=device)
        all_probs = torch.zeros((N_batch, N_nodes, 21), device=device, dtype=torch.float32)
        h_S = torch.zeros_like(h_V, device=device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V, device=device) for _ in range(len(self.decoder_layers))]

        # Create neighbors of X
        E_idx_flat = E_idx.view((*E_idx.shape[:-2], -1))
        E_idx_flat = E_idx_flat[..., None, None].expand(-1, -1, *X.shape[-2:])
        X_neighbors = torch.gather(X, -3, E_idx_flat)
        X_neighbors = X_neighbors.view((*E_idx.shape, -1, 3))

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for t_ in range(N_nodes):
            t = decoding_order[:,t_] #[B]
            chain_mask_gathered = torch.gather(chain_mask, 1, t[:,None]) #[B]
            mask_gathered = torch.gather(mask, 1, t[:,None]) #[B]
            if (mask_gathered==0).all(): #for padded or missing regions only
                S_t = torch.gather(S_true, 1, t[:,None])
            else:
                # Hidden layers
                if self.use_ipmp:
                    X_neighbors_t = torch.gather(X_neighbors, 1, t[:,None,None,None,None].repeat(1,1,*X_neighbors.shape[-3:]))
                E_idx_t = torch.gather(E_idx, 1, t[:,None,None].repeat(1,1,E_idx.shape[-1]))
                h_E_t = torch.gather(h_E, 1, t[:,None,None,None].repeat(1,1,h_E.shape[-2], h_E.shape[-1]))
                h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                h_EXV_encoder_t = torch.gather(h_EXV_encoder_fw, 1, t[:,None,None,None].repeat(1,1,h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1]))
                mask_t = torch.gather(mask, 1, t[:,None])
                for l, layer in enumerate(self.decoder_layers):
                    # Updated relational features for future states
                    h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                    h_V_t = torch.gather(h_V_stack[l], 1, t[:,None,None].repeat(1,1,h_V_stack[l].shape[-1]))
                    h_ESV_t = torch.gather(mask_bw, 1, t[:,None,None,None].repeat(1,1,mask_bw.shape[-2], mask_bw.shape[-1])) * h_ESV_decoder_t + h_EXV_encoder_t
                    if not self.use_ipmp:
                        h_V_stack[l+1].scatter_(1, t[:,None,None].repeat(1,1,h_V.shape[-1]), layer(h_V_t, h_ESV_t, mask_V=mask_t))
                    else:
                        h_V_stack[l+1].scatter_(1, t[:,None,None].repeat(1,1,h_V.shape[-1]), layer(h_V_t, h_ESV_t, E_idx_t, X_neighbors_t, mask_V=mask_t))
                # Sampling step
                h_V_t = torch.gather(h_V_stack[-1], 1, t[:,None,None].repeat(1,1,h_V_stack[-1].shape[-1]))[:,0]

                if temperature <= 0.0:
                    logits = self.W_out(h_V_t)
                    probs = F.softmax(logits, dim=-1)
                    log_probs_t = F.log_softmax(logits, dim=-1)
                    S_t = torch.argmax(probs, dim=-1)
                else:
                    logits = self.W_out(h_V_t) / temperature
                    probs = F.softmax(logits, dim=-1)
                    log_probs_t = F.log_softmax(logits * temperature, dim=-1)
                    S_t = torch.multinomial(probs, 1)
                log_probs.scatter_(1, t[:, None, None].repeat(1, 1, 21), (chain_mask_gathered[:, :, None,] * log_probs_t[:, None, :]).float())
                all_probs.scatter_(1, t[:,None,None].repeat(1,1,21), (chain_mask_gathered[:,:,None,]*probs[:,None,:]).float())
            S_true_gathered = torch.gather(S_true, 1, t[:,None])
            S_t = (S_t*chain_mask_gathered+S_true_gathered*(1.0-chain_mask_gathered)).long()
            temp1 = self.W_s(S_t)
            h_S.scatter_(1, t[:,None,None].repeat(1,1,temp1.shape[-1]), temp1)
            S.scatter_(1, t[:,None], S_t)
        output_dict = {"S": S, "probs": all_probs, "decoding_order": decoding_order, "log_probs": log_probs}
        return output_dict


class ProteinMPNN_SRR(ProteinMPNN):
    """Version of ProteinMPNN that does single-residue recovery"""

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        """ Modified FWD function to do single-res prediction """
        device=X.device
        # Prepare node and edge embeddings
        E, E_idx, X = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            if not self.use_ipmp:
                h_V, h_E = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend)
            else:
                h_V, h_E = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, X, mask, mask_attend)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        chain_M = chain_M*mask  # update chain_M to include missing regions

        decoding_order = torch.tensor([list(range(X.size(1)))] * X.shape[0], device=device)
        # decoding_order = torch.argsort((chain_M + 0.0001) * (torch.abs(torch.randn(chain_M.shape, device=device))))  # [numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',
                                           (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
                                           permutation_matrix_reverse, permutation_matrix_reverse)    #     # TODO accumulate log probs from each position
        log_probs_all = torch.zeros((X.shape[0], X.shape[1], 21), device=device)


        for dec_idx in decoding_order[0]:
            # iterating over decoding order - need to decode each residue separately to prevent leakage
            print(dec_idx)
            order_mask_backward = torch.ones_like(order_mask_backward, device=device)
            order_mask_backward[..., :, dec_idx] = 0

            print(order_mask_backward.shape, '***', E_idx.shape, mask.shape)
            # TODO crop encoder inputs to ONLY decode one residue?

            mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
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
                if not self.use_ipmp:
                    h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_ESV, mask)
                else:
                    h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_ESV, E_idx, X_neighbors, mask)

            # TODO fix this to avoid memory issues ?
            logits = self.W_out(h_V)
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs_all[..., dec_idx, :] = log_probs[..., dec_idx, :]


        return log_probs_all



class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

def get_std_opt(parameters, d_model, step):
    return NoamOpt(
        d_model, 2, 4000, torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9), step
    )
