import argparse
import os
import torch
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import pandas as pd

from thermompnn.trainer.v2_trainer import TransferModelPLv2
from thermompnn.train_thermompnn import parse_cfg

from protein_mpnn_utils import alt_parse_PDB
from thermompnn.datasets.v2_datasets import tied_featurize_mut
from thermompnn.datasets.dataset_utils import Mutation


def get_config(mode):
    if mode == 'single':
        config = {
            'model':
            {
                'hidden_dims': [64, 32], 
                'subtract_mut': True, 
                'num_final_layers': 2,
                'freeze_weights': True, 
                'load_pretrained': True, 
                'lightattn': True
            }, 
            'platform': 
            {
                'thermompnn_dir': '/home/hdieckhaus/scripts/ThermoMPNN/'
            }
        }
    elif mode == 'double':
        config = {
            'model':
            {
                'hidden_dims': [64, 32], 
                'subtract_mut': False,
                'single_target': True, 
                'mutant_embedding': True,
                'num_final_layers': 2,
                'freeze_weights': True, 
                'load_pretrained': True, 
                'lightattn': True, 
                'aggregation': 'max', 
                'dropout': 0.
            }, 
            'platform': 
            {
                'thermompnn_dir': '/home/hdieckhaus/scripts/ThermoMPNN/'
            }
        }

    config = OmegaConf.create(config)
    return parse_cfg(config)


def get_ssm_mutations_single(pdb):
        # make mutation list for SSM run
    mutation_list = []
    ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
    MUT_POS, MUT_WT = [], []
    for seq_pos in range(len(pdb['seq'])):
        wtAA = pdb['seq'][seq_pos]
        # check for missing residues
        if wtAA != '-':
            for i in range(20):
                MUT_POS.append(seq_pos)
                MUT_WT.append(ALPHABET.index(wtAA))

    plen = len(MUT_POS) // 20
    # MUT_POS
    MUT_POS = np.array(MUT_POS)
    MUT_POS = torch.tensor(MUT_POS).unsqueeze(-1)

    # MUT_WT
    MUT_WT = np.array(MUT_WT)
    MUT_WT = torch.tensor(MUT_WT).unsqueeze(-1)

    # MUT_MUT
    MUT_MUT = np.arange(20)
    MUT_MUT = torch.tensor(MUT_MUT).unsqueeze(-1).repeat(plen, 1)

    return MUT_POS, MUT_WT, MUT_MUT

def get_ssm_mutations_double(pdb):
    # make mutation list for SSM run
    mutation_list = []
    ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
    MUT_POS, MUT_WT = [], []
    for seq_pos in range(len(pdb['seq'])):
        wtAA = pdb['seq'][seq_pos]
        # check for missing residues
        if wtAA != '-':
            MUT_POS.append(seq_pos)
            MUT_WT.append(wtAA)
        else:
            MUT_WT.append('-')

    # get all positional combos except self-combos
    from itertools import product
    pos_combos = [p for p in product(MUT_POS, MUT_POS) if p[0] != p[1]]
    pos_combos = np.array(pos_combos) # [combos, 2]
    wtAA = np.zeros_like(pos_combos)
    # fill in wtAA for each pos combo
    for p_idx in range(pos_combos.shape[0]):
        wtAA[p_idx, 0] = ALPHABET.index(MUT_WT[pos_combos[p_idx, 0]])
        wtAA[p_idx, 1] = ALPHABET.index(MUT_WT[pos_combos[p_idx, 1]])

    # make default mutAA bundle for broadcasting
    one = np.arange(20).repeat(20)
    two = np.tile(np.arange(20), 20)
    mutAA = np.stack([one, two]).T # [400, 2]

    n_comb = pos_combos.shape[0]
    # tile each to match [combos * 400, 2] shape
    mutAA = np.tile(mutAA, (n_comb, 1))
    wtAA = np.tile(wtAA, (400, 1))
    pos_combos = np.tile(pos_combos, (400, 1))

    # filter out self-mutations and single-mutations
    mask = np.sum(mutAA == wtAA, -1).astype(bool)
    pos_combos = pos_combos[~mask, :]
    mutAA = mutAA[~mask, :]
    wtAA = wtAA[~mask, :]
    return torch.tensor(pos_combos), torch.tensor(wtAA), torch.tensor(mutAA)


def run_single(all_mpnn_hid, mpnn_embed, cfg, loader, args, model):
    """Batched mutation processing using shared protein embeddings and only stability prediction module head"""
    device = 'cuda'
    all_mpnn_hid = torch.cat(all_mpnn_hid[:cfg.model.num_final_layers], -1)
    embeds_all = [all_mpnn_hid, mpnn_embed]

    mpnn_embed = torch.cat(embeds_all, -1)
    mpnn_embed = mpnn_embed.repeat(args.batch_size, 1, 1)
    preds = []
    for b in tqdm(loader):
        pos, wtAA, mutAA = b
        pos = pos.to(device)
        wtAA = wtAA.to(device)
        mutAA = mutAA.to(device)
        mpnn_embed_tmp = torch.gather(mpnn_embed, 1, pos.unsqueeze(-1).expand(pos.size(0), pos.size(1), mpnn_embed.size(2)))
        mpnn_embed_tmp = torch.squeeze(mpnn_embed_tmp, 1) # final shape: (batch, embed_dim)

        if cfg.model.lightattn:
            mpnn_embed_tmp = torch.unsqueeze(mpnn_embed_tmp, -1)  # shape for LA input: (batch, embed_dim, seq_length=1)
            mpnn_embed_tmp = model.light_attention(mpnn_embed_tmp)  # shape for LA output: (batch, embed_dim)

        ddg = model.ddg_out(mpnn_embed_tmp)  # shape: (batch, 21)
        
        # index ddg outputs based on mutant AA indices
        if cfg.model.subtract_mut: # output is [B, L, 21]
            ddg = torch.gather(ddg, 1, mutAA) - torch.gather(ddg, 1, wtAA)
        elif cfg.model.single_target: # output is [B, L, 1]
            pass
        else:  # output is [B, L, 21]
            ddg = torch.gather(ddg, 1, mutAA)
                        
        preds += list(torch.squeeze(ddg, dim=-1).detach().cpu())

    return preds


def run_double(all_mpnn_hid, mpnn_embed, cfg, loader, args, model):
    """Batched mutation processing using shared protein embeddings and only stability prediction module head"""
    device = 'cuda'
    all_mpnn_hid = torch.cat(all_mpnn_hid[:cfg.model.num_final_layers], -1)
    embeds_all = [all_mpnn_hid, mpnn_embed]

    mpnn_embed = torch.cat(embeds_all, -1)
    mpnn_embed = mpnn_embed.repeat(args.batch_size, 1, 1)
    preds = []
    for b in tqdm(loader):
        pos, wtAA, mutAA = b
        pos = pos.to(device)
        wtAA = wtAA.to(device)
        mutAA = mutAA.to(device)
        mut_mutant_AAs = mutAA
        mut_wildtype_AAs = wtAA
        mut_positions = pos

        mut_embed_list = []
        for m in range(mut_mutant_AAs.shape[-1]):
            mut_embed = model.prot_mpnn.W_s(mut_mutant_AAs[:, m])
            mut_embed_list.append(mut_embed)
        mut_embed = torch.cat([m.unsqueeze(-1) for m in mut_embed_list], -1) # shape: (Batch, Embed, N_muts)

        if cfg.model.edges:  # add edges to input for gathering
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
                
        all_mpnn_embed = [] 
        for i in range(mut_mutant_AAs.shape[-1]):
            # gather embedding for a specific position
            current_positions = mut_positions[:, i:i+1] # shape: (B, 1])
            gathered_embed = torch.gather(mpnn_embed, 1, current_positions.unsqueeze(-1).expand(current_positions.size(0), current_positions.size(1), mpnn_embed.size(2)))
            gathered_embed = torch.squeeze(gathered_embed, 1) # final shape: (batch, embed_dim)
            # add specific mutant embedding to gathered embed based on which mutation is being gathered
            gathered_embed = torch.cat([gathered_embed, mut_embed[:, :, i]], -1)
            
            # cat to mpnn edges here if enabled
            if cfg.model.edges:
                gathered_embed = torch.cat([gathered_embed, mpnn_edges[:, :, i]], -1)

            all_mpnn_embed.append(gathered_embed)  # list with length N_mutations - used to make permutations

        for n, emb in enumerate(all_mpnn_embed):
            emb = torch.unsqueeze(emb, -1)  # shape for LA input: (batch, embed_dim, seq_length=1)
            emb = model.light_attention(emb)  # shape for LA output: (batch, embed_dim)
            all_mpnn_embed[n] = emb  # update list of embs

        all_mpnn_embed = torch.stack(all_mpnn_embed, dim=-1)  # shape: (batch, embed_dim, n_mutations)

        mask = (mut_mutant_AAs + mut_wildtype_AAs + mut_positions) == 0
        assert(torch.sum(mask[:, 0]) == 0)  # check that first mutation is ALWAYS visible
        mask = mask.unsqueeze(1).repeat(1, all_mpnn_embed.shape[1], 1)  # expand along embedding dimension

        all_mpnn_embed[mask] = -float("inf")
        mpnn_embed_tmp, _ = torch.max(all_mpnn_embed, dim=-1)

        ddg = model.ddg_out(mpnn_embed_tmp)  # shape: (batch, 21)
        
        # index ddg outputs based on mutant AA indices
        if cfg.model.subtract_mut: # output is [B, L, 21]
            ddg = torch.gather(ddg, 1, mutAA) - torch.gather(ddg, 1, wtAA)
        elif cfg.model.single_target: # output is [B, L, 1]
            pass
        else:  # output is [B, L, 21]
            ddg = torch.gather(ddg, 1, mutAA)
                        
        preds += list(torch.squeeze(ddg, dim=-1).detach().cpu())

    return preds


class SSMDataset(torch.utils.data.Dataset):
    def __init__(self, POS, WTAA, MUTAA):
        self.POS = POS
        self.WTAA = WTAA
        self.MUTAA = MUTAA

    def __len__(self):
        return self.POS.shape[0]
    
    def __getitem__(self, index):

        return self.POS[index, :], self.WTAA[index, :], self.MUTAA[index, :]

def main(args):

    cfg = get_config(args.mode)

    model = TransferModelPLv2.load_from_checkpoint(args.model, cfg=cfg).model
    model.eval()
    model.cuda()

    stime = time.time()
    # parse PDB
    if args.chain is not None:
        chains = [c for c in args.chain]
    else:
        chains = None
    pdb = alt_parse_PDB(args.pdb, input_chain_list=chains)
    pdb[0]['mutation'] = Mutation([0], ['A'], ['A'], [0.], '')

    # load SSM batches
    device = 'cuda'
    batch = tied_featurize_mut(pdb)
    X, S, mask, lengths, chain_M, chain_encoding_all, residue_idx, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask = batch

    X = X.to(device)
    S = S.to(device)
    mask = mask.to(device)
    lengths = torch.Tensor(lengths).to(device)
    chain_M = chain_M.to(device)
    chain_encoding_all = chain_encoding_all.to(device)
    residue_idx = residue_idx.to(device)
    mut_ddGs = mut_ddGs.to(device)

    # run SSM to reload mutation arrays
    if args.mode == 'single':
        MUT_POS, MUT_WT_AA, MUT_MUT_AA = get_ssm_mutations_single(pdb[0])
    elif args.mode == 'double':
        MUT_POS, MUT_WT_AA, MUT_MUT_AA = get_ssm_mutations_double(pdb[0])
    
    print('Running ThermoMPNN v2 on PDB %s of length %s' % (os.path.basename(args.pdb), str(len(pdb[0]['seq']))))
    dataset = SSMDataset(MUT_POS, MUT_WT_AA, MUT_MUT_AA)
    loader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=8)

    # run ProteinMPNN featurization
    X = torch.nan_to_num(X, nan=0.0)
    all_mpnn_hid, mpnn_embed, _, mpnn_edges = model.prot_mpnn(X, S, mask, chain_M, residue_idx, chain_encoding_all)
    
    # run batched predictions
    if args.mode == 'single':
        preds = run_single(all_mpnn_hid, mpnn_embed, cfg, loader, args, model)
    elif args.mode == 'double':
        preds = run_double(all_mpnn_hid, mpnn_embed, cfg, loader, args, model)

    etime = time.time()
    elapsed = etime - stime
    print('%s mutations processed in %s seconds with batch size %s' % (str(len(preds)), str(round(elapsed, 2)), str(args.batch_size)))

    # format output
    if args.mode == 'single':
        wt = torch.squeeze(MUT_WT_AA.detach().cpu())
        mut = torch.squeeze(MUT_MUT_AA.detach().cpu())
        ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
        w_pd, m_pd = [], []
        for w, m in zip(wt, mut):
            w_pd.append(ALPHABET[w])
            m_pd.append(ALPHABET[m])

        p_pd = np.array(preds)

        df = pd.DataFrame({
            'ddG (kcal/mol)': p_pd, 
            'Position': torch.squeeze(MUT_POS).detach().cpu() + 1,
            'wtAA': w_pd,
            'mutAA': m_pd
        })

    elif (args.mode == 'double') and (args.out is not None):
        wt1 = torch.squeeze(MUT_WT_AA[:, 0].detach().cpu())
        mut1 = torch.squeeze(MUT_MUT_AA[:, 0].detach().cpu())
        wt2 = torch.squeeze(MUT_WT_AA[:, 1].detach().cpu())
        mut2 = torch.squeeze(MUT_MUT_AA[:, 1].detach().cpu())
        ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
        w_pd, m_pd, w2_pd, m2_pd = [], [], [], []
        for w, m, w2, m2 in zip(wt1, mut1, wt2, mut2):
            w_pd.append(ALPHABET[w])
            m_pd.append(ALPHABET[m])
            w2_pd.append(ALPHABET[w2])
            m2_pd.append(ALPHABET[m2])

        p_pd = np.array(preds)

        df = pd.DataFrame({
            'ddG (kcal/mol)': p_pd, 
            'Position_1': torch.squeeze(MUT_POS[:, 0]).detach().cpu() + 1,
            'wtAA_1': w_pd,
            'mutAA_1': m_pd, 
            'Position_2': torch.squeeze(MUT_POS[:, 1]).detach().cpu() + 1,
            'wtAA_2': w2_pd, 
            'mutAA_2': m2_pd
        })
    if args.out is not None:
        df.to_csv(args.out, sep=',')

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model file to use for inference', default='./ckpt.ckpt')
    parser.add_argument('--mode', type=str, help='SSM mode to use', default='single')
    parser.add_argument('--pdb', type=str, help='PDB file to run', default='./2OCJ.pdb')
    parser.add_argument('--batch_size', type=int, help='batch size for stability prediction module', default=256)
    parser.add_argument('--out', type=str, help='output mutation CSV to save', default=None)
    parser.add_argument('--chain', type=str, help='chain(s) to use (default is None, will use all chains)', default=None)
    main(parser.parse_args())
