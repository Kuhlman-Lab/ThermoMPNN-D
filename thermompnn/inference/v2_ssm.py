import argparse
import torch
import numpy as np
from omegaconf import OmegaConf

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

    config = OmegaConf.create(config)
    return parse_cfg(config)


def get_ssm_mutations(pdb):
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


def main(args):

    cfg = get_config(args.mode)

    model = TransferModelPLv2.load_from_checkpoint(args.model, cfg=cfg).model
    model.eval()
    model.cuda()

    # parse PDB
    pdb = alt_parse_PDB(args.pdb)
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
    atom_mask = atom_mask.to(device)

    # run SSM to reload mutation arrays
    MUT_POS, MUT_WT_AA, MUT_MUT_AA = get_ssm_mutations(pdb[0])
    MUT_POS = MUT_POS.to(device)
    MUT_WT_AA = MUT_WT_AA.to(device)
    MUT_MUT_AA = MUT_MUT_AA.to(device)

    # batch_size = 256

    # # run ProteinMPNN featurization
    # X = torch.nan_to_num(X, nan=0.0)
    # all_mpnn_hid, mpnn_embed, _, mpnn_edges = model.prot_mpnn(X, S, mask, chain_M, residue_idx, chain_encoding_all)
    
    # # run batched predictions
    # all_mpnn_hid = torch.cat(all_mpnn_hid[:cfg.model.num_final_layers], -1)
    # embeds_all = [all_mpnn_hid, mpnn_embed]

    # mpnn_embed = torch.cat(embeds_all, -1)
    # mpnn_embed = mpnn_embed.repeat(MUT_POS.shape[0], 1, 1)

    # mpnn_embed = torch.gather(mpnn_embed, 1, MUT_POS.unsqueeze(-1).expand(MUT_POS.size(0), MUT_POS.size(1), mpnn_embed.size(2)))
    # mpnn_embed = torch.squeeze(mpnn_embed, 1) # final shape: (batch, embed_dim)

    # if cfg.model.lightattn:
    #     mpnn_embed = torch.unsqueeze(mpnn_embed, -1)  # shape for LA input: (batch, embed_dim, seq_length=1)
    #     mpnn_embed = model.light_attention(mpnn_embed)  # shape for LA output: (batch, embed_dim)

    # ddg = model.ddg_out(mpnn_embed)  # shape: (batch, 21)
        
    # # index ddg outputs based on mutant AA indices
    # if cfg.model.subtract_mut: # output is [B, L, 21]
    #     ddg = torch.gather(ddg, 1, MUT_MUT_AA) - torch.gather(ddg, 1, MUT_WT_AA)
    # elif cfg.model.single_target: # output is [B, L, 1]
    #     pass
    # else:  # output is [B, L, 21]
    #     ddg = torch.gather(ddg, 1, MUT_MUT_AA)
                    
    # print(ddg.shape)

    # format output

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model file to use for inference', default='./ckpt.ckpt')
    parser.add_argument('--mode', type=str, help='SSM mode to use', default='single')
    parser.add_argument('--pdb', type=str, help='PDB file to run', default='./2OCJ.pdb')
    main(parser.parse_args())
