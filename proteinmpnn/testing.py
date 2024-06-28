import argparse

import numpy as np
import torch
import queue
import copy
import random
import os.path
from concurrent.futures import ProcessPoolExecutor    
from tqdm import tqdm
from utils import worker_init_fn, get_pdbs, loader_pdb, build_training_clusters, PDB_dataset, StructureDataset, StructureLoader
from model_utils import featurize, loss_nll, ProteinMPNN


def main(args):


    if args.seed is not None:
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    
    ckpt_path = args.ckpt_path
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    num_edges = ckpt['num_edges']

    data_path = args.path_for_training_data
    params = {
        "LIST"    : f"{data_path}/list.csv", 
        "VAL"     : f"{data_path}/valid_clusters.txt",
        "TEST"    : f"{data_path}/test_clusters.txt",
        "DIR"     : f"{data_path}",
        "DATCUT"  : "2030-Jan-01",
        "RESCUT"  : args.rescut, #resolution cutoff for PDBs
        "HOMO"    : 0.70 #min seq.id. to detect homo chains
    }

    LOAD_PARAM = {'batch_size': 1,
                  'shuffle': True,
                  'pin_memory': False,
                  'num_workers': 8}

    train, valid, test = build_training_clusters(params, False)

    test_set = PDB_dataset(list(test.keys()), loader_pdb, test, params)
    test_loader = torch.utils.data.DataLoader(test_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)

    model = ProteinMPNN(node_features=args.hidden_dim, 
                        edge_features=args.hidden_dim, 
                        hidden_dim=args.hidden_dim, 
                        num_encoder_layers=args.num_encoder_layers, 
                        num_decoder_layers=args.num_decoder_layers, 
                        k_neighbors=num_edges, 
                        dropout=0.0, 
                        augment_eps=0.0,
                        use_ipmp=args.use_ipmp,
                        n_points=args.n_points, 
                        single_res_rec=args.single_res_rec, 
                        side_chains=args.side_chains)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    print(f'Number of parameters: {sum([p.numel() for p in model.parameters()])}')

    with ProcessPoolExecutor(max_workers=12) as executor:
        q = queue.Queue(maxsize=3)
        for i in range(3):
            q.put_nowait(executor.submit(get_pdbs, test_loader, 1, args.max_protein_length, args.num_examples_per_epoch, args.side_chains))
        pdb_dict_test = q.get().result()
       
        dataset_test = StructureDataset(pdb_dict_test, truncate=None, max_length=args.max_protein_length)        
        loader_test = StructureLoader(dataset_test, batch_size=args.batch_size)

        reload_c = 0
        perplexity_list = []
        centrality_accuracy = {'all': [], 'core': [], 'surface': [], 'interface-core': [], 'interface-rim': [], 'monomer': [], 'oligomer': []}
        aatype_accuracy = {aatype: [] for aatype in 'ACDEFGHIKLMNPQRSTVWY'}
        for r in range(args.num_repeats):
            if r % args.reload_data_every_n_repeats == 0:
                if reload_c != 0:
                    pdb_dict_test = q.get().result()
                    dataset_test = StructureDataset(pdb_dict_test, truncate=None, max_length=args.max_protein_length)
                    loader_test = StructureLoader(dataset_test, batch_size=args.batch_size)
                    q.put_nowait(executor.submit(get_pdbs, test_loader, 1, args.max_protein_length, args.num_examples_per_epoch, args.side_chains))
                reload_c += 1
            
            with torch.no_grad():
                test_sum, test_weights = 0., 0.
                test_acc = {'all': [0., 0], 'core': [0., 0], 'surface': [0., 0], 'interface-core': [0., 0], 'interface-rim': [0., 0], 'monomer': [0., 0], 'oligomer': [0., 0]}
                aatype_acc = {aatype: [0., 0] for aatype in aatype_accuracy}
                for _, batch in tqdm(enumerate(loader_test)):
                    X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device, args.side_chains)
                    
                    randn = torch.randn(chain_M.shape, device=device)

                    # separate (one-shot) sample fxn for single residue decoding
                    if args.single_res_rec:
                        sample_out = model.sample_SRR(X, randn, S, chain_M, chain_encoding_all, residue_idx, mask, temperature=args.temperature)
                    else:
                        sample_out = model.sample(X, randn, S, chain_M, chain_encoding_all, residue_idx, mask, temperature=args.temperature)
   
                    mask_for_loss = mask*chain_M

                    loss, _, _ = loss_nll(S, sample_out["log_probs"], mask_for_loss)
                    true_false = (S == sample_out['S']).float()
                    
                    test_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                    test_weights += torch.sum(mask_for_loss).cpu().data.numpy()

                    # Accumulate accuracy for all residues
                    test_acc['all'][0] += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                    test_acc['all'][1] += torch.sum(mask_for_loss).cpu().data.numpy()

                    # Impute CB for all residues
                    X = torch.nan_to_num(X, nan=0.0)
                    
                    b = X[..., 1, :] - X[..., 0, :]
                    c = X[..., 2, :] - X[..., 1, :]
                    a = torch.cross(b, c, dim=-1)
                    Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[..., 1, :]

                    # Compute distances and number of neighbors <= 10A
                    in_sphere = torch.cdist(Cb, Cb) < 10.0
                    in_sphere *= (mask[..., :, None] * mask[..., None, :]).bool()
                    num_neighbors = torch.sum(in_sphere, dim=-1) - 1

                    # Accumulate accuracy for core residues
                    centrality_mask = num_neighbors >= 20
                    test_acc['core'][0] += torch.sum(true_false * mask_for_loss * centrality_mask).cpu().data.numpy()
                    test_acc['core'][1] += torch.sum(mask_for_loss * centrality_mask).cpu().data.numpy()

                    # Accumulate accuracy for surface residues
                    centrality_mask = num_neighbors <= 15
                    test_acc['surface'][0] += torch.sum(true_false * mask_for_loss * centrality_mask).cpu().data.numpy()
                    test_acc['surface'][1] += torch.sum(mask_for_loss * centrality_mask).cpu().data.numpy()

                    # Determine all the per-residue accuracy
                    for aatype in aatype_acc:
                        aatype_mask = S == list('ACDEFGHIKLMNPQRSTVWY').index(aatype)
                        aatype_acc[aatype][0] += torch.sum(true_false * mask_for_loss * aatype_mask).cpu().data.numpy()
                        aatype_acc[aatype][1] += torch.sum(mask_for_loss * aatype_mask).cpu().data.numpy()

                    # Mask out intra-chain neighbors
                    icn_mask = ~(chain_encoding_all[:, None, :] == chain_encoding_all[:, :, None]) # [B, L_max, L_max]
                    in_sphere *= (icn_mask).bool()
                    num_neighbors = torch.sum(in_sphere, dim=-1) - 1

                    # more than 10 nearby cross-chain - "Interface Core"
                    centrality_mask = num_neighbors >= 10
                    test_acc['interface-core'][0] += torch.sum(true_false * mask_for_loss * centrality_mask).cpu().data.numpy()
                    test_acc['interface-core'][1] += torch.sum(mask_for_loss * centrality_mask).cpu().data.numpy()

                    # more than 5 nearby cross-chain - "Interface Rim"
                    centrality_mask = num_neighbors >= 5
                    test_acc['interface-rim'][0] += torch.sum(true_false * mask_for_loss * centrality_mask).cpu().data.numpy()
                    test_acc['interface-rim'][1] += torch.sum(mask_for_loss * centrality_mask).cpu().data.numpy()
                    
                    # stratify by monomer/oligomer state
                    oligomer_mask = torch.zeros_like(centrality_mask)
                    for b in range(chain_encoding_all.shape[0]): # can't do this easily w/vectorized b/c jagged arrays aren't permitted
                        ch_unique = torch.unique(chain_encoding_all[b, :])
                        ch_unique = ch_unique[ch_unique > 0].numel() # zeros only come from array padding, not real chains
                        if ch_unique > 1:
                            oligomer_mask[b, :] = 1

                    oligomer_mask = oligomer_mask.bool()
                    test_acc['oligomer'][0] += torch.sum(true_false * mask_for_loss * oligomer_mask).cpu().data.numpy()
                    test_acc['oligomer'][1] += torch.sum(mask_for_loss * oligomer_mask).cpu().data.numpy()

                    test_acc['monomer'][0] += torch.sum(true_false * mask_for_loss * ~oligomer_mask).cpu().data.numpy()
                    test_acc['monomer'][1] += torch.sum(mask_for_loss * ~oligomer_mask).cpu().data.numpy()
                    
            test_loss = test_sum / test_weights
            test_perplexity = np.exp(test_loss)
            perplexity_list.append(test_perplexity)

            for centrality in centrality_accuracy:
                centrality_accuracy[centrality].append(test_acc[centrality][0] / test_acc[centrality][1])

            for aatype in aatype_accuracy:
                aatype_accuracy[aatype].append(aatype_acc[aatype][0] / aatype_acc[aatype][1])

        print("Perplexity:")
        print(f"\t{np.mean(perplexity_list):.5f} +- {np.std(perplexity_list):.5f}")
        print("Sequence Recovery:")
        for centrality in centrality_accuracy:
            print(f"\t{centrality}: {np.mean(centrality_accuracy[centrality]):.5f} +- {np.std(centrality_accuracy[centrality]):.5f}")
        for aatype in aatype_accuracy:
            print(f"\t{aatype}: {np.mean(aatype_accuracy[aatype]):.5f} +- {np.std(aatype_accuracy[aatype]):.5f}")

        
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--path_for_training_data", type=str, default="my_path/pdb_2021aug02", help="path for loading training data")
    argparser.add_argument("--ckpt_path", type=str, default="../vanilla_model_weights/v_48_010.pt", help="path for previous model weights, e.g. file.pt")
    argparser.add_argument("--num_repeats", type=int, default=3, help="number of testing repeats for variance calculations")
    argparser.add_argument("--reload_data_every_n_repeats", type=int, default=1, help="how often to reload data")
    argparser.add_argument("--num_examples_per_epoch", type=int, default=1000000, help="number of training example to load for one epoch")
    argparser.add_argument("--batch_size", type=int, default=10000, help="number of tokens for one batch")
    argparser.add_argument("--max_protein_length", type=int, default=10000, help="maximum length of the protein complext")
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers") 
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--use_ipmp", action="store_true", help="use ipmp layers instead of mpnn layers for message passing.")
    argparser.add_argument("--n_points", type=int, default=8, help="number of points for IPMP")
    argparser.add_argument("--temperature", type=float, default=1.0, help="temperature for sampling")
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument("--seed", type=int, default=None)
    argparser.add_argument("--single_res_rec", type=bool, default=False, help="Enable single residue recovery mode (use fwd fxn for testing)")
    argparser.add_argument("--side_chains", type=bool, default=False, help="Enable side chain edge features")
    argparser.add_argument("--logits", type=str, default='logits.pt', help='logits file to save')

    args = argparser.parse_args()    
    main(args)   
