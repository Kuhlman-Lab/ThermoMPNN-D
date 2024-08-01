import torch
import argparse
from omegaconf import OmegaConf
import pandas as pd
import os

from thermompnn.inference.v2_inference import load_v2_dataset, run_prediction_batched
from protein_mpnn_utils import ProteinMPNN
# from proteinmpnn.model_utils import ProteinMPNN
# from thermompnn.model.side_chain_model import ProteinMPNN
from thermompnn.trainer.v2_trainer import TransferModelPLv2
from train_thermompnn import parse_cfg


def inference(cfg, args):
    """Catch-all inference function for all ThermoMPNN versions"""
    cfg = parse_cfg(cfg)
    # pre-initialization params
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    ds_name = cfg.data.dataset

    ds = load_v2_dataset(cfg)
    print('Loading model %s' % args.model)

    # load ProteinMPNN baseline model (DEPENDS on config settings)
    # NOTE: this is the ThermoMPNN v2 version, NOT the updated side chain version (TODO set up this)
    if args.transfer:
        model = TransferModelPLv2.load_from_checkpoint(args.model, cfg=cfg, map_location=device).model
        
    else:
        ckpt_path = args.model
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        num_edges = ckpt['num_edges']
        
        model = ProteinMPNN(node_features=args.hidden_dim, 
                            edge_features=args.hidden_dim, 
                            hidden_dim=args.hidden_dim, 
                            num_encoder_layers=args.num_encoder_layers, 
                            num_decoder_layers=args.num_decoder_layers, 
                            k_neighbors=num_edges, 
                            dropout=0.0, 
                            augment_eps=0.0,
                            vocab=21,
                            num_letters=21,
                            use_ipmp=args.use_ipmp,
                            n_points=args.n_points,)

        model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    # run ProteinMPNN logits prediction

    model_name = args.model
    ds_name = cfg.data.dataset
    if args.transfer:
        results = run_prediction_batched(model_name, model, ds_name, ds, [], True, zero_shot=False, cfg=cfg)

    else:
        results = run_prediction_batched(model_name, model, ds_name, ds, [], True, zero_shot=True, cfg=cfg)

    df = pd.DataFrame(results)
    print(df)
    df.to_csv(f"ThermoMPNN_{os.path.basename(model_name).removesuffix('.ckpt')}_{ds_name}_metrics.csv")

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='path to model weights file (pt file)', type=str, default='checkpoint.ckpt')
    parser.add_argument('--local', help='local config file for data storage locations', type=str, default='../local.yaml')
    parser.add_argument('--config', help='local config file for param info', type=str, default='../config.yaml')
    parser.add_argument('--dec_order', help='decoding order to pass to the zero-shot model', type=str, default='autoreg')
    parser.add_argument('--hidden_dim', help='model hidden dim', type=int, default=128)
    parser.add_argument('--num_encoder_layers', help='encoder layers', type=int, default=3)
    parser.add_argument('--num_decoder_layers', help='decoder layers', type=int, default=3)
    parser.add_argument('--use_ipmp', help='enable ipmp encoder/decoder', type=bool, default=False)
    parser.add_argument('--n_points', help='N points for ipmp', type=int, default=8)
    parser.add_argument('--single_res_rec', type=bool, default=False)
    parser.add_argument('--side_chains', type=bool, default=False)
    parser.add_argument('--transfer', type=bool, default=False)
    # parser.add_argument('--epi', help='use epistatic decoding?', default=False, action='store_true', type=bool)

    args = parser.parse_args()
    print(args)

    cfg = OmegaConf.merge(OmegaConf.load(args.local), OmegaConf.load(args.config))
    with torch.no_grad():
        inference(cfg, args)