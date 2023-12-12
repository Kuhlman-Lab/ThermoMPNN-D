import torch
from torch import nn
from torchmetrics import R2Score, MeanSquaredError, SpearmanCorrCoef, PearsonCorrCoef

import argparse
from omegaconf import OmegaConf
import pandas as pd

from thermompnn.model.modules import get_protein_mpnn
from thermompnn.protein_mpnn_utils import tied_featurize
from thermompnn.inference.v1_inference import load_v1_dataset, run_prediction_default, run_prediction_keep_preds
from thermompnn.inference.v2_inference import load_v2_dataset, run_prediction_batched
from thermompnn.inference.siamese_inference import load_siamese_dataset, run_prediction_siamese

from thermompnn.trainer.v1_trainer import TransferModelPL
from thermompnn.trainer.v2_trainer import TransferModelPLv2
from thermompnn.trainer.siamese_trainer import TransferModelSiamesePL


def inference(cfg, args):
    """Catch-all inference function for all ThermoMPNN versions"""

    # pre-initialization params
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    ds_name = cfg.dataset
    model_name = cfg.name 
    
    if cfg.version == 'v1':
        # load specified dataset
        ds = load_v1_dataset(cfg)
        # load model weights
        model = TransferModelPL.load_from_checkpoint(args.model, cfg=cfg, map_location=device).model
        # run inference function 
        if args.keep_preds:
            print('Keeping preds!')
            results = run_prediction_keep_preds(model_name, model, ds_name, ds, [], centrality=args.centrality)
        else:
            results = run_prediction_default(model_name, model, ds_name, ds, [])


    elif cfg.version == 'v2':
        ds = load_v2_dataset(cfg)
        model = TransferModelPLv2.load_from_checkpoint(args.model, cfg=cfg, map_location=device).model
        results = run_prediction_batched(model_name, model, ds_name, ds, [], False)

    elif cfg.version == 'siamese':
        ds = load_siamese_dataset(cfg)
        model = TransferModelSiamesePL.load_from_checkpoint(args.model, cfg=cfg, map_location=device).model
        results = run_prediction_siamese(model_name, model, ds_name, ds, [], keep=False, use_both=True)

    else:
        raise ValueError("Invalid ThermoMPNN version specified. Options are v1, v2, siamese.")

    model = model.eval()
    model = model.cuda()

    df = pd.DataFrame(results)
    print(df)
    df.to_csv(f"ThermoMPNN_{cfg.version}_{ds_name}_metrics.csv")

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file for model and dataset params', type=str, default='../config.yaml')
    parser.add_argument('--model', help='path to model weights file (pt file)', type=str, default='checkpoint.ckpt')
    parser.add_argument('--local', help='local config file for data storage locations', type=str, default='../local.yaml')
    parser.add_argument('--keep_preds', action='store_true', default=False, 
                        help='Save raw model predictions as csv (currently supported for v1 only)')
    parser.add_argument('--centrality', action='store_true', default=False,
                        help='Calculate centrality value for each residue (# neighbors). Only used if --keep_preds is enabled.')

    args = parser.parse_args()

    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.load(args.local))
    with torch.no_grad():
        inference(cfg, args)