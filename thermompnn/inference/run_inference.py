import torch
import argparse
from omegaconf import OmegaConf
import pandas as pd
import os

from thermompnn.inference.v2_inference import load_v2_dataset, run_prediction_batched

from thermompnn.trainer.v2_trainer import TransferModelPLv2, TransferModelPLv2Siamese


def inference(cfg, args):
    """Catch-all inference function for all ThermoMPNN versions"""

    # pre-initialization params
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    ds_name = cfg.data.dataset
    model_name = args.model.removesuffix('.ckpt')
    
    ds = load_v2_dataset(cfg)
    print('Loading model %s' % args.model)
    if cfg.model.aggregation == 'siamese':
        model = TransferModelPLv2Siamese.load_from_checkpoint(args.model, cfg=cfg, map_location=device).model
    else:
        # cfg.data.mut_types = ['insertion', 'deletion']
        cfg.data.mut_types = ['single', 'insertion', 'deletion'] # TODO override to load proper model dims
        model = TransferModelPLv2.load_from_checkpoint(args.model, cfg=cfg, map_location=device).model
    results = run_prediction_batched(model_name, model, ds_name, ds, [], args.keep_preds, cfg=cfg)

    df = pd.DataFrame(results)
    print(df)
    df.to_csv(f"ThermoMPNN_{os.path.basename(model_name).removesuffix('.ckpt')}_{ds_name}_metrics.csv")

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
    parser.add_argument('--cqr', action='store_true', default=False, 
                        help='whether to use conformal inference or not (default=False)')

    args = parser.parse_args()

    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.load(args.local))
    from train_thermompnn import parse_cfg
    cfg = parse_cfg(cfg)

    with torch.no_grad():
        inference(cfg, args)