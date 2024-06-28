import torch
import argparse
from omegaconf import OmegaConf
import pandas as pd
import os

from thermompnn.inference.v1_inference import load_v1_dataset, run_prediction_default, run_prediction_keep_preds
from thermompnn.inference.v2_inference import load_v2_dataset, run_prediction_batched, load_conf_dataset, run_prediction_conf
from thermompnn.inference.siamese_inference import load_siamese_dataset, run_prediction_siamese

from thermompnn.trainer.v1_trainer import TransferModelPL
from thermompnn.trainer.v2_trainer import TransferModelPLv2, TransferModelPLv2Siamese
from thermompnn.trainer.siamese_trainer import TransferModelSiamesePL


def inference(cfg, args):
    """Catch-all inference function for all ThermoMPNN versions"""

    # pre-initialization params
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    ds_name = cfg.data.dataset
    model_name = args.model.removesuffix('.ckpt')
    
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
        print('Loading model %s' % args.model)
        if cfg.model.aggregation == 'siamese':
            model = TransferModelPLv2Siamese.load_from_checkpoint(args.model, cfg=cfg, map_location=device, train_dataset=ds, val_dataset=ds).model
        else:
            model = TransferModelPLv2.load_from_checkpoint(args.model, cfg=cfg, map_location=device, train_dataset=ds, val_dataset=ds).model
        results = run_prediction_batched(model_name, model, ds_name, ds, [], args.keep_preds, cfg=cfg)

    # elif cfg.version == 'conf':
    #     ds1, ds2 = load_conf_dataset(cfg) # TODO load calibration set AND test set
    #     model = TransferModelPLv2.load_from_checkpoint(args.model, cfg=cfg, map_location=device, train_dataset=ds1, val_dataset=ds2).model
    #     results = run_prediction_conf(model_name, ds_name, ds1, ds2, [], args.keep_preds, cfg=cfg)

    elif cfg.version == 'siamese':
        ds = load_siamese_dataset(cfg)
        model = TransferModelSiamesePL.load_from_checkpoint(args.model, cfg=cfg, map_location=device).model
        results = run_prediction_siamese(model_name, model, ds_name, ds, [], keep=False, use_both=True)

    else:
        raise ValueError("Invalid ThermoMPNN version specified. Options are v1, v2, conf, siamese.")

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