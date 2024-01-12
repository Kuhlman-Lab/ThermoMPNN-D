# source ~/.bashrc

# export CUDA_VISIBLE_DEVICES=7
# echo "Running script on GPU $CUDA_VISIBLE_DEVICES"

# conda activate thermoMPNN

# train model
# python train_thermompnn.py ../wout.yaml ../DEBUG.yaml

# benchmark model
# python inference/run_inference.py --config test.yaml --model ./checkpoints/baselineLR.ckpt --local ../wout.yaml

# python inference/run_inference.py --config test.yaml --model ./checkpoints/EXP2.4.de0.ckpt --local ../local.yaml
# python inference/run_inference.py --config test.yaml --model ./checkpoints/EXP2.4.de1.ckpt --local ../local.yaml
# python inference/run_inference.py --config test.yaml --model ./checkpoints/EXP2.4.de2.ckpt --local ../local.yaml
# python inference/run_inference.py --config test.yaml --model ./checkpoints/EXP2.4.de3.ckpt --local ../local.yaml
# python inference/run_inference.py --config test.yaml --model ./checkpoints/EXP2.4.de4.ckpt --local ../local.yaml

python inference/run_inference.py --config test.yaml --model ./checkpoints/EXP2.4.rep0.ckpt --local ../local.yaml
python inference/run_inference.py --config test.yaml --model ./checkpoints/EXP2.4.rep1.ckpt --local ../local.yaml
python inference/run_inference.py --config test.yaml --model ./checkpoints/EXP2.4.rep2.ckpt --local ../local.yaml
python inference/run_inference.py --config test.yaml --model ./checkpoints/EXP2.4.rep3.ckpt --local ../local.yaml
python inference/run_inference.py --config test.yaml --model ./checkpoints/EXP2.4.rep4.ckpt --local ../local.yaml


# python inference/run_inference.py --config test.yaml --model "../checkpoints/B_submut_extra_epoch=81_val_ddG_spearman=0.79.ckpt" --local ../local.yaml

# python inference/run_inference.py --config test.yaml --model "checkpoints/baselineLR_wout_epoch=78_val_ddG_spearman=0.79.ckpt" --local ../wout.yaml
