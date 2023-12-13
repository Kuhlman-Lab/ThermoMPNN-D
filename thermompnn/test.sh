# source ~/.bashrc

# export CUDA_VISIBLE_DEVICES=7
# echo "Running script on GPU $CUDA_VISIBLE_DEVICES"

# conda activate thermoMPNN

# train model
# python train_thermompnn.py ../wout.yaml ../DEBUG.yaml

# benchmark model
python inference/run_inference.py --config test.yaml --model ./checkpoints/baselineLR.ckpt --local ../local.yaml

# python inference/run_inference.py --config test.yaml --model "../checkpoints/B_submut_extra_epoch=81_val_ddG_spearman=0.79.ckpt" --local ../local.yaml
