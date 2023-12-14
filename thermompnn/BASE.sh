source ~/.bashrc

export CUDA_VISIBLE_DEVICES=7
echo "Running script on GPU $CUDA_VISIBLE_DEVICES"

conda activate thermoMPNN

# train model
python train_thermompnn.py ../wout.yaml BASE.yaml

# benchmark model
# python inference/run_inference.py --config test.yaml --model checkpoints/DEBUG.ckpt --local ../wout.yaml
