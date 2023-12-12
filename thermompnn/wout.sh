source ~/.bashrc

export CUDA_VISIBLE_DEVICES=7
echo "Running script on GPU $CUDA_VISIBLE_DEVICES"

conda activate thermoMPNN

# train model
# python train_thermompnn.py ../wout.yaml ../DEBUG.yaml

# benchmark model
python inference/inference_utils.py --config ../DEBUG.yaml --model ../m
