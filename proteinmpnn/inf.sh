source ~/.bashrc

conda activate thermoMPNN

export CUDA_VISIBLE_DEVICES=7
echo "Running script on GPU $CUDA_VISIBLE_DEVICES"

cd /home/hdieckhaus/scripts/ThermoMPNN/proteinmpnn/

python testing.py \
	--path_for_training_data /home/hdieckhaus/datasets/pdb_2021aug02 \
	--ckpt_path ProteinMPNN_100_3/model_weights/epoch_last.pt \
	--num_repeats 1 \
	--temperature 0.00001 \
	--seed 1234 \
	--num_examples_per_epoch 100000 \
	--num_decoder_layers 3

# python testing.py \
# 	--path_for_training_data /home/hdieckhaus/datasets/pdb_2021aug02 \
# 	--ckpt_path ProteinMPNN_002_1_SRR_SCA/model_weights/epoch_last.pt \
# 	--num_repeats 1 \
# 	--temperature 0.00001 \
# 	--seed 1234 \
# 	--num_examples_per_epoch 100000 \
# 	--num_decoder_layers 1
	# --single_res_rec True \
	# --side_chains True

