## This is the official implementation of ThermoMPNN-D, a Siamese neural network designed to predict stability changes from protein double point mutations.

[Picture Here - TODO]

This work is an extension of ThermoMPNN (https://github.com/Kuhlman-Lab/ThermoMPNN), which is itself an extension of ProteinMPNN (https://github.com/dauparas/ProteinMPNN). For details, see our manuscript [here](https://www.biorxiv.org/content/10.1101/2024.08.20.608844v1).


### Installation

First, close the repository:
```
git clone https://github.com/Kuhlman-Lab/ThermoMPNN-D.git
cd ThermoMPNN-D
```


Then, install the python dependencies found in ```environment.yaml``` (I recommend mamba):
```
mamba env create -f environment.yaml -n ThermoMPNN-D
```

Add ThermoMPNN to your ```PYTHONPATH``` so that python can find all the modules: 
```
export PYTHONPATH=$PYTHONPATH:/path/to/ThermoMPNN-D
```

Finally, modify the local filepath information found in ```ThermoMPNN/examples/configs/local.yaml``` to match your system.

Before running any ThermoMPNN-D scripts, just run ```mamba activate ThermoMPNN-D``` to load the necessary python packages.

### Inference

We provide a script called ```v2_ssm.py``` which does inference on all possible single or double mutants in the protein. The output for this script is a CSV file with mutation and ddG values.

#### Options

There is an important option called ```--threshold``` which dictates which mutations will get saved to disk. By default, ThermoMPNN will only save stabilizing mutations (ddG < -0.5 kcal/mol), since this is fastest for saving to disk. To get all the mutations, including destabilizing mutations, set --threshold very high (e.g., 100).

The other useful option is ```--distance``` which is used for additive or epistatic predictions. This is the distance threshold used to filter for "nearby" residues that are likely to have epistatic interactions. A smaller value will lead to stricter filtering.

#### Single mutant model
This is an updated version of single mutant ThermoMPNN that uses fewer parameters and proper batched inference for faster prediction.

To run this on a custom pdb, use the following command:
```python v2_ssm.py --mode single --pdb 1vii.pdb --batch_size 256 --out 1vii```

#### Additive double mutant model
This sums the individual contributions from each single mutation without attempting to quantify epistatic coupling terms. Inference is faster than with the epistatic model since it just needs to add the terms rather than predict each one separately.

To run this model, use the following command:
```python v2_ssm.py --mode additive --pdb 1vii.pdb --batch_size 256 --out 1vii```

#### Epistatic double mutant model
This model attempts to capture epistatic interactions between double mutations, which requires running inference on every individual mutation. This is slower than the single or additive model but is still reasonably fast (<1 minute) due to some vectorizing and batching tricks.

To run this model, use the following command:
```python v2_ssm.py --mode additive --pdb 1vii.pdb --batch_size 2048 --out 1vii```

Note the higher batch size, which takes advantage of the lightweight prediction head to significantly speed up inference.

### Benchmarking
To repeat the benchmarks in the ThermoMPNN-D paper, see the ```benchmarks.ipynb``` notebook. 

The datasets used in this study can be obtained from https://zenodo.org/records/13345274. Note that you may need to slightly modify the column names and/or filepaths to line up with what the benchmark script expects.

### Training

Training requires proper CUDA drivers and an accessible GPU. Single mutant epochs should take 2-3 minutes on a V100 GPU, while epochs for epistatic models take a bit longer (8-10 minutes) due to data augmentation which provides a larger dataset. Training typically converges in 30-40 epochs. 

#### Single mutant (aka Additive) model

```python train_thermompnn.py ../examples/configs/local.yaml ../examples/configs/train_single.yaml```

### Double mutant model

```python train_thermompnn.py ../examples/configs/local.yaml ../examples/configs/train_epistatic.yaml```

Metric curves can be logged using W&B if desired - simply un-comment the ```Project``` and ```name``` fields in ```train.yaml``` and hook up your W&B account.

#### License

This work is made available under an MIT license (see LICENSE file for details).

#### Citation

If this work is useful to you, please use the following citation:
```
@article {Dieckhaus2024.08.20.608844,
	author = {Dieckhaus, Henry and Kuhlman, Brian},
	title = {Protein stability models fail to capture epistatic interactions of double point mutations},
    journal = {bioRxiv},
	elocation-id = {2024.08.20.608844},
	year = {2024},
	doi = {10.1101/2024.08.20.608844},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/08/21/2024.08.20.608844},
	eprint = {https://www.biorxiv.org/content/early/2024/08/21/2024.08.20.608844.full.pdf},
}
```
