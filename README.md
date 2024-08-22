## This is the official implementation of ThermoMPNN-D, a Siamese neural network designed to predict stability changes from protein double point mutations.

[Picture Here - TODO]

This work is an extension of ThermoMPNN (https://github.com/Kuhlman-Lab/ThermoMPNN), which is itself an extension of ProteinMPNN (https://github.com/dauparas/ProteinMPNN).

[Preprint reference here - TODO]

### Installation

First, close the repository:
```git clone https://github.com/Kuhlman-Lab/ThermoMPNN.git```

Then, install the python dependencies found in ```environment.yaml``` (I recommend mamba):
```mamba env create -f environment.yaml```

Add ThermoMPNN to your ```PYTHONPATH``` so that python can find all the modules: 
```export PYTHONPATH=$PYTHONPATH:/path/to/ThermoMPNN```

Finally, modify the local filepath information found in ```ThermoMPNN/examples/configs/local.yaml``` to match your system.

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

Training requires proper CUDA drivers and an accessible GPU.

#### Single mutant model

```python ```
[TODO]

#### License

This work is made available under an MIT license (see LICENSE file for details).

#### Citation

If this work is useful to you, please use the following citation:
[TODO]