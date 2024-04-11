#!/bin/bash

# loop over all PDBs in a dir

loc="/proj/kuhl_lab/users/dieckhau/ThermoMPNN/data/binder-SSM/parents"

out="/proj/kuhl_lab/users/dieckhau/ThermoMPNN/data/binder-SSM/dssp"

mkdir $out
cd $loc

py="/nas/longleaf/home/dieckhau/miniconda3/envs/proteinMPNN/bin/python"
script="/proj/kuhl_lab/users/dieckhau/ThermoMPNN/file_utils/sasa.py"

for f in *.pdb; do
	echo $f;
	# do complex dssp calculation
	pkl="$out/${f::-4}_c.pkl"
	$py $script -i $f -o $pkl

	# extract binder chain and re-run dssp
	pdb_selchain -A $f > ../BINDER.pdb
	pkl="$out/${f::-4}_m.pkl"
	$py $script -i ../BINDER.pdb -o $pkl
done;


