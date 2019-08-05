#!/bin/bash

# Print arguments
echo "Arguments"
dim=$1
dropout=$2
heads=$3

echo 'dim = ' $dim # dimensionality
echo 'dropout = ' $dropout # dropout
echo 'heads = ' $heads # heads

# Compute and print depth
depth=$((256/$dim))
echo 'depth = ' $depth

output_dir=/mnt/results/mimic3/phenotyping
echo 'output_dir = ' $output_dir

data=/mnt/data/mimic3/phenotyping
echo 'data = ' $data

# Run discretized script
python -um mimic3models.phenotyping.main --network mimic3models/keras_models/transformer.py --dim $dim --timestep 1.0 --depth $depth --dropout $dropout --mode train --batch_size 8 --num_heads $heads --output_dir $output_dir --verbose 1 --epochs 50 --max_seq_len 2000 --mask_value 0. --seed 0 --data $data

# Run time false script
python -um mimic3models.phenotyping.main_transformer --network mimic3models/keras_models/transformer.py --dim $dim --depth $depth --dropout $dropout --mode train --batch_size 8 --num_heads $heads --output_dir $output_dir --verbose 1 --epochs 50 --max_seq_len 2000 --mask_value 0. --seed 0 --data $data

# Run time true script
python -um mimic3models.phenotyping.main_transformer --network mimic3models/keras_models/transformer.py --dim $dim --depth $depth --dropout $dropout --mode train --batch_size 8 --num_heads $heads --output_dir $output_dir --use_time True --verbose 1 --epochs 50 --max_seq_len 2000 --mask_value 0. --seed 0 --data $data
