#!/bin/bash
#PBS -q gpuq_V100_32
#PBS -N suzuki_11
#PBS -l select=1:ngpus=1:host=bdgpu11
cd $PBS_O_WORKDIR
~/miniconda3/envs/PredictionReactionYield/bin/python ~/For_Github/BuchwaldHartwig/main.py --sample_num 3000000 --seed 97 --extrapolation True --target suzuki-miyaura
