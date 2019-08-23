#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=120G
#SBATCH --time=0-10:00:00
#SBATCH --output vectors%a.out
#SBATCH -p msps
module load Python/2.7.12-goolf-2015a
python BEAGLE_HOLO.py "train" "TASA_RP_spat" "TASA_RP_spat" ${SLURM_ARRAY_TASK_ID} 32

