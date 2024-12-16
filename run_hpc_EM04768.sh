#!/usr/bin/env bash
#SBATCH --job-name=multiview_stitcher_test
#SBATCH --part=ncpu
#SBATCH --cpus-per-task=16
#SBATCH --time=240          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --mem=64G   # Memory pool for all cores (see also --mem-per-cpu)

export PYTHONUNBUFFERED=TRUE
ml purge
ml Anaconda3
source /camp/apps/eb/software/Anaconda/conda.env.sh
conda activate multiview-stitcher-source-env
python run.py --params=resources/params_EM04768_01_Sub_04_Cell_1_10nm_10us.yml
python run.py --params=resources/params_EM04768_01_Sub_04_Cell_1_20nm_5us.yml