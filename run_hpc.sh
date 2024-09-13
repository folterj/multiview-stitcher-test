#!/usr/bin/env bash
#SBATCH --job-name=multiview_stitcher_test
#SBATCH --part=ncpu
#SBATCH --cpus-per-task=16
#SBATCH --time=480          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --mem=64G   # Memory pool for all cores (see also --mem-per-cpu)

export PYTHONUNBUFFERED=TRUE
ml purge
ml Anaconda3
source /camp/apps/eb/software/Anaconda/conda.env.sh
conda activate multiview-stitcher-cont-env
python run.py