#!/bin/bash
#SBATCH --account=p31340
#SBATCH --partition=normal
#SBATCH --time=48:00:00
#SBATCH --mem=10G
#SBATCH --job-name=posts.py
#SBATCH --output=outlog
#SBATCH --error=errlog

cd /kellogg/proj/dwa382/Duomai/
module purge all
module load python-anaconda3
source activate video_env_1
conda deactivate
source activate video_env_1
python posts.py