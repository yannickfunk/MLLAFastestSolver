#!/bin/bash -x
#SBATCH --account="haf"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=collect_filepaths-out.%j
#SBATCH --error=collect_filepaths-err.%j
#SBATCH --time=10:00:00
#SBATCH --partition=batch
#SBATCH --gres=gpu:2
module load Python
module load SciPy-Stack
export HOME=$PROJECT/data/funk1
srun python $PROJECT/users/funk1/src/shared/scripts/collect_filepaths.py
