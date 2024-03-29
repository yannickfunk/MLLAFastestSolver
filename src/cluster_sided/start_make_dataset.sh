#!/bin/bash -x
#SBATCH --account="haf"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=make_dataset-out.%j
#SBATCH --error=make_dataset-err.%j
#SBATCH --time=24:00:00
#SBATCH --partition=batch
#SBATCH --gres=gpu:0
module load Python
module load SciPy-Stack
export HOME=$PROJECT/users/funk1
srun python -u $PROJECT/users/funk1/src/shared/scripts/make_dataset.py
