#!/bin/bash -x
#SBATCH --account="haf"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=eval-out.%j
#SBATCH --error=eval-err.%j
#SBATCH --time=24:00:00
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
module load Python
module load SciPy-Stack
module load GCCcore
srun python -u $PROJECT/users/funk1/src/shared/scripts/nn_training/eval.py