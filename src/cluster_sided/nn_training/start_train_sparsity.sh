#!/bin/bash -x
#SBATCH --account="haf"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=train_sparsity-out.%j
#SBATCH --error=train_sparsity-err.%j
#SBATCH --time=24:00:00
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
module load Python
module load Keras
module load Scipy-Stack
srun python -u $PROJECT/users/funk1/src/shared/scripts/nn_training/train_sparsity.py