#!/bin/bash -x
#SBATCH --account="haf"
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --output=hyperop-out.%j
#SBATCH --error=hyperop-err.%j
#SBATCH --time=24:00:00
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
module use $OTHERSTAGES
module load Stages/2020
module load CUDA/11
module load GCC
module load ParaStationMPI
module load Python
module load SciPy-Stack
module load mpi4py
module load PyTorch
srun python -u $PROJECT/users/funk1/src/shared/scripts/nn_training/hyperop/example.py