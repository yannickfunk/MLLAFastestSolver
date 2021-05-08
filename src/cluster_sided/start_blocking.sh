#!/bin/bash -x
#SBATCH --account="haf"
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --output=blocking-out.%j
#SBATCH --error=blocking-err.%j
#SBATCH --time=24:00:00
#SBATCH --partition=batch
#SBATCH --gres=gpu:0
module use $OTHERSTAGES
module load Stages/2020
module load CUDA/11
module load GCC
module load MVAPICH2
module load ParaStationMPI
module load Python
module load SciPy-Stack
module load mpi4py
export HOME=$PROJECT/users/funk1
pip install tqdm --user
srun python -u $PROJECT/users/funk1/src/shared/scripts/blocking.py
