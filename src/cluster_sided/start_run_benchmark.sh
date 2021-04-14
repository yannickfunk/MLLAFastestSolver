#!/bin/bash -x
#SBATCH --account="haf"
#SBATCH --nodes=7
#SBATCH --ntasks=7
#SBATCH --output=run_benchmark-out.%j
#SBATCH --error=run_benchmark-err.%j
#SBATCH --time=24:00:00
#SBATCH --partition=batch
#SBATCH --gres=gpu:2
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
srun python $PROJECT/users/funk1/src/shared/scripts/run_benchmark.py
