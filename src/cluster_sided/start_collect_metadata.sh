#!/bin/bash -x
#SBATCH --account="haf"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=collect_metadata-out.%j
#SBATCH --error=collect_metadata-err.%j
#SBATCH --time=00:15:00
#SBATCH --partition=batch
#SBATCH --gres=gpu:2
module load Python
module load SciPy-Stack
srun python $PROJECT/users/funk1/src/shared/scripts/collect_metadata.py